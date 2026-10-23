#include "model_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "tensor.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define BM_HAVE_MMAP 1
#endif

namespace bm {
namespace {

constexpr size_t kHeaderBytes = 64;

struct Reader {  // bounds-checked cursor over the file image
  const uint8_t* base;
  size_t size, pos = kHeaderBytes;

  const void* take(size_t bytes) {
    pos = round_up(pos);  // every tensor starts on a 64-byte boundary
    if (pos + bytes > size) throw std::runtime_error("model file is truncated");
    const void* p = base + pos;
    pos += bytes;
    return p;
  }
  const float* f32(size_t n) { return static_cast<const float*>(take(n * 4)); }

  Matrix matrix(int rows, int cols, bool int8) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    if (int8) {
      m.scales = f32(rows);
      m.q8 = static_cast<const int8_t*>(take(static_cast<size_t>(rows) * cols));
    } else {
      m.f32 = f32(static_cast<size_t>(rows) * cols);
    }
    return m;
  }
};

template <typename T>
T read_at(const uint8_t* p, size_t off) {
  T v;
  std::memcpy(&v, p + off, sizeof(T));
  return v;
}

}  // namespace

ModelFile::ModelFile(const std::string& path) {
#ifdef BM_HAVE_MMAP
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st;
  if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("cannot stat " + path); }
  size_ = static_cast<size_t>(st.st_size);
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
  base_ = static_cast<const uint8_t*>(p);
  mapped_ = true;
#else
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::fseek(f, 0, SEEK_END);
  size_ = static_cast<size_t>(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  void* buf = std::aligned_alloc(kAlign, round_up(size_));
  if (!buf || std::fread(buf, 1, size_, f) != size_) { std::fclose(f); std::free(buf); throw std::runtime_error("read failed"); }
  std::fclose(f);
  base_ = static_cast<const uint8_t*>(buf);
#endif

  try {
    if (size_ < kHeaderBytes || std::memcmp(base_, "BMTF", 4) != 0) throw std::runtime_error("not a BMTF model file");
    if (read_at<uint32_t>(base_, 4) != 1) throw std::runtime_error("unsupported BMTF version");
    int8 = read_at<uint32_t>(base_, 8) == 1;
    cfg.vocab = static_cast<int>(read_at<uint32_t>(base_, 12));
    cfg.d_model = static_cast<int>(read_at<uint32_t>(base_, 16));
    cfg.n_layers = static_cast<int>(read_at<uint32_t>(base_, 20));
    cfg.n_heads = static_cast<int>(read_at<uint32_t>(base_, 24));
    cfg.d_ff = static_cast<int>(read_at<uint32_t>(base_, 28));
    cfg.max_seq = static_cast<int>(read_at<uint32_t>(base_, 32));
    cfg.rope_theta = read_at<float>(base_, 36);
    cfg.norm_eps = read_at<float>(base_, 40);
    if (cfg.n_heads <= 0 || cfg.d_model % cfg.n_heads != 0 || cfg.head_dim() % 2 != 0)
      throw std::runtime_error("invalid model dimensions");

    Reader r{base_, size_};
    const int d = cfg.d_model, ff = cfg.d_ff;
    emb = r.matrix(cfg.vocab, d, int8);
    layers.resize(cfg.n_layers);
    for (auto& L : layers) {  // same order as export_weights.tensor_names()
      L.attn_norm = r.f32(d);
      L.wq = r.matrix(d, d, int8);
      L.wk = r.matrix(d, d, int8);
      L.wv = r.matrix(d, d, int8);
      L.wo = r.matrix(d, d, int8);
      L.mlp_norm = r.f32(d);
      L.w1 = r.matrix(ff, d, int8);
      L.w2 = r.matrix(d, ff, int8);
    }
    final_norm = r.f32(d);
  } catch (...) {
    release();
    throw;
  }
}

ModelFile::~ModelFile() { release(); }

void ModelFile::release() {
  if (!base_) return;
#ifdef BM_HAVE_MMAP
  if (mapped_) ::munmap(const_cast<uint8_t*>(base_), size_);
#else
  std::free(const_cast<uint8_t*>(base_));
#endif
  base_ = nullptr;
}

Reference load_reference(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[4];
  uint32_t n = 0, v = 0;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "BMRF", 4) != 0 ||
      std::fread(&n, 4, 1, f) != 1 || std::fread(&v, 4, 1, f) != 1) {
    std::fclose(f);
    throw std::runtime_error("not a BMRF reference file");
  }
  Reference ref;
  ref.vocab = static_cast<int>(v);
  ref.ids.resize(n);
  ref.logits_fp32.resize(static_cast<size_t>(n) * v);
  ref.logits_int8.resize(static_cast<size_t>(n) * v);
  bool ok = std::fread(ref.ids.data(), 4, n, f) == n &&
            std::fread(ref.logits_fp32.data(), 4, ref.logits_fp32.size(), f) == ref.logits_fp32.size() &&
            std::fread(ref.logits_int8.data(), 4, ref.logits_int8.size(), f) == ref.logits_int8.size();
  std::fclose(f);
  if (!ok) throw std::runtime_error("reference file is truncated");
  return ref;
}

}  // namespace bm
