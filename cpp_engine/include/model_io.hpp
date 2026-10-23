// Loads the BMTF binary format written by python_scratch/export_weights.py.
// The file is mmap'd read-only: weights are used in place (no copy), the OS pages them in
// lazily, and 64-byte tensor alignment in the file survives because mmap is page aligned.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bm {

struct ModelConfig {
  int vocab = 0, d_model = 0, n_layers = 0, n_heads = 0, d_ff = 0, max_seq = 0;
  float rope_theta = 10000.f, norm_eps = 1e-5f;
  int head_dim() const { return d_model / n_heads; }
};

// A weight matrix stored [rows = out, cols = in]; either FP32 or INT8 + per-row scales.
struct Matrix {
  const float* f32 = nullptr;
  const int8_t* q8 = nullptr;
  const float* scales = nullptr;
  int rows = 0, cols = 0;
  bool quantized() const { return q8 != nullptr; }
};

struct LayerWeights {
  const float* attn_norm = nullptr;
  Matrix wq, wk, wv, wo;
  const float* mlp_norm = nullptr;
  Matrix w1, w2;
};

class ModelFile {
 public:
  explicit ModelFile(const std::string& path);  // throws std::runtime_error on a bad file
  ~ModelFile();
  ModelFile(const ModelFile&) = delete;
  ModelFile& operator=(const ModelFile&) = delete;

  ModelConfig cfg;
  bool int8 = false;
  Matrix emb;  // [vocab, d_model]: token lookup rows and the tied output projection
  std::vector<LayerWeights> layers;
  const float* final_norm = nullptr;

  size_t file_bytes() const { return size_; }

 private:
  void release();
  const uint8_t* base_ = nullptr;
  size_t size_ = 0;
  bool mapped_ = false;
};

// Reference data written next to a model by the exporter, used by the tests. Each variant holds the
// logits [n, vocab] PyTorch produced under one numeric mode (order = export_weights.REFERENCE_VARIANTS).
enum RefVariant { kRefFp32 = 0, kRefW8, kRefW8A8, kRefW8Kv8, kRefFp32Kv8, kRefVariants };
struct Reference {
  std::vector<int32_t> ids;
  int vocab = 0;
  std::vector<std::vector<float>> logits;  // [kRefVariants][n * vocab]
};
Reference load_reference(const std::string& path);

}  // namespace bm
