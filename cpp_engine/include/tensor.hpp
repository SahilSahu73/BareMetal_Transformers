// Memory model of the engine: one 64-byte aligned arena, non-owning tensor views on top.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace bm {

constexpr size_t kAlign = 64;  // one cache line; also satisfies aligned 256-bit SIMD loads

constexpr size_t round_up(size_t n, size_t a = kAlign) { return (n + a - 1) / a * a; }

// Bump allocator over a single aligned block. Everything the forward pass needs
// (activations, KV-cache, RoPE tables) is carved out of it up front, so generating
// tokens never touches malloc.
class Arena {
 public:
  explicit Arena(size_t capacity_bytes) : capacity_(round_up(capacity_bytes)) {
    base_ = static_cast<uint8_t*>(std::aligned_alloc(kAlign, capacity_));
    if (!base_) throw std::bad_alloc();
    std::memset(base_, 0, capacity_);
  }
  ~Arena() { std::free(base_); }
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  void* alloc(size_t bytes) {
    size_t start = round_up(used_);
    if (start + bytes > capacity_) throw std::bad_alloc();
    used_ = start + bytes;
    return base_ + start;
  }
  float* alloc_f32(size_t n) { return static_cast<float*>(alloc(n * sizeof(float))); }

  size_t used() const { return used_; }
  size_t capacity() const { return capacity_; }

 private:
  uint8_t* base_ = nullptr;
  size_t capacity_ = 0, used_ = 0;
};

// Non-owning 2-D row-major view: a pointer plus shape and row stride. Copying it is free
// and slicing never copies data, e.g. one attention head's columns of the KV-cache.
struct Tensor {
  float* data = nullptr;
  int rows = 0, cols = 0, stride = 0;

  Tensor() = default;
  Tensor(float* d, int r, int c) : data(d), rows(r), cols(c), stride(c) {}
  Tensor(float* d, int r, int c, int s) : data(d), rows(r), cols(c), stride(s) {}

  float* row(int i) const { return data + static_cast<size_t>(i) * stride; }
  float& operator()(int i, int j) const { return row(i)[j]; }
  Tensor slice_cols(int j0, int n) const { return Tensor(data + j0, rows, n, stride); }
  Tensor slice_rows(int i0, int n) const { return Tensor(row(i0), n, cols, stride); }
};

}  // namespace bm
