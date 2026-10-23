// Decoder-only transformer forward pass for autoregressive generation, with a KV-cache.
#pragma once
#include <cstdint>
#include <vector>

#include "kernels.hpp"
#include "model_io.hpp"
#include "tensor.hpp"

namespace bm {

// How activations meet INT8 weights (ignored for FP32 model files).
enum class ActMode {
  F32,  // W8A32: weights int8, activations stay FP32 (the decode default: memory bound anyway)
  Q8,   // W8A8: activations quantised per token as well; integer dot products everywhere
};

struct TransformerOptions {
  KernelTable kernels = naive_kernels();
  ActMode act = ActMode::F32;
  bool kv_int8 = false;  // store K/V as int8 with one scale per (position, head): 4x smaller cache
};

class Transformer {
 public:
  explicit Transformer(const ModelFile& model, KernelTable kernels = naive_kernels());
  Transformer(const ModelFile& model, const TransformerOptions& options);

  // Feeds one token at the current position, appends its K/V to the cache and returns the
  // next-token logits (vocab floats, valid until the next call). Throws past max_seq.
  const float* forward(int token);

  // Processes n prompt tokens at once: every weight matrix is applied to all n tokens with a
  // single GEMM (compute bound) instead of n matrix-vector products (memory bound). Appends the
  // n positions to the KV-cache and returns the logits of the last token, exactly as n calls of
  // forward() would. May follow earlier tokens (the sequence continues from pos()).
  const float* prefill(const int* tokens, int n);

  // Same, but writes the logits of every position to out[n * vocab] (for perplexity evaluation).
  void prefill_all(const int* tokens, int n, float* out);

  void reset() { pos_ = 0; }  // start a new sequence (the cache is simply overwritten)
  int pos() const { return pos_; }
  const ModelConfig& config() const { return cfg_; }
  size_t arena_bytes() const { return arena_.used(); }
  size_t kv_cache_bytes() const { return kv_bytes_; }

 private:
  void linear(const Matrix& w, const float* x, float* y) const;
  void linear_batch(const Matrix& w, const float* X, int m, float* Y) const;  // Y[m, rows] = X[m, cols] W^T
  void embed(int token, float* out) const;
  void store_kv(int layer, int pos, const float* k, const float* v);
  void attention(int layer, int pos, const float* q, float* out);  // q, out: d_model floats
  void prefill_layers(const int* tokens, int n);                   // leaves the final hidden states in px_

  const ModelFile& m_;
  ModelConfig cfg_;
  TransformerOptions opt_;
  KernelTable& kern_;
  Arena arena_;
  int pos_ = 0;
  size_t kv_bytes_ = 0;

  // activations, all carved out of the arena
  float *x_, *xb_, *q_, *k_, *v_, *att_out_, *scores_, *hidden_, *logits_;
  float *px_, *pxb_, *pq_, *pk_, *pv_, *patt_, *phid_;  // prefill: the same buffers, one row per token
  float *rope_cos_, *rope_sin_;                         // [max_seq, head_dim / 2]
  int8_t* xq_;                                          // quantised activations for W8A8 (one row per token)
  float* xqs_;                                          // ... and their scales

  // KV-cache, per layer: FP32 [max_seq, d_model] views, or int8 + per-(position, head) scales
  std::vector<Tensor> kcache_, vcache_;
  std::vector<int8_t*> kq_, vq_;
  std::vector<float*> ks_, vs_;
};

// Temperature / top-k / top-p sampling with a seeded xorshift RNG (reproducible runs).
class Sampler {
 public:
  Sampler(int vocab, float temperature, int top_k, float top_p, uint64_t seed);
  int sample(const float* logits);  // temperature <= 0 means greedy

 private:
  float uniform();
  int vocab_;
  float temperature_, top_p_;
  int top_k_;
  uint64_t state_;
  std::vector<int> idx_;
  std::vector<float> probs_;
};

}  // namespace bm
