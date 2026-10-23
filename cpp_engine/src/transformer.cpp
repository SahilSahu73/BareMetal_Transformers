#include "transformer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace bm {
namespace {

size_t arena_capacity(const ModelConfig& c) {
  const size_t f = sizeof(float);
  size_t b = 0;
  b += 6 * round_up(c.d_model * f);                                          // x, xb, q, k, v, att_out
  b += round_up(static_cast<size_t>(c.n_heads) * c.max_seq * f);             // scores
  b += round_up(c.d_ff * f) + round_up(c.vocab * f);                         // hidden, logits
  b += 6 * round_up(static_cast<size_t>(c.max_seq) * c.d_model * f);         // prefill activations
  b += round_up(static_cast<size_t>(c.max_seq) * c.d_ff * f);
  b += 2 * round_up(static_cast<size_t>(c.max_seq) * (c.head_dim() / 2) * f);  // rope tables
  b += 2 * c.n_layers * round_up(static_cast<size_t>(c.max_seq) * c.d_model * f);  // KV-cache
  return b + 16 * kAlign;  // slack
}

}  // namespace

Transformer::Transformer(const ModelFile& model, KernelTable kernels)
    : m_(model), cfg_(model.cfg), kern_(kernels), arena_(arena_capacity(model.cfg)) {
  const int d = cfg_.d_model, T = cfg_.max_seq, hd = cfg_.head_dim();
  x_ = arena_.alloc_f32(d);
  xb_ = arena_.alloc_f32(d);
  q_ = arena_.alloc_f32(d);
  k_ = arena_.alloc_f32(d);
  v_ = arena_.alloc_f32(d);
  att_out_ = arena_.alloc_f32(d);
  scores_ = arena_.alloc_f32(static_cast<size_t>(cfg_.n_heads) * T);
  hidden_ = arena_.alloc_f32(cfg_.d_ff);
  logits_ = arena_.alloc_f32(cfg_.vocab);

  const size_t rows = static_cast<size_t>(T);
  px_ = arena_.alloc_f32(rows * d);
  pxb_ = arena_.alloc_f32(rows * d);
  pq_ = arena_.alloc_f32(rows * d);
  pk_ = arena_.alloc_f32(rows * d);
  pv_ = arena_.alloc_f32(rows * d);
  patt_ = arena_.alloc_f32(rows * d);
  phid_ = arena_.alloc_f32(rows * cfg_.d_ff);

  // angle(p, i) = p * theta^(-2i/head_dim); computed in double, stored as float
  rope_cos_ = arena_.alloc_f32(static_cast<size_t>(T) * (hd / 2));
  rope_sin_ = arena_.alloc_f32(static_cast<size_t>(T) * (hd / 2));
  for (int p = 0; p < T; ++p)
    for (int i = 0; i < hd / 2; ++i) {
      const double a = p * std::pow(static_cast<double>(cfg_.rope_theta), -2.0 * i / hd);
      rope_cos_[p * (hd / 2) + i] = static_cast<float>(std::cos(a));
      rope_sin_[p * (hd / 2) + i] = static_cast<float>(std::sin(a));
    }

  for (int l = 0; l < cfg_.n_layers; ++l) {
    kcache_.emplace_back(arena_.alloc_f32(static_cast<size_t>(T) * d), T, d);
    vcache_.emplace_back(arena_.alloc_f32(static_cast<size_t>(T) * d), T, d);
  }
  kv_bytes_ = 2 * static_cast<size_t>(cfg_.n_layers) * T * d * sizeof(float);
}

void Transformer::embed(int token, float* out) const {  // dequantises the row if the table is INT8
  const int d = cfg_.d_model;
  if (m_.emb.quantized()) {
    const int8_t* q = m_.emb.q8 + static_cast<size_t>(token) * d;
    for (int j = 0; j < d; ++j) out[j] = static_cast<float>(q[j]) * m_.emb.scales[token];
  } else {
    std::copy_n(m_.emb.f32 + static_cast<size_t>(token) * d, d, out);
  }
}

void Transformer::linear_batch(const Matrix& w, const float* X, int m, float* Y) const {
  if (w.quantized()) {  // INT8 prefill: one matrix-vector product per token for now
    for (int i = 0; i < m; ++i)
      kern_.gemv_q8(w.rows, w.cols, w.q8, w.scales, X + static_cast<size_t>(i) * w.cols, Y + static_cast<size_t>(i) * w.rows);
  } else {
    kern_.gemm_nt_f32(m, w.rows, w.cols, X, w.f32, Y);
  }
}

void Transformer::linear(const Matrix& w, const float* x, float* y) const {
  if (w.quantized())
    kern_.gemv_q8(w.rows, w.cols, w.q8, w.scales, x, y);
  else
    kern_.gemv_f32(w.rows, w.cols, w.f32, x, y);
}

void Transformer::attention(int layer, int pos, const float* q_all, float* out_all) {
  const int hd = cfg_.head_dim(), T = cfg_.max_seq;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  for (int h = 0; h < cfg_.n_heads; ++h) {
    // this head's columns of the cache: row t is K/V of position t, hd contiguous floats
    const Tensor kh = kcache_[layer].slice_cols(h * hd, hd);
    const Tensor vh = vcache_[layer].slice_cols(h * hd, hd);
    const float* q = q_all + h * hd;
    float* s = scores_ + static_cast<size_t>(h) * T;

    for (int t = 0; t <= pos; ++t) {  // causality: only positions <= pos exist in the cache
      const float* kr = kh.row(t);
      float dot = 0.f;
      for (int i = 0; i < hd; ++i) dot += q[i] * kr[i];
      s[t] = dot * scale;
    }
    softmax_inplace(s, pos + 1);

    float* out = out_all + h * hd;
    std::fill(out, out + hd, 0.f);
    for (int t = 0; t <= pos; ++t) {
      const float* vr = vh.row(t);
      const float w = s[t];
      for (int i = 0; i < hd; ++i) out[i] += w * vr[i];
    }
  }
}

const float* Transformer::forward(int token) {
  if (pos_ >= cfg_.max_seq) throw std::runtime_error("sequence is longer than max_seq");
  if (token < 0 || token >= cfg_.vocab) throw std::out_of_range("token id out of range");
  const int d = cfg_.d_model, hd = cfg_.head_dim(), pos = pos_;
  const float* cs = rope_cos_ + static_cast<size_t>(pos) * (hd / 2);
  const float* sn = rope_sin_ + static_cast<size_t>(pos) * (hd / 2);

  embed(token, x_);

  for (int l = 0; l < cfg_.n_layers; ++l) {
    const LayerWeights& L = m_.layers[l];

    rmsnorm(x_, L.attn_norm, xb_, d, cfg_.norm_eps);
    linear(L.wq, xb_, q_);
    linear(L.wk, xb_, k_);
    linear(L.wv, xb_, v_);
    for (int h = 0; h < cfg_.n_heads; ++h) {
      rope_inplace(q_ + h * hd, cs, sn, hd);
      rope_inplace(k_ + h * hd, cs, sn, hd);
    }
    std::copy_n(k_, d, kcache_[l].row(pos));  // append this position to the cache
    std::copy_n(v_, d, vcache_[l].row(pos));
    attention(l, pos, q_, att_out_);
    linear(L.wo, att_out_, xb_);
    for (int j = 0; j < d; ++j) x_[j] += xb_[j];  // residual

    rmsnorm(x_, L.mlp_norm, xb_, d, cfg_.norm_eps);
    linear(L.w1, xb_, hidden_);
    gelu_inplace(hidden_, cfg_.d_ff);
    linear(L.w2, hidden_, xb_);
    for (int j = 0; j < d; ++j) x_[j] += xb_[j];
  }

  rmsnorm(x_, m_.final_norm, xb_, d, cfg_.norm_eps);
  linear(m_.emb, xb_, logits_);  // tied output projection: logits = E * h
  ++pos_;
  return logits_;
}

const float* Transformer::prefill(const int* tokens, int n) {
  if (n < 1) throw std::invalid_argument("prefill needs at least one token");
  if (pos_ + n > cfg_.max_seq) throw std::runtime_error("sequence is longer than max_seq");
  for (int i = 0; i < n; ++i)
    if (tokens[i] < 0 || tokens[i] >= cfg_.vocab) throw std::out_of_range("token id out of range");
  const int d = cfg_.d_model, hd = cfg_.head_dim(), p0 = pos_;
  auto row = [&](float* base, int i, int width) { return base + static_cast<size_t>(i) * width; };

  for (int i = 0; i < n; ++i) embed(tokens[i], row(px_, i, d));

  for (int l = 0; l < cfg_.n_layers; ++l) {
    const LayerWeights& L = m_.layers[l];

    for (int i = 0; i < n; ++i) rmsnorm(row(px_, i, d), L.attn_norm, row(pxb_, i, d), d, cfg_.norm_eps);
    linear_batch(L.wq, pxb_, n, pq_);  // three GEMMs: n tokens share each pass over the weights
    linear_batch(L.wk, pxb_, n, pk_);
    linear_batch(L.wv, pxb_, n, pv_);
    for (int i = 0; i < n; ++i) {
      const float* cs = rope_cos_ + static_cast<size_t>(p0 + i) * (hd / 2);
      const float* sn = rope_sin_ + static_cast<size_t>(p0 + i) * (hd / 2);
      for (int h = 0; h < cfg_.n_heads; ++h) {
        rope_inplace(row(pq_, i, d) + h * hd, cs, sn, hd);
        rope_inplace(row(pk_, i, d) + h * hd, cs, sn, hd);
      }
      std::copy_n(row(pk_, i, d), d, kcache_[l].row(p0 + i));
      std::copy_n(row(pv_, i, d), d, vcache_[l].row(p0 + i));
    }
    for (int i = 0; i < n; ++i) attention(l, p0 + i, row(pq_, i, d), row(patt_, i, d));
    linear_batch(L.wo, patt_, n, pxb_);
    for (size_t j = 0; j < static_cast<size_t>(n) * d; ++j) px_[j] += pxb_[j];

    for (int i = 0; i < n; ++i) rmsnorm(row(px_, i, d), L.mlp_norm, row(pxb_, i, d), d, cfg_.norm_eps);
    linear_batch(L.w1, pxb_, n, phid_);
    gelu_inplace(phid_, n * cfg_.d_ff);
    linear_batch(L.w2, phid_, n, pxb_);
    for (size_t j = 0; j < static_cast<size_t>(n) * d; ++j) px_[j] += pxb_[j];
  }

  // only the last token's logits are needed to continue generation
  rmsnorm(row(px_, n - 1, d), m_.final_norm, xb_, d, cfg_.norm_eps);
  linear(m_.emb, xb_, logits_);
  pos_ += n;
  return logits_;
}

// ---------------------------------------------------------------------------------------
Sampler::Sampler(int vocab, float temperature, int top_k, float top_p, uint64_t seed)
    : vocab_(vocab), temperature_(temperature), top_p_(top_p), top_k_(top_k),
      state_(seed ? seed : 0x9E3779B97F4A7C15ull), idx_(vocab), probs_(vocab) {
  std::iota(idx_.begin(), idx_.end(), 0);
}

float Sampler::uniform() {  // xorshift64*, top 24 bits -> [0, 1)
  state_ ^= state_ >> 12;
  state_ ^= state_ << 25;
  state_ ^= state_ >> 27;
  return static_cast<float>((state_ * 0x2545F4914F6CDD1Dull) >> 40) * (1.0f / 16777216.0f);
}

int Sampler::sample(const float* logits) {
  if (temperature_ <= 0.f) return static_cast<int>(std::max_element(logits, logits + vocab_) - logits);

  int keep = (top_k_ > 0 && top_k_ < vocab_) ? top_k_ : vocab_;
  auto by_logit = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  std::partial_sort(idx_.begin(), idx_.begin() + keep, idx_.end(), by_logit);

  const float mx = logits[idx_[0]];
  float sum = 0.f;
  for (int i = 0; i < keep; ++i) {
    probs_[i] = std::exp((logits[idx_[i]] - mx) / temperature_);
    sum += probs_[i];
  }
  if (top_p_ > 0.f && top_p_ < 1.f) {  // smallest prefix whose probability mass reaches top_p
    float cum = 0.f;
    int n = 0;
    while (n < keep) {
      cum += probs_[n++] / sum;
      if (cum >= top_p_) break;
    }
    keep = n;
    sum = 0.f;
    for (int i = 0; i < keep; ++i) sum += probs_[i];
  }
  float r = uniform() * sum;
  for (int i = 0; i < keep; ++i) {
    r -= probs_[i];
    if (r <= 0.f) return idx_[i];
  }
  return idx_[keep - 1];
}

}  // namespace bm
