// Ternary packing, the multiplier-less GEMM, stochastic rounding and block-floating-point gradients.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels.hpp"
#include "lowbit_train.hpp"
#include "quantize.hpp"
#include "tensor.hpp"
#include "ternary.hpp"
#include "testing.hpp"

using namespace bm;

static std::vector<float> randn(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d;
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

static std::vector<int8_t> rand_ternary(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::vector<int8_t> v(n);
  for (auto& w : v) w = static_cast<int8_t>(int(g() % 3) - 1);
  return v;
}

static std::vector<int8_t> rand_act(size_t n, unsigned seed, bool extreme = false) {
  std::mt19937 g(seed);
  std::vector<int8_t> v(n);
  for (auto& x : v) x = extreme ? ((g() & 1) ? 127 : -127) : static_cast<int8_t>(int(g() % 255) - 127);  // [-127, 127]
  return v;
}

static void test_pack_layout_is_exactly_four_weights_per_byte() {
  // w0 | w1<<2 | w2<<4 | w3<<6 with codes 0 -> 00, +1 -> 01, -1 -> 10
  const int8_t w[4] = {1, -1, 0, 1};
  std::vector<uint8_t> p(ternary_row_bytes(4));
  pack_ternary(w, 1, 4, p.data());
  CHECK(p[0] == (1u | (2u << 2) | (0u << 4) | (1u << 6)));
  CHECK(p[0] == 0x49);
  CHECK(p.size() == 8);                                   // a row is padded to 32 weights = 8 bytes
  for (size_t i = 1; i < p.size(); ++i) CHECK(p[i] == 0);  // padding is zero codes
  CHECK(ternary_row_bytes(32) == 8 && ternary_row_bytes(33) == 16 && ternary_row_bytes(1) == 8);
  CHECK(ternary_row_bytes(2048) * 4 == 2048);              // 2 bits/weight: 16x smaller than FP32
}

static void test_pack_unpack_roundtrip_all_shapes() {
  for (int K : {1, 2, 3, 4, 5, 7, 31, 32, 33, 63, 64, 100, 257})
    for (int R : {1, 3, 8}) {
      auto W = rand_ternary(size_t(R) * K, 17 + K + R);
      std::vector<uint8_t> p(size_t(R) * ternary_row_bytes(K), 0xFF);  // poisoned: pack must overwrite padding
      std::vector<int8_t> back(size_t(R) * K, 5);
      pack_ternary(W.data(), R, K, p.data());
      unpack_ternary(p.data(), R, K, back.data());
      CHECK(W == back);
      for (int r = 0; r < R; ++r)  // padding codes are zero
        for (int k = K; k < ternary_row_bytes(K) * 4; ++k) CHECK(((p[size_t(r) * ternary_row_bytes(K) + (k >> 2)] >> ((k & 3) * 2)) & 3) == 0);
    }
}

static void test_all_256_byte_values_decode_consistently() {
  // every packed byte decodes to four weights; the reserved code 11 decodes as 0 everywhere
  std::vector<uint8_t> bytes(32, 0);
  std::vector<int8_t> x(32);
  for (int b = 0; b < 256; ++b) {
    std::fill(bytes.begin(), bytes.end(), 0);
    bytes[0] = static_cast<uint8_t>(b);
    int8_t un[32];
    unpack_ternary(bytes.data(), 1, 32, un);
    for (int j = 0; j < 4; ++j) {
      const unsigned code = (b >> (2 * j)) & 3;
      CHECK(un[j] == (code == 1 ? 1 : code == 2 ? -1 : 0));
    }
    for (int k = 0; k < 32; ++k) x[k] = static_cast<int8_t>(k + 1);
    int32_t acc = 0, ref = 0;
    ternary_gemm_i32(1, 1, 32, x.data(), bytes.data(), &acc);
    for (int k = 0; k < 4; ++k) ref += un[k] * x[k];
    CHECK(acc == ref);  // the SIMD mask path and the unpacked reference agree on every byte value
  }
}

static void test_absmean_quantizer() {
  auto W = randn(1000, 5);
  std::vector<int8_t> q(1000);
  const float gamma = ternary_quantize_absmean(W.data(), W.size(), q.data());
  double mean_abs = 0;
  for (float v : W) mean_abs += std::fabs(v);
  CHECK_NEAR(gamma, mean_abs / 1000.0, 1e-6);
  int zeros = 0;
  for (size_t i = 0; i < W.size(); ++i) {
    CHECK(q[i] >= -1 && q[i] <= 1);
    CHECK(q[i] == std::max(-1.f, std::min(1.f, std::nearbyint(W[i] / gamma))));
    zeros += q[i] == 0;
  }
  CHECK(zeros > 200 && zeros < 600);  // a Gaussian puts roughly 40-45% of weights below gamma/2
  std::vector<float> zeroW(10, 0.f);
  std::vector<int8_t> qz(10, 3);
  CHECK(ternary_quantize_absmean(zeroW.data(), 10, qz.data()) == 0.f);
  for (auto v : qz) CHECK(v == 0);
  const float tie[4] = {1.f, 0.5f, -0.5f, 0.f};  // mean|w| = 0.5625
  int8_t qt[4];
  const float g = ternary_quantize_absmean(tie, 4, qt);
  CHECK_NEAR(g, 0.5, 1e-6);  // mean(1, .5, .5, 0) = 0.5 -> W/gamma = 2, 1, -1, 0
  CHECK(qt[0] == 1 && qt[1] == 1 && qt[2] == -1 && qt[3] == 0);
}

// Brute-force reference that uses real multiplication, to show the add/sub kernel computes the same integer.
static void check_gemm(int M, int N, int K, unsigned seed, bool extreme) {
  auto W = rand_ternary(size_t(N) * K, seed);
  auto X = rand_act(size_t(M) * K, seed + 1, extreme);
  std::vector<uint8_t> p(size_t(N) * ternary_row_bytes(K));
  pack_ternary(W.data(), N, K, p.data());
  std::vector<int32_t> a(size_t(M) * N, -7), b(size_t(M) * N, -7), c(size_t(M) * N);
  ternary_gemm_i32(M, N, K, X.data(), p.data(), a.data());
  ternary_gemm_i32_ref(M, N, K, X.data(), p.data(), b.data());
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k) s += int32_t(W[size_t(n) * K + k]) * X[size_t(m) * K + k];
      c[size_t(m) * N + n] = s;
    }
  if (a != c || b != c) std::fprintf(stderr, "    mismatch M=%d N=%d K=%d extreme=%d\n", M, N, K, extreme);
  CHECK(a == c);
  CHECK(b == c);
}

static void test_ternary_gemm_exact_on_all_shapes() {
  int cases = 0;
  for (int nt : {1, 4}) {
    set_num_threads(nt);
    for (int M : {1, 2, 3, 4, 5, 63, 64, 65, 130})
      for (int N : {1, 3, 4, 5, 33})
        for (int K : {1, 4, 31, 32, 33, 64, 100, 257, 2048}) {
          check_gemm(M, N, K, 100 + M + N + K, false);
          ++cases;
        }
  }
  set_num_threads(num_threads());
  std::printf("      %d shape/thread combinations, exact integer equality\n", cases);
}

static void test_extreme_activations_do_not_overflow() {
  // the byte-wise (x & pos) - (x & neg) step must hold for |x| = 127 with every sign pattern
  for (int K : {32, 100, 2048}) check_gemm(7, 9, K, 4242 + K, true);
  const int K = 2048;  // all-+1 weights against all-127: 127 * 2048 per output, exactly
  std::vector<int8_t> W(size_t(3) * K, 1), X(K, 127);
  for (int k = 0; k < K; ++k) W[size_t(K) + k] = -1;
  std::vector<uint8_t> p(size_t(3) * ternary_row_bytes(K));
  pack_ternary(W.data(), 3, K, p.data());
  int32_t y[3];
  ternary_gemv_i32(3, K, p.data(), X.data(), y);
  CHECK(y[0] == 127 * K && y[1] == -127 * K);
}

static void test_zero_and_sign_semantics() {
  const int K = 64;
  std::vector<int8_t> X(K);
  for (int k = 0; k < K; ++k) X[k] = static_cast<int8_t>((k % 2) ? -(k + 1) : (k + 1));
  std::vector<int8_t> Wz(K, 0), Wp(K, 1), Wn(K, -1);
  std::vector<uint8_t> p(3 * ternary_row_bytes(K));
  pack_ternary(Wz.data(), 1, K, p.data());
  pack_ternary(Wp.data(), 1, K, p.data() + ternary_row_bytes(K));
  pack_ternary(Wn.data(), 1, K, p.data() + 2 * ternary_row_bytes(K));
  int32_t y[3], sum = 0;
  for (int k = 0; k < K; ++k) sum += X[k];
  ternary_gemv_i32(3, K, p.data(), X.data(), y);
  CHECK(y[0] == 0 && y[1] == sum && y[2] == -sum);
}

static void test_f32_wrapper_rescales_once_per_output() {
  const int M = 5, N = 7, K = 100;
  auto Wf = randn(size_t(N) * K, 9);
  std::vector<int8_t> Wq(size_t(N) * K);
  const float gamma = ternary_quantize_absmean(Wf.data(), Wf.size(), Wq.data());
  std::vector<uint8_t> p(size_t(N) * ternary_row_bytes(K));
  pack_ternary(Wq.data(), N, K, p.data());
  auto X = randn(size_t(M) * K, 10);
  std::vector<int8_t> Xq(size_t(M) * K);
  std::vector<float> xs(M), Y(size_t(M) * N);
  for (int m = 0; m < M; ++m) xs[m] = quantize_row_q8(X.data() + size_t(m) * K, K, Xq.data() + size_t(m) * K);
  ternary_gemm_f32(M, N, K, Xq.data(), xs.data(), p.data(), gamma, Y.data());
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double ref = 0;
      for (int k = 0; k < K; ++k) ref += double(Wq[size_t(n) * K + k]) * Xq[size_t(m) * K + k];
      CHECK_NEAR(Y[size_t(m) * N + n], ref * gamma * xs[m], 1e-4 * (1 + std::fabs(ref * gamma * xs[m])));
    }
  std::vector<float> y1(N);
  ternary_gemv_f32(N, K, p.data(), gamma, Xq.data(), xs[0], y1.data());
  for (int n = 0; n < N; ++n) CHECK(y1[n] == Y[n]);
}

// ---- stochastic rounding ------------------------------------------------------------------------
static void test_stochastic_rounding_is_unbiased() {
  // For a fixed x the SR result is floor(x) or ceil(x); averaged over many independent draws it must be x.
  const size_t n = 200000;
  for (float x : {0.0f, 0.1f, 0.5f, 0.9f, 2.3f, -2.3f, -0.25f, 100.75f, 1e-3f, -1e-3f}) {
    std::vector<float> in(n, x), out(n);
    stochastic_round_array(in.data(), n, 12345, out.data());
    double mean = 0;
    for (float v : out) {
      CHECK(v == std::floor(x) || v == std::ceil(x));
      mean += v;
    }
    mean /= double(n);
    const double frac = x - std::floor(x);
    CHECK_NEAR(mean, x, 5.0 * std::sqrt(frac * (1 - frac) / double(n)) + 1e-4);  // 5 sigma of a Bernoulli mean
    if (x == 1e-3f || x == -1e-3f) {  // the property that matters: tiny values do not all collapse to zero
      int nonzero = 0;
      for (float v : out) nonzero += v != 0.f;
      CHECK(nonzero > 0);
    }
  }
  CHECK(std::nearbyint(0.4f) == 0.f);  // round-to-nearest: every x < 0.5 maps to 0, the bias SR removes
}

static void test_stochastic_rounding_deterministic_and_thread_independent() {
  const size_t n = 300000;  // above the parallel threshold
  auto x = randn(n, 77);
  std::vector<float> a(n), b(n), c(n);
  set_num_threads(1);
  stochastic_round_array(x.data(), n, 99, a.data());
  set_num_threads(8);
  stochastic_round_array(x.data(), n, 99, b.data());
  stochastic_round_array(x.data(), n, 100, c.data());
  set_num_threads(num_threads());
  CHECK(a == b);  // the draw for element i depends only on (seed, i)
  CHECK(a != c);
}

static void test_accumulated_tiny_updates_survive_only_with_sr() {
  // 2000 updates of 0.1 LSB: round-to-nearest drops every one, stochastic rounding keeps ~200 LSB.
  const int steps = 2000, n = 4096;
  std::vector<float> w_near(n, 0.f), w_sr(n, 0.f), one(n), out(n);
  for (int t = 0; t < steps; ++t) {
    for (int i = 0; i < n; ++i) {
      w_near[i] = std::nearbyint(w_near[i] + 0.1f);
      one[i] = w_sr[i] + 0.1f;
    }
    stochastic_round_array(one.data(), n, 1000 + t, out.data());
    w_sr = out;
  }
  double mean_sr = 0, mean_near = 0;
  for (int i = 0; i < n; ++i) mean_sr += w_sr[i], mean_near += w_near[i];
  mean_sr /= n;
  mean_near /= n;
  CHECK(mean_near == 0.0);
  CHECK_NEAR(mean_sr, 0.1 * steps, 3.0);
}

// ---- block floating point -----------------------------------------------------------------------
static void test_bfp_nearest_matches_hand_computation() {
  // block of 4, 8-bit mantissas (qmax 127): amax 1.27 -> amax/qmax = 0.01 -> e = ceil(log2(0.01)) = -6 (scale 1/64)
  const float x[4] = {1.27f, -0.5f, 0.0078125f, 0.0234375f};
  int8_t m[4], e[1];
  bfp_quantize(x, 4, 4, 8, Rounding::Nearest, 0, m, e);
  CHECK(e[0] == -6);
  // x * 64 = 81.28, -32, 0.5 (tie -> even 0), 1.5 (tie -> even 2)
  CHECK(m[0] == 81 && m[1] == -32 && m[2] == 0 && m[3] == 2);
  float back[4];
  bfp_dequantize(m, e, 4, 4, back);
  CHECK_NEAR(back[0], 81.0 / 64, 1e-7);
  const float exact[4] = {127.f, 1.f, 0.f, 0.f};  // amax/qmax = 1 exactly -> e = 0 (power-of-two boundary)
  bfp_quantize(exact, 4, 4, 8, Rounding::Nearest, 0, m, e);
  CHECK(e[0] == 0 && m[0] == 127 && m[1] == 1);
  const float zeros[4] = {0, 0, 0, 0};
  bfp_quantize(zeros, 4, 4, 8, Rounding::Nearest, 0, m, e);
  CHECK(e[0] == 0 && m[0] == 0);
}

static void test_bfp_error_bounds_and_range() {
  for (int bits : {4, 8})
    for (int block : {8, 32}) {
      const size_t n = 1000;  // not a multiple of the block: the last block is short
      auto x = randn(n, 31 + bits + block);
      for (size_t i = 0; i < n; i += 97) x[i] *= 50.f;  // outliers
      const int qmax = (1 << (bits - 1)) - 1;
      std::vector<int8_t> m(n), e((n + block - 1) / block);
      std::vector<float> back(n);
      for (Rounding mode : {Rounding::Nearest, Rounding::Stochastic}) {
        bfp_quantize(x.data(), n, block, bits, mode, 5, m.data(), e.data());
        bfp_dequantize(m.data(), e.data(), n, block, back.data());
        for (size_t i = 0; i < n; ++i) {
          CHECK(std::abs(int(m[i])) <= qmax);
          const double step = std::ldexp(1.0, e[i / block]);
          const double bound = (mode == Rounding::Nearest ? 0.5 : 1.0) * step + 1e-6;  // SR errs by < 1 step
          CHECK(std::fabs(back[i] - x[i]) <= bound);
        }
      }
      for (size_t b = 0; b < e.size(); ++b) {  // the exponent is the smallest one that fits the block
        float amax = 0;
        for (size_t i = b * block; i < std::min(n, (b + 1) * block); ++i) amax = std::max(amax, std::fabs(x[i]));
        CHECK(amax / std::ldexp(1.0f, e[b]) <= qmax + 1e-3);
        CHECK(amax / std::ldexp(1.0f, e[b] - 1) > qmax);
      }
    }
}

static void test_bfp_stochastic_is_unbiased_per_element() {
  // average the dequantised gradient over many independent seeds: it converges to the original
  const size_t n = 96;
  auto g = randn(n, 8);
  for (auto& v : g) v *= 1e-3f;
  std::vector<int8_t> m(n), e(n / 32);
  std::vector<float> back(n), mean(n, 0.f), mean_near(n);
  const int draws = 4000;
  for (int s = 0; s < draws; ++s) {
    bfp_quantize(g.data(), n, 32, 4, Rounding::Stochastic, 7000 + s, m.data(), e.data());  // 4-bit: very coarse
    bfp_dequantize(m.data(), e.data(), n, 32, back.data());
    for (size_t i = 0; i < n; ++i) mean[i] += back[i] / draws;
  }
  bfp_quantize(g.data(), n, 32, 4, Rounding::Nearest, 0, m.data(), e.data());
  bfp_dequantize(m.data(), e.data(), n, 32, mean_near.data());
  double err_sr = 0, err_near = 0, scale = 0;
  for (size_t i = 0; i < n; ++i) {
    err_sr += std::fabs(mean[i] - g[i]);
    err_near += std::fabs(mean_near[i] - g[i]);
    scale += std::fabs(g[i]);
  }
  CHECK(err_sr < 0.05 * scale);   // SR average is within 5% of the true gradient
  CHECK(err_near > 3 * err_sr);   // while the deterministic 4-bit rounding keeps its systematic error
}

static void test_grid_quantizer_matches_int8_rule() {
  const float x[6] = {-200.f, -1.4f, 0.4f, 0.6f, 2.5f, 300.f};
  int8_t q[6];
  quantize_to_grid_i8(x, 6, 1.0f, 8, Rounding::Nearest, 0, q);
  CHECK(q[0] == -127 && q[1] == -1 && q[2] == 0 && q[3] == 1 && q[4] == 2 && q[5] == 127);  // clamp, half-to-even
  quantize_to_grid_i8(x, 6, 0.5f, 8, Rounding::Nearest, 0, q);
  CHECK(q[3] == 1 && q[2] == 1);  // 0.4/0.5 = 0.8 -> 1
}

int main() {
  RUN(test_pack_layout_is_exactly_four_weights_per_byte);
  RUN(test_pack_unpack_roundtrip_all_shapes);
  RUN(test_all_256_byte_values_decode_consistently);
  RUN(test_absmean_quantizer);
  RUN(test_ternary_gemm_exact_on_all_shapes);
  RUN(test_extreme_activations_do_not_overflow);
  RUN(test_zero_and_sign_semantics);
  RUN(test_f32_wrapper_rescales_once_per_output);
  RUN(test_stochastic_rounding_is_unbiased);
  RUN(test_stochastic_rounding_deterministic_and_thread_independent);
  RUN(test_accumulated_tiny_updates_survive_only_with_sr);
  RUN(test_bfp_nearest_matches_hand_computation);
  RUN(test_bfp_error_bounds_and_range);
  RUN(test_bfp_stochastic_is_unbiased_per_element);
  RUN(test_grid_quantizer_matches_int8_rule);
  return finish();
}
