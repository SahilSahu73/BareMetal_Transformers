#include <algorithm>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include "kernels.hpp"
#include "quantize.hpp"
#include "testing.hpp"
#include "tensor.hpp"
#include "transformer.hpp"

using namespace bm;

static std::vector<float> randn(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d;
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

static void test_gemm_naive_matches_double_reference() {
  for (auto [M, N, K] : {std::tuple{1, 1, 1}, {5, 7, 3}, {16, 16, 16}, {33, 17, 65}}) {
    auto A = randn(M * K, 1), B = randn(K * N, 2);
    std::vector<float> C(M * N);
    gemm_naive(M, N, K, A.data(), B.data(), C.data());
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        double ref = 0;
        for (int k = 0; k < K; ++k) ref += double(A[i * K + k]) * B[k * N + j];
        CHECK_NEAR(C[i * N + j], ref, 1e-4);
      }
  }
}

static void test_gemv_matches_gemm_with_one_column() {
  const int N = 19, K = 37;
  auto W = randn(N * K, 3), x = randn(K, 4);
  std::vector<float> y(N), ref(N);
  gemv_f32_naive(N, K, W.data(), x.data(), y.data());
  // y = W x  is  C[N,1] = A[N,K] * B[K,1]
  gemm_naive(N, 1, K, W.data(), x.data(), ref.data());
  for (int i = 0; i < N; ++i) CHECK_NEAR(y[i], ref[i], 1e-5);
}

static void test_gemv_q8_matches_dequantised_f32() {
  const int N = 11, K = 50;
  std::mt19937 g(5);
  std::vector<int8_t> Q(N * K);
  for (auto& q : Q) q = static_cast<int8_t>(int(g() % 255) - 127);
  auto scales = randn(N, 6);
  for (auto& s : scales) s = std::fabs(s) + 0.01f;
  auto x = randn(K, 7);
  std::vector<float> W(N * K), y(N), ref(N);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) W[n * K + k] = Q[n * K + k] * scales[n];
  gemv_q8_naive(N, K, Q.data(), scales.data(), x.data(), y.data());
  gemv_f32_naive(N, K, W.data(), x.data(), ref.data());
  for (int i = 0; i < N; ++i) CHECK_NEAR(y[i], ref[i], 1e-3);
}

static void test_ops() {
  std::vector<float> x = {1000.f, 1001.f, 999.f, -5.f}, y(4), g = {1, 1, 1, 1};
  softmax_inplace(x.data(), 4);  // huge logits must not overflow
  float s = 0;
  for (float v : x) { CHECK(std::isfinite(v)); s += v; }
  CHECK_NEAR(s, 1.0, 1e-6);

  std::vector<float> a = {3.f, 4.f};
  std::vector<float> ones = {1.f, 1.f}, out(2);
  rmsnorm(a.data(), ones.data(), out.data(), 2, 0.f);  // rms = sqrt(12.5)
  CHECK_NEAR(out[0], 3.0 / std::sqrt(12.5), 1e-6);

  std::vector<float> z = {0.f, 1.f, -1.f, 3.f};
  gelu_inplace(z.data(), 4);
  CHECK_NEAR(z[0], 0.0, 1e-7);
  CHECK_NEAR(z[1], 0.8411920, 1e-5);   // GELU(1) tanh approximation
  CHECK_NEAR(z[2], -0.1588080, 1e-5);

  float head[4] = {1.f, 0.f, 0.f, 1.f};
  const float cs[2] = {0.f, 1.f}, sn[2] = {1.f, 0.f};  // 90 degrees on pair 0, identity on pair 1
  rope_inplace(head, cs, sn, 4);
  CHECK_NEAR(head[0], 0.0, 1e-7);
  CHECK_NEAR(head[1], 1.0, 1e-7);
  CHECK_NEAR(head[2], 0.0, 1e-7);
  CHECK_NEAR(head[3], 1.0, 1e-7);
}

static void test_arena_alignment_and_views() {
  Arena arena(4096);
  float* a = arena.alloc_f32(3);
  float* b = arena.alloc_f32(5);
  CHECK(reinterpret_cast<uintptr_t>(a) % kAlign == 0);
  CHECK(reinterpret_cast<uintptr_t>(b) % kAlign == 0);
  CHECK(b > a);
  bool threw = false;
  try { arena.alloc(1 << 20); } catch (const std::bad_alloc&) { threw = true; }
  CHECK(threw);

  Tensor t(arena.alloc_f32(4 * 6), 4, 6);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 6; ++j) t(i, j) = float(10 * i + j);
  Tensor cols = t.slice_cols(2, 3);  // a view, not a copy
  CHECK(cols.rows == 4 && cols.cols == 3 && cols.stride == 6);
  CHECK(cols(2, 1) == 23.f);
  cols(0, 0) = -1.f;
  CHECK(t(0, 2) == -1.f);
  CHECK(t.slice_rows(1, 2)(1, 5) == 25.f);
}

static void test_sampler() {
  const std::vector<float> logits = {0.1f, 3.0f, 0.2f, 2.9f, -1.f};
  Sampler greedy(5, 0.f, 0, 1.f, 1);
  CHECK(greedy.sample(logits.data()) == 1);

  Sampler top1(5, 1.0f, 1, 1.f, 42);  // top-k = 1 collapses to argmax
  for (int i = 0; i < 20; ++i) CHECK(top1.sample(logits.data()) == 1);

  Sampler a(5, 1.0f, 0, 1.f, 7), b(5, 1.0f, 0, 1.f, 7);  // same seed, same stream
  for (int i = 0; i < 50; ++i) CHECK(a.sample(logits.data()) == b.sample(logits.data()));

  Sampler topk2(5, 1.0f, 2, 1.f, 3);  // only the two best tokens (1 and 3) may appear
  int seen[5] = {0};
  for (int i = 0; i < 400; ++i) ++seen[topk2.sample(logits.data())];
  CHECK(seen[0] + seen[2] + seen[4] == 0 && seen[1] > 0 && seen[3] > 0);

  // softmax mass: token 1 = 0.491, token 3 = 0.444, the rest 0.065
  Sampler nucleus(5, 1.0f, 0, 0.4f, 9);  // 0.491 alone already exceeds top_p
  for (int i = 0; i < 50; ++i) CHECK(nucleus.sample(logits.data()) == 1);

  Sampler nucleus2(5, 1.0f, 0, 0.9f, 11);  // 0.491 + 0.444 >= 0.9: exactly tokens {1, 3}
  int seen2[5] = {0};
  for (int i = 0; i < 400; ++i) ++seen2[nucleus2.sample(logits.data())];
  CHECK(seen2[0] + seen2[2] + seen2[4] == 0 && seen2[1] > 0 && seen2[3] > 0);
}


// ---- optimised kernels vs a double-precision reference ---------------------------------------
// C = A * B (or A * B^T) with explicit row strides; returns the largest error relative to tolerance.
static double ref_gemm_err(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                           bool bt, const float* C, int ldc) {
  double worst = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double ref = 0;
      for (int k = 0; k < K; ++k)
        ref += double(A[size_t(i) * lda + k]) * (bt ? B[size_t(j) * ldb + k] : B[size_t(k) * ldb + j]);
      const double tol = 1e-4 * (1.0 + std::sqrt(double(K)));
      worst = std::max(worst, std::fabs(C[size_t(i) * ldc + j] - ref) / tol);
    }
  return worst;
}

static void test_reordered_and_tiled_match_reference() {
  for (auto [M, N, K] : {std::tuple{1, 1, 1}, {7, 9, 5}, {64, 64, 64}, {70, 130, 65}, {129, 33, 257}}) {
    auto A = randn(size_t(M) * K, 11), B = randn(size_t(K) * N, 12);
    std::vector<float> C(size_t(M) * N);
    gemm_reordered(M, N, K, A.data(), B.data(), C.data());
    CHECK(ref_gemm_err(M, N, K, A.data(), K, B.data(), N, false, C.data(), N) <= 1.0);
    for (int block : {1, 7, 16, 64}) {
      gemm_tiled(M, N, K, A.data(), B.data(), C.data(), block);
      CHECK(ref_gemm_err(M, N, K, A.data(), K, B.data(), N, false, C.data(), N) <= 1.0);
    }
  }
}

// Remainder handling is where packed kernels break: sweep sizes around the micro-tile and block edges.
static void test_gemm_simd_edge_shapes() {
  const int MR = micro_tile_m(), NR = micro_tile_n();
  const int Ms[] = {1, 2, MR - 1, MR, MR + 1, 2 * MR + 1, 13, 72, 73};
  const int Ns[] = {1, 3, NR - 1, NR, NR + 1, 2 * NR + 5, 130};
  const int Ks[] = {1, 2, 15, 16, 17, 255, 256, 257, 300};
  int cases = 0;
  for (int nt : {1, 4}) {
    set_num_threads(nt);
    for (int M : Ms)
      for (int N : Ns)
        for (int K : Ks)
          for (bool bt : {false, true}) {
            auto A = randn(size_t(M) * K, 20 + M), B = randn(size_t(K) * N, 30 + N);
            std::vector<float> C(size_t(M) * N, -777.f);  // stale values must be overwritten
            gemm_simd(M, N, K, A.data(), B.data(), C.data(), bt);
            const double e = ref_gemm_err(M, N, K, A.data(), K, B.data(), bt ? K : N, bt, C.data(), N);
            if (e > 1.0) std::fprintf(stderr, "    M=%d N=%d K=%d bt=%d threads=%d err/tol=%.2f\n", M, N, K, bt, nt, e);
            CHECK(e <= 1.0);
            ++cases;
          }
  }
  set_num_threads(num_threads());
  std::printf("      %d shape/transpose/thread combinations (isa %s, tile %dx%d)\n", cases, simd_isa(), MR, NR);
}

// Tiny blocking parameters force many mc/kc/nc blocks and exercise the accumulate-into-C path.
static void test_gemm_simd_small_blocking() {
  const int M = 50, N = 70, K = 100;
  auto A = randn(size_t(M) * K, 41), B = randn(size_t(K) * N, 42);
  for (auto blk : {GemmBlocking{12, 7, 32}, GemmBlocking{6, 1, 16}, GemmBlocking{1000, 1000, 1000}, GemmBlocking{5, 33, 17}}) {
    for (int nt : {1, 3}) {
      set_num_threads(nt);
      std::vector<float> C(size_t(M) * N);
      gemm_simd(M, N, K, A.data(), B.data(), C.data(), false, blk);
      CHECK(ref_gemm_err(M, N, K, A.data(), K, B.data(), N, false, C.data(), N) <= 1.0);
    }
  }
  set_num_threads(num_threads());
}

// Leading dimensions larger than the logical width: the kernel must honour strides, not assume contiguity.
static void test_gemm_simd_padded_leading_dimensions() {
  const int M = 19, N = 23, K = 31, lda = K + 5, ldb = N + 3, ldc = N + 9;
  auto A = randn(size_t(M) * lda, 51), B = randn(size_t(K) * ldb, 52);
  std::vector<float> C(size_t(M) * ldc, 12345.f);
  gemm_simd_ld(M, N, K, A.data(), lda, B.data(), ldb, false, C.data(), ldc);
  CHECK(ref_gemm_err(M, N, K, A.data(), lda, B.data(), ldb, false, C.data(), ldc) <= 1.0);
  for (int i = 0; i < M; ++i)
    for (int j = N; j < ldc; ++j) CHECK(C[size_t(i) * ldc + j] == 12345.f);  // padding untouched

  const int ldbt = K + 7;  // transposed B: W[N, K] with padded rows
  auto W = randn(size_t(N) * ldbt, 53);
  std::vector<float> C2(size_t(M) * ldc, 0.f);
  gemm_simd_ld(M, N, K, A.data(), lda, W.data(), ldbt, true, C2.data(), ldc);
  CHECK(ref_gemm_err(M, N, K, A.data(), lda, W.data(), ldbt, true, C2.data(), ldc) <= 1.0);
}

static void test_gemv_simd_all_shapes() {
  for (int nt : {1, 4}) {
    set_num_threads(nt);
    for (int N : {1, 2, 3, 4, 5, 7, 8, 63, 64, 65, 257})
      for (int K : {1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 513}) {
        auto W = randn(size_t(N) * K, 60 + N), x = randn(K, 70 + K);
        std::vector<float> y(N, -1.f);
        gemv_f32_simd(N, K, W.data(), x.data(), y.data());
        CHECK(ref_gemm_err(N, 1, K, W.data(), K, x.data(), 1, false, y.data(), 1) <= 1.0);
      }
    // force the parallel path on a small matrix as well
    const size_t saved = gemv_parallel_threshold();
    gemv_parallel_threshold() = 0;
    const int N = 37, K = 70;
    auto W = randn(size_t(N) * K, 81), x = randn(K, 82);
    std::vector<float> y(N);
    gemv_f32_simd(N, K, W.data(), x.data(), y.data());
    CHECK(ref_gemm_err(N, 1, K, W.data(), K, x.data(), 1, false, y.data(), 1) <= 1.0);
    gemv_parallel_threshold() = saved;
  }
  set_num_threads(num_threads());
}

static void test_gemm_nt_kernels_agree() {
  const int M = 9, N = 41, K = 70;
  auto X = randn(size_t(M) * K, 91), W = randn(size_t(N) * K, 92);
  std::vector<float> a(size_t(M) * N), b(size_t(M) * N), one(N), one_ref(N);
  gemm_nt_naive(M, N, K, X.data(), W.data(), a.data());
  gemm_nt_simd(M, N, K, X.data(), W.data(), b.data());
  for (size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-3);
  gemm_nt_simd(1, N, K, X.data(), W.data(), one.data());  // M == 1 takes the GEMV path
  gemm_nt_naive(1, N, K, X.data(), W.data(), one_ref.data());
  for (int i = 0; i < N; ++i) CHECK_NEAR(one[i], one_ref[i], 1e-3);
}

static void test_simd_gemm_is_deterministic_across_thread_counts() {
  // Each C element is reduced by exactly one thread in a fixed k order, so results must not
  // depend on how many threads ran.
  const int M = 90, N = 100, K = 300;
  auto A = randn(size_t(M) * K, 101), B = randn(size_t(K) * N, 102);
  std::vector<float> c1(size_t(M) * N), c8(size_t(M) * N);
  set_num_threads(1);
  gemm_simd(M, N, K, A.data(), B.data(), c1.data());
  set_num_threads(8);
  gemm_simd(M, N, K, A.data(), B.data(), c8.data());
  set_num_threads(num_threads());
  CHECK(c1 == c8);
}

// ---- INT8 --------------------------------------------------------------------------------------
static std::vector<int8_t> rand_i8(size_t n, unsigned seed, bool extremes = false) {
  std::mt19937 g(seed);
  std::vector<int8_t> v(n);
  for (auto& q : v) q = extremes ? ((g() & 1) ? 127 : -127) : static_cast<int8_t>(int(g() % 255) - 127);
  return v;
}

static void test_quantize_properties() {
  for (int n : {1, 3, 7, 8, 9, 16, 31, 64, 100}) {  // scalar tail, SIMD body, and both
    auto x = randn(n, 200 + n);
    std::vector<int8_t> q(n);
    const float s = quantize_row_q8(x.data(), n, q.data());
    float amax = 0;
    for (float v : x) amax = std::max(amax, std::fabs(v));
    CHECK_NEAR(s, amax / 127.f, 1e-7);
    int maxq = 0;
    for (int i = 0; i < n; ++i) {
      CHECK(q[i] >= -127);  // -128 is never produced (it would break the vpmaddubsw trick)
      maxq = std::max(maxq, std::abs(int(q[i])));
      CHECK(std::fabs(q[i] * s - x[i]) <= s / 2 + 1e-6f);  // round-trip error bound
    }
    CHECK(maxq == 127);  // the largest element uses the full range
  }
  std::vector<float> z(20, 0.f);
  std::vector<int8_t> qz(20, 5);
  CHECK(quantize_row_q8(z.data(), 20, qz.data()) == 1.f);
  for (auto q : qz) CHECK(q == 0);
}

static void test_quantize_rounds_half_to_even() {
  // amax = 127 -> scale 1, so the values below are exact ties
  for (int n : {10, 4}) {  // 10 exercises the SIMD loop (8 + tail), 4 the scalar path
    std::vector<float> x(n, 0.f);
    x[0] = 127.f; x[1] = 0.5f; x[2] = 1.5f; x[3] = 2.5f;
    if (n > 4) { x[4] = -0.5f; x[5] = -1.5f; x[6] = -2.5f; x[7] = 3.5f; x[8] = -3.5f; }
    std::vector<int8_t> q(n);
    CHECK(quantize_row_q8(x.data(), n, q.data()) == 1.f);
    CHECK(q[0] == 127 && q[1] == 0 && q[2] == 2 && q[3] == 2);
    if (n > 4) CHECK(q[4] == 0 && q[5] == -2 && q[6] == -2 && q[7] == 4 && q[8] == -4);
  }
}

static void test_quantize_matrix_is_per_row() {
  const int R = 5, C = 21;
  auto W = randn(R * C, 211);
  for (int c = 0; c < C; ++c) W[2 * C + c] *= 100.f;  // one large row must not affect the others
  std::vector<int8_t> Q(R * C);
  std::vector<float> sc(R);
  quantize_matrix_q8(W.data(), R, C, Q.data(), sc.data());
  CHECK(sc[2] > 50.f * sc[0]);
  for (int r = 0; r < R; ++r) {
    int maxq = 0;
    for (int c = 0; c < C; ++c) maxq = std::max(maxq, std::abs(int(Q[r * C + c])));
    CHECK(maxq == 127);  // every row uses its own full range
  }
}

static void test_gemv_q8_simd_matches_scalar() {
  for (int nt : {1, 4}) {
    set_num_threads(nt);
    for (int N : {1, 3, 4, 5, 8, 33, 130})
      for (int K : {1, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 513}) {
        auto Q = rand_i8(size_t(N) * K, 300 + N + K);
        auto sc = randn(N, 301);
        for (auto& v : sc) v = std::fabs(v) * 0.01f + 0.001f;
        auto x = randn(K, 302);
        std::vector<float> a(N), b(N);
        gemv_q8_naive(N, K, Q.data(), sc.data(), x.data(), a.data());
        gemv_q8_simd(N, K, Q.data(), sc.data(), x.data(), b.data());
        for (int i = 0; i < N; ++i) CHECK_NEAR(a[i], b[i], 1e-4 * (1 + std::fabs(a[i])));
      }
  }
  set_num_threads(num_threads());
}

static void test_q8a8_gemv_and_gemm_match_scalar() {
  for (int nt : {1, 4}) {
    set_num_threads(nt);
    for (int M : {1, 2, 3, 5, 8})
      for (int N : {1, 3, 4, 5, 31, 32, 33, 70})
        for (int K : {1, 31, 32, 33, 64, 100, 2048}) {
          auto Xq = rand_i8(size_t(M) * K, 400 + M + K), Q = rand_i8(size_t(N) * K, 401 + N + K);
          auto ws = randn(N, 402), xs = randn(M, 403);
          for (auto& v : ws) v = std::fabs(v) * 0.01f + 0.001f;
          for (auto& v : xs) v = std::fabs(v) * 0.01f + 0.001f;
          std::vector<float> a(size_t(M) * N), b(size_t(M) * N);
          gemm_nt_q8a8_naive(M, N, K, Xq.data(), xs.data(), Q.data(), ws.data(), a.data());
          gemm_nt_q8a8_simd(M, N, K, Xq.data(), xs.data(), Q.data(), ws.data(), b.data());
          for (size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-6 * (1 + std::fabs(a[i])));
        }
  }
  set_num_threads(num_threads());
}

// The vpmaddubsw trick must never saturate: every operand at +-127 is the worst case.
static void test_q8a8_extreme_values_do_not_saturate() {
  const int N = 6, K = 2048;
  std::vector<int8_t> Q = rand_i8(size_t(N) * K, 7, /*extremes=*/true), x(K, 127);
  for (int k = 0; k < K; ++k) Q[k] = 127, Q[K + k] = -127;  // row 0 all +127, row 1 all -127
  std::vector<float> ones(N, 1.f), y(N), ref(N);
  gemv_q8a8_simd(N, K, Q.data(), ones.data(), x.data(), 1.f, y.data());
  gemv_q8a8_naive(N, K, Q.data(), ones.data(), x.data(), 1.f, ref.data());
  CHECK(y[0] == 127.f * 127.f * K);  // 33,030,144 = 2048 * 16129: no int16 saturation on the way
  CHECK(y[1] == -127.f * 127.f * K);
  for (int i = 0; i < N; ++i) CHECK(y[i] == ref[i]);

  std::vector<int8_t> x2(size_t(2) * K, 127);  // the 2x4 GEMM micro-kernel has its own copy of the trick
  const float xs2[2] = {1.f, 1.f};
  std::vector<float> yg(size_t(2) * N);
  gemm_nt_q8a8_simd(2, N, K, x2.data(), xs2, Q.data(), ones.data(), yg.data());
  CHECK(yg[0] == y[0] && yg[N + 1] == y[1]);
}

static void test_gemm_nt_q8_dequant_pack_matches_fp32_reference() {
  for (int nt : {1, 4}) {
    set_num_threads(nt);
    for (auto [M, N, K] : {std::tuple{2, 5, 7}, {7, 33, 100}, {13, 130, 300}, {40, 64, 513}, {1, 50, 64}}) {
      auto W = randn(size_t(N) * K, 500 + N), X = randn(size_t(M) * K, 501 + M);
      std::vector<int8_t> Q(size_t(N) * K);
      std::vector<float> sc(N), Wd(size_t(N) * K);
      quantize_matrix_q8(W.data(), N, K, Q.data(), sc.data());
      for (int n = 0; n < N; ++n) dequantize_row_q8(Q.data() + size_t(n) * K, sc[n], K, Wd.data() + size_t(n) * K);
      std::vector<float> Y(size_t(M) * N, -1.f);
      gemm_nt_q8_simd(M, N, K, X.data(), Q.data(), sc.data(), Y.data());
      CHECK(ref_gemm_err(M, N, K, X.data(), K, Wd.data(), K, true, Y.data(), N) <= 1.0);
    }
  }
  set_num_threads(num_threads());
}

static void test_attention_helpers_match_scalar() {
  for (int n : {1, 7, 8, 9, 16, 17, 64, 65}) {
    auto a = randn(n, 700 + n), b = randn(n, 701 + n);
    CHECK_NEAR(dot_f32(a.data(), b.data(), n), dot_f32_naive(a.data(), b.data(), n), 1e-4);
    std::vector<float> y = randn(n, 702), y2 = y;
    axpy_f32(y.data(), -0.61f, a.data(), n);
    axpy_f32_naive(y2.data(), -0.61f, a.data(), n);
    for (int i = 0; i < n; ++i) CHECK_NEAR(y[i], y2[i], 1e-5);
  }
}

static void test_kv_int8_helpers() {
  for (int n : {1, 7, 8, 9, 64, 65}) {
    auto a = randn(n, 600 + n);
    auto q = rand_i8(n, 601 + n);
    double ref = 0;
    for (int i = 0; i < n; ++i) ref += double(a[i]) * q[i];
    CHECK_NEAR(dot_f32_i8(a.data(), q.data(), n), ref, 1e-3);
    std::vector<float> y = randn(n, 602), y0 = y;
    axpy_f32_i8(y.data(), 0.37f, q.data(), n);
    for (int i = 0; i < n; ++i) CHECK_NEAR(y[i], y0[i] + 0.37 * q[i], 1e-4);
  }
}

int main() {
  RUN(test_gemm_naive_matches_double_reference);
  RUN(test_gemv_matches_gemm_with_one_column);
  RUN(test_gemv_q8_matches_dequantised_f32);
  RUN(test_ops);
  RUN(test_arena_alignment_and_views);
  RUN(test_sampler);
  RUN(test_reordered_and_tiled_match_reference);
  RUN(test_gemm_simd_edge_shapes);
  RUN(test_gemm_simd_small_blocking);
  RUN(test_gemm_simd_padded_leading_dimensions);
  RUN(test_gemv_simd_all_shapes);
  RUN(test_gemm_nt_kernels_agree);
  RUN(test_simd_gemm_is_deterministic_across_thread_counts);
  RUN(test_quantize_properties);
  RUN(test_quantize_rounds_half_to_even);
  RUN(test_quantize_matrix_is_per_row);
  RUN(test_gemv_q8_simd_matches_scalar);
  RUN(test_q8a8_gemv_and_gemm_match_scalar);
  RUN(test_q8a8_extreme_values_do_not_saturate);
  RUN(test_gemm_nt_q8_dequant_pack_matches_fp32_reference);
  RUN(test_attention_helpers_match_scalar);
  RUN(test_kv_int8_helpers);
  return finish();
}
