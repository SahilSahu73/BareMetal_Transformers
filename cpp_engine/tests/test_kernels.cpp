#include <algorithm>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include "kernels.hpp"
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
  return finish();
}
