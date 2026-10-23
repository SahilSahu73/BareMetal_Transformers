// Kernel 2: the GotoBLAS / BLIS recipe, scaled down.
//
//   for jc (nc columns of B)                 packed B block lives in L3
//     for pc (kc of the shared dim)
//       pack B[pc:pc+kc, jc:jc+nc] into NR-wide panels      <- parallel
//       pack A[:, pc:pc+kc]        into MR-tall panels      <- parallel
//       for (ic: mc rows of A) x (jr: chunk of B panels)    <- parallel tiles
//         for each B panel (kc x NR, stays in L1)
//           for each A panel (MR x kc, the mc-row block stays in L2)
//             micro-kernel: MR x NR tile of C held in registers across the whole kc loop
//
// The micro-kernel is where nearly all the speed comes from: it never touches C until the
// end and issues MR*NR/lanes independent FMAs per step, enough to keep both FMA ports busy.
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <thread>

#include "kernels.hpp"
#include "tensor.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define BM_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define BM_NEON 1
#endif

#ifdef _OPENMP
#include <omp.h>
#define BM_PRAGMA(x) _Pragma(#x)
#else
#define BM_PRAGMA(x)
#endif

namespace bm {
namespace {

#if BM_AVX2
// 16 ymm registers: 12 accumulators (6 rows x 2 vectors) + 2 for B + 1 broadcast of A.
constexpr int MR = 6, NR = 16;
#elif BM_NEON
// 32 q registers: 24 accumulators (8 rows x 3 vectors) + 3 for B + 2 for A.
constexpr int MR = 8, NR = 12;
#else
constexpr int MR = 4, NR = 8;  // portable fallback
#endif

constexpr int kTileCols = 128;  // columns of C per parallel task (a few B panels)

constexpr size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

// Packing buffers persist between calls so the hot path never allocates.
// Not safe for concurrent gemm calls from several threads.
struct AlignedBuf {
  float* p = nullptr;
  size_t cap = 0;
  float* get(size_t n) {
    if (n > cap) {
      std::free(p);
      p = static_cast<float*>(std::aligned_alloc(kAlign, round_up(n * sizeof(float))));
      cap = n;
    }
    return p;
  }
  ~AlignedBuf() { std::free(p); }
};
AlignedBuf g_apack, g_bpack;

// ---- micro-kernel: C[MR x NR] (+)= Apanel[kc x MR]^T * Bpanel[kc x NR] ----------------------
// a: kc steps of MR contiguous values; b: kc steps of NR contiguous values (both packed).
// mr/nr < MR/NR for edge tiles: the packed panels are zero padded, so we compute the full tile
// and only write back the valid part.
#if BM_AVX2
// The 12 accumulators are named variables, not an array: GCC keeps an array on the stack
// (every FMA then waits on a store-to-load round trip), but 12 scalars stay in ymm registers.
#define BM_ROW(r)                                          \
  {                                                        \
    const __m256 av = _mm256_broadcast_ss(a + (r));        \
    c##r##0 = _mm256_fmadd_ps(av, b0, c##r##0);            \
    c##r##1 = _mm256_fmadd_ps(av, b1, c##r##1);            \
  }
inline void ukernel(int kc, const float* a, const float* b, float* c, int ldc, int mr, int nr,
                    bool accumulate) {
  __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00;
  __m256 c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;

  for (int p = 0; p < kc; ++p) {
    const __m256 b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8);
    BM_ROW(0) BM_ROW(1) BM_ROW(2) BM_ROW(3) BM_ROW(4) BM_ROW(5)
    a += MR;
    b += NR;
  }
#undef BM_ROW

  const __m256 acc[MR][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}, {c40, c41}, {c50, c51}};
  if (mr == MR && nr == NR) {
    for (int r = 0; r < MR; ++r) {
      float* cr = c + static_cast<size_t>(r) * ldc;
      if (accumulate) {
        _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), acc[r][0]));
        _mm256_storeu_ps(cr + 8, _mm256_add_ps(_mm256_loadu_ps(cr + 8), acc[r][1]));
      } else {
        _mm256_storeu_ps(cr, acc[r][0]);
        _mm256_storeu_ps(cr + 8, acc[r][1]);
      }
    }
    return;
  }
  alignas(32) float tmp[MR * NR];
  for (int r = 0; r < MR; ++r) {
    _mm256_store_ps(tmp + r * NR, acc[r][0]);
    _mm256_store_ps(tmp + r * NR + 8, acc[r][1]);
  }
  for (int r = 0; r < mr; ++r)
    for (int j = 0; j < nr; ++j) {
      float& dst = c[static_cast<size_t>(r) * ldc + j];
      dst = accumulate ? dst + tmp[r * NR + j] : tmp[r * NR + j];
    }
}

#elif BM_NEON
inline void ukernel(int kc, const float* a, const float* b, float* c, int ldc, int mr, int nr,
                    bool accumulate) {
  float32x4_t acc[MR][3];
  for (int r = 0; r < MR; ++r)
    for (int j = 0; j < 3; ++j) acc[r][j] = vdupq_n_f32(0.f);

  for (int p = 0; p < kc; ++p) {
    const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
    const float32x4_t alo = vld1q_f32(a), ahi = vld1q_f32(a + 4);
#define BM_ROW(r, av, lane)                                  \
  acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, av, lane);      \
  acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, av, lane);      \
  acc[r][2] = vfmaq_laneq_f32(acc[r][2], b2, av, lane);
    BM_ROW(0, alo, 0) BM_ROW(1, alo, 1) BM_ROW(2, alo, 2) BM_ROW(3, alo, 3)
    BM_ROW(4, ahi, 0) BM_ROW(5, ahi, 1) BM_ROW(6, ahi, 2) BM_ROW(7, ahi, 3)
#undef BM_ROW
    a += MR;
    b += NR;
  }

  float tmp[MR * NR];
  for (int r = 0; r < MR; ++r)
    for (int j = 0; j < 3; ++j) vst1q_f32(tmp + r * NR + 4 * j, acc[r][j]);
  for (int r = 0; r < mr; ++r)
    for (int j = 0; j < nr; ++j) {
      float& dst = c[static_cast<size_t>(r) * ldc + j];
      dst = accumulate ? dst + tmp[r * NR + j] : tmp[r * NR + j];
    }
}

#else
inline void ukernel(int kc, const float* a, const float* b, float* c, int ldc, int mr, int nr,
                    bool accumulate) {
  float acc[MR][NR] = {};
  for (int p = 0; p < kc; ++p, a += MR, b += NR)
    for (int r = 0; r < MR; ++r)
      for (int j = 0; j < NR; ++j) acc[r][j] += a[r] * b[j];
  for (int r = 0; r < mr; ++r)
    for (int j = 0; j < nr; ++j) {
      float& dst = c[static_cast<size_t>(r) * ldc + j];
      dst = accumulate ? dst + acc[r][j] : acc[r][j];
    }
}
#endif

// A panel: MR rows x kc, stored k-major so the micro-kernel reads MR consecutive values per step.
void pack_a_panel(const float* A, int lda, int i0, int mr, int pc, int kc, float* dst) {
  for (int r = 0; r < mr; ++r) {
    const float* src = A + static_cast<size_t>(i0 + r) * lda + pc;  // contiguous read
    for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * MR + r] = src[p];
  }
  for (int r = mr; r < MR; ++r)
    for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * MR + r] = 0.f;
}

// B panel: kc x NR columns, stored k-major so the micro-kernel reads NR consecutive values per step.
void pack_b_panel(const float* B, int ldb, bool trans, int j0, int nr, int pc, int kc, float* dst) {
  if (!trans) {  // B[k, j]
    for (int p = 0; p < kc; ++p) {
      const float* src = B + static_cast<size_t>(pc + p) * ldb + j0;
      float* d = dst + static_cast<size_t>(p) * NR;
      std::memcpy(d, src, sizeof(float) * nr);
      for (int c = nr; c < NR; ++c) d[c] = 0.f;
    }
  } else {  // B[k, j] stored at W[j, k]: read each output channel's weights contiguously
    for (int c = 0; c < nr; ++c) {
      const float* src = B + static_cast<size_t>(j0 + c) * ldb + pc;
      for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = src[p];
    }
    for (int c = nr; c < NR; ++c)
      for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = 0.f;
  }
}

// INT8 weights packed straight into FP32 panels: B[k, j] = Q[j, k] * scale[j]. The dequantisation
// rides along with the copy we already do, so the micro-kernel is the unchanged FP32 one.
void pack_b_panel_q8(const int8_t* Q, const float* scales, int ldq, int j0, int nr, int pc, int kc, float* dst) {
  for (int c = 0; c < nr; ++c) {
    const int8_t* src = Q + static_cast<size_t>(j0 + c) * ldq + pc;
    const float sc = scales[j0 + c];
    for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = static_cast<float>(src[p]) * sc;
  }
  for (int c = nr; c < NR; ++c)
    for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = 0.f;
}

}  // namespace

// B is read either as floats (B / ldb / b_transposed) or, when bq != nullptr, as the int8 matrix
// Q[N, K] with per-row scales (always "transposed": a weight matrix).
static void gemm_core(int M, int N, int K, const float* A, int lda, const float* B, int ldb, bool b_transposed,
                      const int8_t* bq, const float* bscales, float* C, int ldc, const GemmBlocking& blk) {
  if (M <= 0 || N <= 0) return;
  if (K <= 0) {
    for (int i = 0; i < M; ++i) std::memset(C + static_cast<size_t>(i) * ldc, 0, sizeof(float) * N);
    return;
  }
  const int mc = std::max(MR, blk.mc / MR * MR);
  const int kc = std::max(1, blk.kc);
  const int nc = std::max(NR, blk.nc / NR * NR);
  const int tile_cols = static_cast<int>(ceil_div(kTileCols, NR)) * NR;

  const int m_panels = static_cast<int>(ceil_div(M, MR));
  float* apack = g_apack.get(static_cast<size_t>(m_panels) * MR * std::min(kc, K));
  float* bpack = g_bpack.get(static_cast<size_t>(std::min<size_t>(nc, round_up(N, NR))) * std::min(kc, K));
  const bool parallel = static_cast<double>(M) * N * K >= 2e6;  // tiny products are not worth a thread team

  // Every thread runs the same loop nest; the omp-for worksharing loops split the packing and
  // the tiles, and their implicit barriers order "pack" before "compute" before "re-pack".
  BM_PRAGMA(omp parallel if (parallel))
  {
    for (int jc = 0; jc < N; jc += nc) {
      const int ncur = std::min(nc, N - jc);
      const int b_panels = static_cast<int>(ceil_div(ncur, NR));
      for (int pc = 0; pc < K; pc += kc) {
        const int kcur = std::min(kc, K - pc);
        const bool accumulate = pc > 0;

        BM_PRAGMA(omp for schedule(static))
        for (int jp = 0; jp < b_panels; ++jp)
          if (bq)
            pack_b_panel_q8(bq, bscales, K, jc + jp * NR, std::min(NR, ncur - jp * NR), pc, kcur,
                            bpack + static_cast<size_t>(jp) * NR * kcur);
          else
            pack_b_panel(B, ldb, b_transposed, jc + jp * NR, std::min(NR, ncur - jp * NR), pc, kcur,
                         bpack + static_cast<size_t>(jp) * NR * kcur);

        BM_PRAGMA(omp for schedule(static))
        for (int ip = 0; ip < m_panels; ++ip)
          pack_a_panel(A, lda, ip * MR, std::min(MR, M - ip * MR), pc, kcur,
                       apack + static_cast<size_t>(ip) * MR * kcur);

        const int n_ic = static_cast<int>(ceil_div(M, mc));
        const int n_jr = static_cast<int>(ceil_div(ncur, tile_cols));
        BM_PRAGMA(omp for collapse(2) schedule(dynamic))
        for (int ic = 0; ic < n_ic; ++ic)
          for (int jr = 0; jr < n_jr; ++jr) {
            const int i_end = std::min(M, (ic + 1) * mc);
            const int j_end = std::min(ncur, (jr + 1) * tile_cols);
            for (int j = jr * tile_cols; j < j_end; j += NR) {  // B panel stays in L1 ...
              const float* bp = bpack + static_cast<size_t>(j / NR) * NR * kcur;
              const int nr = std::min(NR, ncur - j);
              for (int i = ic * mc; i < i_end; i += MR) {       // ... while A panels stream from L2
                const float* ap = apack + static_cast<size_t>(i / MR) * MR * kcur;
                ukernel(kcur, ap, bp, C + static_cast<size_t>(i) * ldc + jc + j, ldc,
                        std::min(MR, M - i), nr, accumulate);
              }
            }
          }
      }
    }
  }
}

void gemm_simd_ld(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                  bool b_transposed, float* C, int ldc, const GemmBlocking& blk) {
  gemm_core(M, N, K, A, lda, B, ldb, b_transposed, nullptr, nullptr, C, ldc, blk);
}

void gemm_nt_q8_simd(int M, int N, int K, const float* X, const int8_t* Q, const float* wscales, float* Y) {
  if (M == 1)
    gemv_q8_simd(N, K, Q, wscales, X, Y);
  else
    gemm_core(M, N, K, X, K, nullptr, K, true, Q, wscales, Y, N, GemmBlocking{});
}

void gemm_simd(int M, int N, int K, const float* A, const float* B, float* C, bool b_transposed,
               const GemmBlocking& blk) {
  gemm_simd_ld(M, N, K, A, K, B, b_transposed ? K : N, b_transposed, C, N, blk);
}

void gemm_nt_simd(int M, int N, int K, const float* X, const float* W, float* Y) {
  if (M == 1)
    gemv_f32_simd(N, K, W, X, Y);
  else
    gemm_simd(M, N, K, X, W, Y, /*b_transposed=*/true);
}

// ---- GEMV ------------------------------------------------------------------------------------
namespace {

#if BM_AVX2
inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

// rows [n0, n1) of y = W x. Four rows per pass share every load of x; two accumulators per
// row give 8 independent FMA chains.
void gemv_rows(const float* W, int K, const float* x, float* y, int n0, int n1) {
  int n = n0;
  for (; n + 4 <= n1; n += 4) {
    const float* w0 = W + static_cast<size_t>(n) * K;
    const float *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    int k = 0;
    for (; k + 16 <= K; k += 16) {
      const __m256 xa = _mm256_loadu_ps(x + k), xb = _mm256_loadu_ps(x + k + 8);
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + k), xa, a0);
      b0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + k + 8), xb, b0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + k), xa, a1);
      b1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + k + 8), xb, b1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + k), xa, a2);
      b2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + k + 8), xb, b2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + k), xa, a3);
      b3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + k + 8), xb, b3);
    }
    for (; k + 8 <= K; k += 8) {
      const __m256 xa = _mm256_loadu_ps(x + k);
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + k), xa, a0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + k), xa, a1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + k), xa, a2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + k), xa, a3);
    }
    float s0 = hsum(_mm256_add_ps(a0, b0)), s1 = hsum(_mm256_add_ps(a1, b1));
    float s2 = hsum(_mm256_add_ps(a2, b2)), s3 = hsum(_mm256_add_ps(a3, b3));
    for (; k < K; ++k) {
      s0 += w0[k] * x[k]; s1 += w1[k] * x[k]; s2 += w2[k] * x[k]; s3 += w3[k] * x[k];
    }
    y[n] = s0; y[n + 1] = s1; y[n + 2] = s2; y[n + 3] = s3;
  }
  for (; n < n1; ++n) {
    const float* w = W + static_cast<size_t>(n) * K;
    __m256 a = _mm256_setzero_ps(), b = a;
    int k = 0;
    for (; k + 16 <= K; k += 16) {
      a = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + k), a);
      b = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 8), _mm256_loadu_ps(x + k + 8), b);
    }
    for (; k + 8 <= K; k += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + k), a);
    float s = hsum(_mm256_add_ps(a, b));
    for (; k < K; ++k) s += w[k] * x[k];
    y[n] = s;
  }
}

#elif BM_NEON
void gemv_rows(const float* W, int K, const float* x, float* y, int n0, int n1) {
  for (int n = n0; n < n1; ++n) {
    const float* w = W + static_cast<size_t>(n) * K;
    float32x4_t a = vdupq_n_f32(0.f), b = a, c = a, d = a;
    int k = 0;
    for (; k + 16 <= K; k += 16) {
      a = vfmaq_f32(a, vld1q_f32(w + k), vld1q_f32(x + k));
      b = vfmaq_f32(b, vld1q_f32(w + k + 4), vld1q_f32(x + k + 4));
      c = vfmaq_f32(c, vld1q_f32(w + k + 8), vld1q_f32(x + k + 8));
      d = vfmaq_f32(d, vld1q_f32(w + k + 12), vld1q_f32(x + k + 12));
    }
    for (; k + 4 <= K; k += 4) a = vfmaq_f32(a, vld1q_f32(w + k), vld1q_f32(x + k));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    for (; k < K; ++k) s += w[k] * x[k];
    y[n] = s;
  }
}

#else
void gemv_rows(const float* W, int K, const float* x, float* y, int n0, int n1) {
  for (int n = n0; n < n1; ++n) {
    const float* w = W + static_cast<size_t>(n) * K;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int k = 0;
    for (; k + 4 <= K; k += 4) {
      s0 += w[k] * x[k]; s1 += w[k + 1] * x[k + 1]; s2 += w[k + 2] * x[k + 2]; s3 += w[k + 3] * x[k + 3];
    }
    for (; k < K; ++k) s0 += w[k] * x[k];
    y[n] = (s0 + s1) + (s2 + s3);
  }
}
#endif

}  // namespace

size_t& gemv_parallel_threshold() {
  static size_t t = 1u << 18;  // weights elements (1 MB of FP32); below this threads cost more than they save
  return t;
}

void gemv_f32_simd(int N, int K, const float* W, const float* x, float* y) {
  const size_t work = static_cast<size_t>(N) * K;
  const int blocks = static_cast<int>(ceil_div(N, 4));
  BM_PRAGMA(omp parallel for schedule(static) if (work >= gemv_parallel_threshold()))
  for (int b = 0; b < blocks; ++b) gemv_rows(W, K, x, y, b * 4, std::min(N, b * 4 + 4));
}

float dot_f32(const float* a, const float* b, int n) {
  int i = 0;
  float s = 0.f;
#if BM_AVX2
  __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  for (; i + 8 <= n; i += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(_mm256_add_ps(acc0, acc1)), _mm256_extractf128_ps(_mm256_add_ps(acc0, acc1), 1));
  h = _mm_hadd_ps(h, h);
  h = _mm_hadd_ps(h, h);
  s = _mm_cvtss_f32(h);
#elif BM_NEON
  float32x4_t acc = vdupq_n_f32(0.f), acc1 = acc;
  for (; i + 8 <= n; i += 8) {
    acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  s = vaddvq_f32(vaddq_f32(acc, acc1));
#endif
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

void axpy_f32(float* y, float alpha, const float* x, int n) {
  int i = 0;
#if BM_AVX2
  const __m256 va = _mm256_set1_ps(alpha);
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
#elif BM_NEON
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_n_f32(vld1q_f32(y + i), vld1q_f32(x + i), alpha));
#endif
  for (; i < n; ++i) y[i] += alpha * x[i];
}

KernelTable optimized_kernels() {
  return {gemv_f32_simd, gemv_q8_simd, gemm_nt_simd, gemm_nt_q8_simd, gemv_q8a8_simd, gemm_nt_q8a8_simd,
          dot_f32,       axpy_f32,     dot_f32_i8,   axpy_f32_i8};
}

const char* simd_isa() {
#if BM_AVX2
  return "AVX2+FMA";
#elif BM_NEON
  return "NEON";
#else
  return "scalar";
#endif
}
int micro_tile_m() { return MR; }
int micro_tile_n() { return NR; }

void set_num_threads(int n) {
#ifdef _OPENMP
  omp_set_num_threads(std::max(1, n));
#else
  (void)n;
#endif
}
int num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int physical_cores() {
  std::set<std::string> cores;  // one entry per group of SMT siblings
  const int hw = static_cast<int>(std::thread::hardware_concurrency());
  for (int i = 0; i < hw; ++i) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
    std::string line;
    if (!f || !std::getline(f, line)) return std::max(1, hw);  // not Linux / no sysfs
    cores.insert(line);
  }
  return cores.empty() ? std::max(1, hw) : static_cast<int>(cores.size());
}

}  // namespace bm
