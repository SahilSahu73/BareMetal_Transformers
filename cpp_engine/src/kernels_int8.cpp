// INT8 kernels.
//
// W8A32 GEMV (decode): weights are int8 + one scale per row, activations stay FP32. Decoding is
//   memory bound, so the win is moving 4x fewer weight bytes; the int8->fp32 conversion is cheap
//   next to the memory stalls it removes.
//
// W8A8 (integer dot products): activations are quantised per vector, and the inner loop is
//   vpabsb / vpsignb / vpmaddubsw / vpmaddwd / vpaddd: 32 multiply-accumulates per 5 instructions.
//   vpmaddubsw wants one unsigned and one signed operand, so we multiply |w| (unsigned) by
//   sign(x, w) (the activation with w's sign applied): the product is unchanged. Each int16 pair
//   sum is at most 2 * 127 * 127 = 32258 < 32767, so the saturating add never saturates -
//   provided no operand is -128, which symmetric quantisation guarantees.
#include <algorithm>
#include <cstddef>

#include "kernels.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define BM_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define BM_NEON 1
#endif

#ifdef _OPENMP
#define BM_PRAGMA(x) _Pragma(#x)
#else
#define BM_PRAGMA(x)
#endif

namespace bm {

#if BM_AVX2
namespace {

inline float hsum_ps(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

inline int hsum_epi32(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

// ---- W8A32 GEMV ----------------------------------------------------------------------------
void rows_w8a32(const int8_t* Q, const float* scales, int K, const float* x, float* y, int n0, int n1) {
  int n = n0;
  for (; n + 4 <= n1; n += 4) {
    const int8_t* w0 = Q + static_cast<size_t>(n) * K;
    const int8_t *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    int k = 0;
    for (; k + 16 <= K; k += 16) {
      const __m256 xa = _mm256_loadu_ps(x + k), xb = _mm256_loadu_ps(x + k + 8);
#define BM_STEP(w, a, b)                                                                        \
  {                                                                                             \
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k));                 \
    a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), xa, a);                    \
    b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8))), xb, b); \
  }
      BM_STEP(w0, a0, b0) BM_STEP(w1, a1, b1) BM_STEP(w2, a2, b2) BM_STEP(w3, a3, b3)
#undef BM_STEP
    }
    for (; k + 8 <= K; k += 8) {
      const __m256 xa = _mm256_loadu_ps(x + k);
#define BM_STEP8(w, a) \
  a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k)))), xa, a);
      BM_STEP8(w0, a0) BM_STEP8(w1, a1) BM_STEP8(w2, a2) BM_STEP8(w3, a3)
#undef BM_STEP8
    }
    float s0 = hsum_ps(_mm256_add_ps(a0, b0)), s1 = hsum_ps(_mm256_add_ps(a1, b1));
    float s2 = hsum_ps(_mm256_add_ps(a2, b2)), s3 = hsum_ps(_mm256_add_ps(a3, b3));
    for (; k < K; ++k) {
      s0 += w0[k] * x[k]; s1 += w1[k] * x[k]; s2 += w2[k] * x[k]; s3 += w3[k] * x[k];
    }
    y[n] = s0 * scales[n]; y[n + 1] = s1 * scales[n + 1]; y[n + 2] = s2 * scales[n + 2]; y[n + 3] = s3 * scales[n + 3];
  }
  for (; n < n1; ++n) {
    const int8_t* w = Q + static_cast<size_t>(n) * K;
    __m256 a = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= K; k += 8)
      a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k)))),
                          _mm256_loadu_ps(x + k), a);
    float s = hsum_ps(a);
    for (; k < K; ++k) s += w[k] * x[k];
    y[n] = s * scales[n];
  }
}

// ---- W8A8 integer dot products ---------------------------------------------------------------
// One 32-element step of an int8 dot product into an int32 accumulator.
#define BM_DOT32(acc, wv, xv)                                                              \
  acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(wv),  \
                                                                     _mm256_sign_epi8(xv, wv)), ones))

inline int dot_i8(const int8_t* w, const int8_t* x, int K) {
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i acc = _mm256_setzero_si256();
  int k = 0;
  for (; k + 32 <= K; k += 32) {
    const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k));
    const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k));
    BM_DOT32(acc, wv, xv);
  }
  int s = hsum_epi32(acc);
  for (; k < K; ++k) s += static_cast<int>(w[k]) * x[k];
  return s;
}

// rows [n0, n1) of y = W x with an int8 x: four weight rows share each load of x.
void rows_w8a8(const int8_t* Q, const float* wscales, int K, const int8_t* xq, float xscale, float* y, int n0, int n1) {
  const __m256i ones = _mm256_set1_epi16(1);
  int n = n0;
  for (; n + 4 <= n1; n += 4) {
    const int8_t* w0 = Q + static_cast<size_t>(n) * K;
    const int8_t *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    int k = 0;
    for (; k + 32 <= K; k += 32) {
      const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + k));
#define BM_ROW(w, a)                                                                      \
  {                                                                                       \
    const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k));       \
    BM_DOT32(a, wv, xv);                                                                  \
  }
      BM_ROW(w0, a0) BM_ROW(w1, a1) BM_ROW(w2, a2) BM_ROW(w3, a3)
#undef BM_ROW
    }
    int s0 = hsum_epi32(a0), s1 = hsum_epi32(a1), s2 = hsum_epi32(a2), s3 = hsum_epi32(a3);
    for (; k < K; ++k) {
      s0 += w0[k] * xq[k]; s1 += w1[k] * xq[k]; s2 += w2[k] * xq[k]; s3 += w3[k] * xq[k];
    }
    y[n] = static_cast<float>(s0) * wscales[n] * xscale;
    y[n + 1] = static_cast<float>(s1) * wscales[n + 1] * xscale;
    y[n + 2] = static_cast<float>(s2) * wscales[n + 2] * xscale;
    y[n + 3] = static_cast<float>(s3) * wscales[n + 3] * xscale;
  }
  for (; n < n1; ++n)
    y[n] = static_cast<float>(dot_i8(Q + static_cast<size_t>(n) * K, xq, K)) * wscales[n] * xscale;
}

// 2 activation rows x 4 weight rows: 8 int32 accumulators; every weight load is reused for two
// activation rows and |w| is computed once per weight row.
inline void micro_2x4(const int8_t* x0, const int8_t* x1, const int8_t* w, int K, int out[8]) {
  const __m256i ones = _mm256_set1_epi16(1);
  const int8_t *w0 = w, *w1 = w + K, *w2 = w + 2 * static_cast<size_t>(K), *w3 = w + 3 * static_cast<size_t>(K);
  __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  int k = 0;
  for (; k + 32 <= K; k += 32) {
    const __m256i xa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + k));
    const __m256i xb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x1 + k));
#define BM_COL(j)                                                                                   \
  {                                                                                                 \
    const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w##j + k));              \
    const __m256i aw = _mm256_abs_epi8(wv);                                                         \
    c0##j = _mm256_add_epi32(c0##j, _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xa, wv)), ones)); \
    c1##j = _mm256_add_epi32(c1##j, _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xb, wv)), ones)); \
  }
    BM_COL(0) BM_COL(1) BM_COL(2) BM_COL(3)
#undef BM_COL
  }
  out[0] = hsum_epi32(c00); out[1] = hsum_epi32(c01); out[2] = hsum_epi32(c02); out[3] = hsum_epi32(c03);
  out[4] = hsum_epi32(c10); out[5] = hsum_epi32(c11); out[6] = hsum_epi32(c12); out[7] = hsum_epi32(c13);
  for (; k < K; ++k)
    for (int j = 0; j < 4; ++j) {
      out[j] += static_cast<int>(x0[k]) * w[static_cast<size_t>(j) * K + k];
      out[4 + j] += static_cast<int>(x1[k]) * w[static_cast<size_t>(j) * K + k];
    }
}

}  // namespace

void gemv_q8_simd(int N, int K, const int8_t* Q, const float* scales, const float* x, float* y) {
  const int blocks = (N + 3) / 4;
  BM_PRAGMA(omp parallel for schedule(static) if (static_cast<size_t>(N) * K >= gemv_parallel_threshold()))
  for (int b = 0; b < blocks; ++b) rows_w8a32(Q, scales, K, x, y, b * 4, std::min(N, b * 4 + 4));
}

void gemv_q8a8_simd(int N, int K, const int8_t* Q, const float* wscales, const int8_t* xq, float xscale, float* y) {
  const int blocks = (N + 3) / 4;
  BM_PRAGMA(omp parallel for schedule(static) if (static_cast<size_t>(N) * K >= gemv_parallel_threshold()))
  for (int b = 0; b < blocks; ++b) rows_w8a8(Q, wscales, K, xq, xscale, y, b * 4, std::min(N, b * 4 + 4));
}

void gemm_nt_q8a8_simd(int M, int N, int K, const int8_t* Xq, const float* xscales, const int8_t* Q,
                       const float* wscales, float* Y) {
  if (M == 1) {
    gemv_q8a8_simd(N, K, Q, wscales, Xq, xscales[0], Y);
    return;
  }
  // A block of 32 weight rows (64 KB at K = 2048) stays in L2 while every activation pair sweeps it.
  constexpr int kBlock = 32;
  const int blocks = (N + kBlock - 1) / kBlock;
  BM_PRAGMA(omp parallel for schedule(dynamic) if (static_cast<double>(M) * N * K >= 1e6))
  for (int nb = 0; nb < blocks; ++nb) {
    const int n0 = nb * kBlock, n1 = std::min(N, n0 + kBlock);
    int m = 0;
    for (; m + 2 <= M; m += 2) {
      const int8_t* x0 = Xq + static_cast<size_t>(m) * K;
      const int8_t* x1 = x0 + K;
      int n = n0;
      for (; n + 4 <= n1; n += 4) {
        int acc[8];
        micro_2x4(x0, x1, Q + static_cast<size_t>(n) * K, K, acc);
        for (int j = 0; j < 4; ++j) {
          Y[static_cast<size_t>(m) * N + n + j] = static_cast<float>(acc[j]) * xscales[m] * wscales[n + j];
          Y[static_cast<size_t>(m + 1) * N + n + j] = static_cast<float>(acc[4 + j]) * xscales[m + 1] * wscales[n + j];
        }
      }
      for (; n < n1; ++n) {  // weight-row remainder
        const int8_t* w = Q + static_cast<size_t>(n) * K;
        Y[static_cast<size_t>(m) * N + n] = static_cast<float>(dot_i8(w, x0, K)) * xscales[m] * wscales[n];
        Y[static_cast<size_t>(m + 1) * N + n] = static_cast<float>(dot_i8(w, x1, K)) * xscales[m + 1] * wscales[n];
      }
    }
    for (; m < M; ++m)  // odd last activation row
      for (int n = n0; n < n1; ++n)
        Y[static_cast<size_t>(m) * N + n] =
            static_cast<float>(dot_i8(Q + static_cast<size_t>(n) * K, Xq + static_cast<size_t>(m) * K, K)) * xscales[m] * wscales[n];
  }
}

float dot_f32_i8(const float* a, const int8_t* b, int n) {
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                          _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i)))), acc);
  float s = hsum_ps(acc);
  for (; i < n; ++i) s += a[i] * static_cast<float>(b[i]);
  return s;
}

void axpy_f32_i8(float* y, float alpha, const int8_t* x, int n) {
  const __m256 va = _mm256_set1_ps(alpha);
  int i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va,
                     _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x + i)))),
                     _mm256_loadu_ps(y + i)));
  for (; i < n; ++i) y[i] += alpha * static_cast<float>(x[i]);
}

#elif BM_NEON
// ---- NEON (baseline ARMv8, no dot-product extension required) ---------------------------------
namespace {
inline int dot_i8(const int8_t* w, const int8_t* x, int K) {
  int32x4_t acc = vdupq_n_s32(0);
  int k = 0;
  for (; k + 16 <= K; k += 16) {
    const int8x16_t wv = vld1q_s8(w + k), xv = vld1q_s8(x + k);
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(wv), vget_low_s8(xv)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(wv), vget_high_s8(xv)));
  }
  int s = vaddvq_s32(acc);
  for (; k < K; ++k) s += static_cast<int>(w[k]) * x[k];
  return s;
}
}  // namespace

void gemv_q8_simd(int N, int K, const int8_t* Q, const float* scales, const float* x, float* y) {
  BM_PRAGMA(omp parallel for schedule(static) if (static_cast<size_t>(N) * K >= gemv_parallel_threshold()))
  for (int n = 0; n < N; ++n) {
    const int8_t* w = Q + static_cast<size_t>(n) * K;
    float32x4_t a = vdupq_n_f32(0.f), b = a;
    int k = 0;
    for (; k + 8 <= K; k += 8) {
      const int16x8_t w16 = vmovl_s8(vld1_s8(w + k));
      a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))), vld1q_f32(x + k));
      b = vfmaq_f32(b, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))), vld1q_f32(x + k + 4));
    }
    float s = vaddvq_f32(vaddq_f32(a, b));
    for (; k < K; ++k) s += w[k] * x[k];
    y[n] = s * scales[n];
  }
}

void gemv_q8a8_simd(int N, int K, const int8_t* Q, const float* wscales, const int8_t* xq, float xscale, float* y) {
  BM_PRAGMA(omp parallel for schedule(static) if (static_cast<size_t>(N) * K >= gemv_parallel_threshold()))
  for (int n = 0; n < N; ++n)
    y[n] = static_cast<float>(dot_i8(Q + static_cast<size_t>(n) * K, xq, K)) * wscales[n] * xscale;
}

// Not register-blocked on NEON: correct and threaded, but untuned.
void gemm_nt_q8a8_simd(int M, int N, int K, const int8_t* Xq, const float* xscales, const int8_t* Q,
                       const float* wscales, float* Y) {
  BM_PRAGMA(omp parallel for schedule(static) if (static_cast<double>(M) * N * K >= 1e6))
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      Y[static_cast<size_t>(m) * N + n] =
          static_cast<float>(dot_i8(Q + static_cast<size_t>(n) * K, Xq + static_cast<size_t>(m) * K, K)) * xscales[m] * wscales[n];
}

float dot_f32_i8(const float* a, const int8_t* b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; ++i) s += a[i] * static_cast<float>(b[i]);
  return s;
}
void axpy_f32_i8(float* y, float alpha, const int8_t* x, int n) {
  for (int i = 0; i < n; ++i) y[i] += alpha * static_cast<float>(x[i]);
}

#else
// ---- portable fallback: the scalar reference kernels ------------------------------------------
void gemv_q8_simd(int N, int K, const int8_t* Q, const float* scales, const float* x, float* y) { gemv_q8_naive(N, K, Q, scales, x, y); }
void gemv_q8a8_simd(int N, int K, const int8_t* Q, const float* ws, const int8_t* xq, float xs, float* y) { gemv_q8a8_naive(N, K, Q, ws, xq, xs, y); }
void gemm_nt_q8a8_simd(int M, int N, int K, const int8_t* Xq, const float* xs, const int8_t* Q, const float* ws, float* Y) {
  gemm_nt_q8a8_naive(M, N, K, Xq, xs, Q, ws, Y);
}
float dot_f32_i8(const float* a, const int8_t* b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; ++i) s += a[i] * static_cast<float>(b[i]);
  return s;
}
void axpy_f32_i8(float* y, float alpha, const int8_t* x, int n) {
  for (int i = 0; i < n; ++i) y[i] += alpha * static_cast<float>(x[i]);
}
#endif

}  // namespace bm
