// The multiplier-less core: integer accumulation of INT8 activations selected by ternary weights.
//
// This file is deliberately self-contained and must stay free of multiplications - the build
// disassembles its object code and fails if a multiply/FMA instruction appears
// (scripts/check_multiplierless.sh). The only arithmetic in the hot loops is and / sub / xor / add
// and a sum-of-absolute-differences against zero, which is an adder tree.
//
// AVX2 step, 32 weights (8 packed bytes) x 32 INT8 activations:
//   1. broadcast each packed byte to its 4 lanes (pshufb), AND with that lane's bit-pair mask,
//      compare with the "+1" and "-1" codes -> byte masks pos / neg           (no multiply)
//   2. d = (x & pos) - (x & neg)           : x, -x or 0 per lane, in [-127, 127]
//   3. u = d ^ 0x80                        : d + 128 as unsigned, so psadbw can add it
//   4. acc64 += psadbw(u, 0)               : horizontal add of 8 bytes per 64-bit lane
// The +128 bias adds 128 * 32 per step and is subtracted once at the end.
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "ternary.hpp"

#if defined(__AVX2__)
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
namespace {

inline unsigned code_at(const uint8_t* row, int k) { return (row[k >> 2] >> ((k & 3) * 2)) & 3u; }

// weights [k0, K) of one row, one at a time: add, subtract or skip
inline int32_t tail_dot(const uint8_t* row, const int8_t* x, int k0, int K) {
  int32_t acc = 0;
  for (int k = k0; k < K; ++k) {
    const unsigned c = code_at(row, k);
    if (c == 1u) acc += x[k];
    else if (c == 2u) acc -= x[k];
  }
  return acc;
}

#if BM_AVX2
struct Consts {
  __m256i shuf, sel, pos, neg, bias, zero;
};
inline Consts make_consts() {
  Consts c;
  c.shuf = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
  c.sel = _mm256_set1_epi32(static_cast<int>(0xC0300C03u));  // lane i: 0x03 << 2*(i % 4)
  c.pos = _mm256_set1_epi32(0x40100401);                     // code 01 in each bit pair
  c.neg = _mm256_set1_epi32(static_cast<int>(0x80200802u));  // code 10 in each bit pair
  c.bias = _mm256_set1_epi8(static_cast<char>(0x80));
  c.zero = _mm256_setzero_si256();
  return c;
}

// 8 packed bytes -> byte masks for 32 weights: pos lane = 0xFF where w = +1, neg lane = 0xFF where w = -1
inline void masks(const uint8_t* p, const Consts& c, __m256i& pos, __m256i& neg) {
  const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
  const __m256i v = _mm256_inserti128_si256(_mm256_castsi128_si256(b), _mm_srli_si128(b, 4), 1);
  const __m256i m = _mm256_and_si256(_mm256_shuffle_epi8(v, c.shuf), c.sel);
  pos = _mm256_cmpeq_epi8(m, c.pos);
  neg = _mm256_cmpeq_epi8(m, c.neg);
}

inline __m256i accumulate(__m256i acc, __m256i x, __m256i pos, __m256i neg, const Consts& c) {
  const __m256i d = _mm256_sub_epi8(_mm256_and_si256(x, pos), _mm256_and_si256(x, neg));
  return _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_xor_si256(d, c.bias), c.zero));
}

inline int32_t finish(__m256i acc, int blocks) {
  alignas(32) int64_t lanes[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
  return static_cast<int32_t>(lanes[0] + lanes[1] + lanes[2] + lanes[3] - static_cast<int64_t>(blocks) * 32 * 128);
}

// rows [n0, n1) of acc = W x; four rows share every load of x
void gemv_rows(const uint8_t* Wp, int K, const int8_t* x, int32_t* y, int n0, int n1) {
  const Consts c = make_consts();
  const int nb = K / kTernaryBlock, rb = ternary_row_bytes(K);
  int n = n0;
  for (; n + 4 <= n1; n += 4) {
    const uint8_t* w0 = Wp + static_cast<size_t>(n) * rb;
    const uint8_t *w1 = w0 + rb, *w2 = w1 + rb, *w3 = w2 + rb;
    __m256i a0 = c.zero, a1 = c.zero, a2 = c.zero, a3 = c.zero;
    for (int b = 0; b < nb; ++b) {
      const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + b * kTernaryBlock));
      __m256i p, q;
      masks(w0 + b * 8, c, p, q); a0 = accumulate(a0, xv, p, q, c);
      masks(w1 + b * 8, c, p, q); a1 = accumulate(a1, xv, p, q, c);
      masks(w2 + b * 8, c, p, q); a2 = accumulate(a2, xv, p, q, c);
      masks(w3 + b * 8, c, p, q); a3 = accumulate(a3, xv, p, q, c);
    }
    y[n] = finish(a0, nb) + tail_dot(w0, x, nb * kTernaryBlock, K);
    y[n + 1] = finish(a1, nb) + tail_dot(w1, x, nb * kTernaryBlock, K);
    y[n + 2] = finish(a2, nb) + tail_dot(w2, x, nb * kTernaryBlock, K);
    y[n + 3] = finish(a3, nb) + tail_dot(w3, x, nb * kTernaryBlock, K);
  }
  for (; n < n1; ++n) {
    const uint8_t* w = Wp + static_cast<size_t>(n) * rb;
    __m256i a = c.zero;
    for (int b = 0; b < nb; ++b) {
      __m256i p, q;
      masks(w + b * 8, c, p, q);
      a = accumulate(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + b * kTernaryBlock)), p, q, c);
    }
    y[n] = finish(a, nb) + tail_dot(w, x, nb * kTernaryBlock, K);
  }
}

// one weight row against tokens [m0, m1): the unpacked masks are computed once per block and
// reused for every token, four tokens at a time
void gemm_row(const uint8_t* w, int K, const int8_t* X, int M, int N, int n, int32_t* acc, int m0, int m1) {
  const Consts c = make_consts();
  const int nb = K / kTernaryBlock;
  int m = m0;
  for (; m + 4 <= m1; m += 4) {
    const int8_t *x0 = X + static_cast<size_t>(m) * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
    __m256i a0 = c.zero, a1 = c.zero, a2 = c.zero, a3 = c.zero;
    for (int b = 0; b < nb; ++b) {
      __m256i p, q;
      masks(w + b * 8, c, p, q);
      const size_t off = static_cast<size_t>(b) * kTernaryBlock;
      a0 = accumulate(a0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + off)), p, q, c);
      a1 = accumulate(a1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x1 + off)), p, q, c);
      a2 = accumulate(a2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x2 + off)), p, q, c);
      a3 = accumulate(a3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x3 + off)), p, q, c);
    }
    acc[static_cast<size_t>(m) * N + n] = finish(a0, nb) + tail_dot(w, x0, nb * kTernaryBlock, K);
    acc[static_cast<size_t>(m + 1) * N + n] = finish(a1, nb) + tail_dot(w, x1, nb * kTernaryBlock, K);
    acc[static_cast<size_t>(m + 2) * N + n] = finish(a2, nb) + tail_dot(w, x2, nb * kTernaryBlock, K);
    acc[static_cast<size_t>(m + 3) * N + n] = finish(a3, nb) + tail_dot(w, x3, nb * kTernaryBlock, K);
  }
  for (; m < m1; ++m) {
    const int8_t* x = X + static_cast<size_t>(m) * K;
    __m256i a = c.zero;
    for (int b = 0; b < nb; ++b) {
      __m256i p, q;
      masks(w + b * 8, c, p, q);
      a = accumulate(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + static_cast<size_t>(b) * kTernaryBlock)), p, q, c);
    }
    acc[static_cast<size_t>(m) * N + n] = finish(a, nb) + tail_dot(w, x, nb * kTernaryBlock, K);
  }
  (void)M;
}

#elif BM_NEON
// same idea on 128-bit registers: two steps of 16 weights (4 packed bytes) per 32-weight block
struct Consts {
  uint8x16_t shuf_lo, shuf_hi, sel, pos, neg;
};
inline Consts make_consts() {
  static const uint8_t lo[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  static const uint8_t hi[16] = {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
  static const uint8_t sel[16] = {0x03, 0x0C, 0x30, 0xC0, 0x03, 0x0C, 0x30, 0xC0, 0x03, 0x0C, 0x30, 0xC0, 0x03, 0x0C, 0x30, 0xC0};
  static const uint8_t pos[16] = {0x01, 0x04, 0x10, 0x40, 0x01, 0x04, 0x10, 0x40, 0x01, 0x04, 0x10, 0x40, 0x01, 0x04, 0x10, 0x40};
  static const uint8_t neg[16] = {0x02, 0x08, 0x20, 0x80, 0x02, 0x08, 0x20, 0x80, 0x02, 0x08, 0x20, 0x80, 0x02, 0x08, 0x20, 0x80};
  return {vld1q_u8(lo), vld1q_u8(hi), vld1q_u8(sel), vld1q_u8(pos), vld1q_u8(neg)};
}

// acc += (x & pos) - (x & neg), widened with pairwise adds only
inline int32x4_t step16(int32x4_t acc, const uint8x16_t packed8, const uint8x16_t shuf, const int8_t* x, const Consts& c) {
  const uint8x16_t m = vandq_u8(vqtbl1q_u8(packed8, shuf), c.sel);
  const int8x16_t pos = vreinterpretq_s8_u8(vceqq_u8(m, c.pos));
  const int8x16_t neg = vreinterpretq_s8_u8(vceqq_u8(m, c.neg));
  const int8x16_t xv = vld1q_s8(x);
  const int8x16_t d = vsubq_s8(vandq_s8(xv, pos), vandq_s8(xv, neg));
  return vpadalq_s16(acc, vpaddlq_s8(d));
}

inline int32_t dot_row(const uint8_t* w, const int8_t* x, int K) {
  const Consts c = make_consts();
  const int nb = K / kTernaryBlock;
  int32x4_t acc = vdupq_n_s32(0);
  for (int b = 0; b < nb; ++b) {
    const uint8x16_t packed = vcombine_u8(vld1_u8(w + b * 8), vdup_n_u8(0));
    acc = step16(acc, packed, c.shuf_lo, x + b * kTernaryBlock, c);
    acc = step16(acc, packed, c.shuf_hi, x + b * kTernaryBlock + 16, c);
  }
  return vaddvq_s32(acc) + tail_dot(w, x, nb * kTernaryBlock, K);
}

void gemv_rows(const uint8_t* Wp, int K, const int8_t* x, int32_t* y, int n0, int n1) {
  const int rb = ternary_row_bytes(K);
  for (int n = n0; n < n1; ++n) y[n] = dot_row(Wp + static_cast<size_t>(n) * rb, x, K);
}

void gemm_row(const uint8_t* w, int K, const int8_t* X, int M, int N, int n, int32_t* acc, int m0, int m1) {
  (void)M;
  for (int m = m0; m < m1; ++m) acc[static_cast<size_t>(m) * N + n] = dot_row(w, X + static_cast<size_t>(m) * K, K);
}

#else
void gemv_rows(const uint8_t* Wp, int K, const int8_t* x, int32_t* y, int n0, int n1) {
  const int rb = ternary_row_bytes(K);
  for (int n = n0; n < n1; ++n) y[n] = tail_dot(Wp + static_cast<size_t>(n) * rb, x, 0, K);
}
void gemm_row(const uint8_t* w, int K, const int8_t* X, int M, int N, int n, int32_t* acc, int m0, int m1) {
  (void)M;
  for (int m = m0; m < m1; ++m) acc[static_cast<size_t>(m) * N + n] = tail_dot(w, X + static_cast<size_t>(m) * K, 0, K);
}
#endif

}  // namespace

void ternary_gemv_i32(int N, int K, const uint8_t* Wp, const int8_t* xq, int32_t* acc) {
  const int blocks = (N + 3) / 4;
  BM_PRAGMA(omp parallel for schedule(static) if (static_cast<size_t>(N) * K >= (1u << 18)))  // same threshold as the FP32/INT8 GEMV kernels
  for (int b = 0; b < blocks; ++b) gemv_rows(Wp, K, xq, acc, b * 4, std::min(N, b * 4 + 4));
}

void ternary_gemm_i32(int M, int N, int K, const int8_t* Xq, const uint8_t* Wp, int32_t* acc) {
  if (M == 1) {
    ternary_gemv_i32(N, K, Wp, Xq, acc);
    return;
  }
  const int rb = ternary_row_bytes(K);
  constexpr int kTokTile = 64;  // tokens per task: the activations of one tile stay in L1/L2 while rows stream by
  const int tiles = (M + kTokTile - 1) / kTokTile;
  BM_PRAGMA(omp parallel for collapse(2) schedule(static) if (static_cast<size_t>(M) * N * K >= 1000000u))
  for (int t = 0; t < tiles; ++t)
    for (int n = 0; n < N; ++n)
      gemm_row(Wp + static_cast<size_t>(n) * rb, K, Xq, M, N, n, acc, t * kTokTile, std::min(M, (t + 1) * kTokTile));
}

void ternary_gemm_i32_ref(int M, int N, int K, const int8_t* Xq, const uint8_t* Wp, int32_t* acc) {
  const int rb = ternary_row_bytes(K);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) acc[static_cast<size_t>(m) * N + n] = tail_dot(Wp + static_cast<size_t>(n) * rb, Xq + static_cast<size_t>(m) * K, 0, K);
}

}  // namespace bm
