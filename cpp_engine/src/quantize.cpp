#include "quantize.hpp"

#include <algorithm>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace bm {

float quantize_row_q8(const float* x, int n, int8_t* q) {
  float amax = 0.f;
  int i = 0;
#if defined(__AVX2__)
  const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  __m256 vmax = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), abs_mask));
  __m128 m = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
  amax = _mm_cvtss_f32(m);
#elif defined(__ARM_NEON)
  float32x4_t vmax = vdupq_n_f32(0.f);
  for (; i + 4 <= n; i += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i)));
  amax = vmaxvq_f32(vmax);
#endif
  for (int j = i; j < n; ++j) amax = std::max(amax, std::fabs(x[j]));

  const float scale = amax > 0.f ? amax / 127.f : 1.f;
  i = 0;
#if defined(__AVX2__)
  const __m256 vs = _mm256_set1_ps(scale);
  for (; i + 8 <= n; i += 8) {
    // cvtps_epi32 rounds to nearest-even (the default MXCSR mode); packs saturate to the int8 range
    const __m256i r = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_loadu_ps(x + i), vs));
    const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(q + i), _mm_packs_epi16(p16, p16));
  }
#elif defined(__ARM_NEON)
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 8 <= n; i += 8) {
    // vcvtn rounds to nearest, ties to even; the narrowing moves saturate to the int8 range
    const int32x4_t lo = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + i), vs));
    const int32x4_t hi = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + i + 4), vs));
    vst1_s8(q + i, vqmovn_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))));
  }
#endif
  for (; i < n; ++i) q[i] = static_cast<int8_t>(std::clamp(static_cast<int>(std::lrintf(x[i] / scale)), -127, 127));
  return scale;
}

void dequantize_row_q8(const int8_t* q, float scale, int n, float* x) {
  for (int i = 0; i < n; ++i) x[i] = static_cast<float>(q[i]) * scale;
}

void quantize_matrix_q8(const float* W, int rows, int cols, int8_t* Q, float* scales) {
  for (int r = 0; r < rows; ++r)
    scales[r] = quantize_row_q8(W + static_cast<size_t>(r) * cols, cols, Q + static_cast<size_t>(r) * cols);
}

}  // namespace bm
