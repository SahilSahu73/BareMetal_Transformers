#include "ternary.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#define BM_PRAGMA(x) _Pragma(#x)
#else
#define BM_PRAGMA(x)
#endif

namespace bm {

float ternary_quantize_absmean(const float* W, size_t n, int8_t* Wq) {
  double sum = 0.0;  // double accumulator: n can be millions of elements
  for (size_t i = 0; i < n; ++i) sum += std::fabs(W[i]);
  const float gamma = n ? static_cast<float>(sum / static_cast<double>(n)) : 0.f;
  if (gamma == 0.f) {
    std::memset(Wq, 0, n);
    return 0.f;
  }
  const float inv = 1.0f / gamma;
  for (size_t i = 0; i < n; ++i) {
    const float r = std::nearbyint(W[i] * inv);  // round half to even, like numpy.rint
    Wq[i] = static_cast<int8_t>(std::min(1.f, std::max(-1.f, r)));
  }
  return gamma;
}

void pack_ternary(const int8_t* Wq, int rows, int K, uint8_t* packed) {
  const int rb = ternary_row_bytes(K);
  for (int r = 0; r < rows; ++r) {
    const int8_t* src = Wq + static_cast<size_t>(r) * K;
    uint8_t* dst = packed + static_cast<size_t>(r) * rb;
    std::memset(dst, 0, rb);  // padding weights are code 00 (= 0)
    for (int k = 0; k < K; ++k) {
      const unsigned code = src[k] > 0 ? 1u : (src[k] < 0 ? 2u : 0u);
      dst[k >> 2] = static_cast<uint8_t>(dst[k >> 2] | (code << ((k & 3) * 2)));
    }
  }
}

void unpack_ternary(const uint8_t* packed, int rows, int K, int8_t* Wq) {
  const int rb = ternary_row_bytes(K);
  for (int r = 0; r < rows; ++r) {
    const uint8_t* src = packed + static_cast<size_t>(r) * rb;
    int8_t* dst = Wq + static_cast<size_t>(r) * K;
    for (int k = 0; k < K; ++k) {
      const unsigned code = (src[k >> 2] >> ((k & 3) * 2)) & 3u;
      dst[k] = code == 1u ? 1 : (code == 2u ? -1 : 0);
    }
  }
}

void ternary_gemv_f32(int N, int K, const uint8_t* Wp, float gamma, const int8_t* xq, float xscale, float* y) {
  std::vector<int32_t> acc(N);
  ternary_gemv_i32(N, K, Wp, xq, acc.data());
  const float s = gamma * xscale;
  for (int n = 0; n < N; ++n) y[n] = static_cast<float>(acc[n]) * s;
}

void ternary_gemm_f32(int M, int N, int K, const int8_t* Xq, const float* xscales, const uint8_t* Wp, float gamma, float* Y) {
  std::vector<int32_t> acc(static_cast<size_t>(M) * N);
  ternary_gemm_i32(M, N, K, Xq, Wp, acc.data());
  BM_PRAGMA(omp parallel for schedule(static) if (static_cast<size_t>(M) * N >= 65536))
  for (int m = 0; m < M; ++m) {
    const float s = gamma * xscales[m];
    for (int n = 0; n < N; ++n) Y[static_cast<size_t>(m) * N + n] = static_cast<float>(acc[static_cast<size_t>(m) * N + n]) * s;
  }
}

}  // namespace bm
