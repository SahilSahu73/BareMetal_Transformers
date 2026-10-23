#include <cstddef>

#include "kernels.hpp"

namespace bm {

void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];  // B walks a column: stride N
      C[i * N + j] = s;
    }
}

void gemv_f32_naive(int N, int K, const float* W, const float* x, float* y) {
  for (int n = 0; n < N; ++n) {
    const float* w = W + static_cast<size_t>(n) * K;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s += w[k] * x[k];
    y[n] = s;
  }
}

void gemv_q8_naive(int N, int K, const int8_t* Q, const float* scales, const float* x, float* y) {
  for (int n = 0; n < N; ++n) {
    const int8_t* q = Q + static_cast<size_t>(n) * K;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s += static_cast<float>(q[k]) * x[k];
    y[n] = s * scales[n];  // the scale is constant over the row, so apply it once at the end
  }
}

void gemm_nt_naive(int M, int N, int K, const float* X, const float* W, float* Y) {
  for (int m = 0; m < M; ++m) gemv_f32_naive(N, K, W, X + static_cast<size_t>(m) * K, Y + static_cast<size_t>(m) * N);
}

KernelTable naive_kernels() { return {gemv_f32_naive, gemv_q8_naive, gemm_nt_naive}; }

}  // namespace bm
