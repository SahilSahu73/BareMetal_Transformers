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

void gemm_nt_q8_naive(int M, int N, int K, const float* X, const int8_t* Q, const float* wscales, float* Y) {
  for (int m = 0; m < M; ++m)
    gemv_q8_naive(N, K, Q, wscales, X + static_cast<size_t>(m) * K, Y + static_cast<size_t>(m) * N);
}

void gemv_q8a8_naive(int N, int K, const int8_t* Q, const float* wscales, const int8_t* xq, float xscale, float* y) {
  for (int n = 0; n < N; ++n) {
    const int8_t* q = Q + static_cast<size_t>(n) * K;
    int32_t acc = 0;  // exact: |sum| <= K * 127 * 127 fits int32 for K < 133,000
    for (int k = 0; k < K; ++k) acc += static_cast<int32_t>(q[k]) * xq[k];
    y[n] = static_cast<float>(acc) * wscales[n] * xscale;
  }
}

void gemm_nt_q8a8_naive(int M, int N, int K, const int8_t* Xq, const float* xscales, const int8_t* Q,
                        const float* wscales, float* Y) {
  for (int m = 0; m < M; ++m)
    gemv_q8a8_naive(N, K, Q, wscales, Xq + static_cast<size_t>(m) * K, xscales[m], Y + static_cast<size_t>(m) * N);
}

float dot_f32_naive(const float* a, const float* b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
void axpy_f32_naive(float* y, float alpha, const float* x, int n) {
  for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}
float dot_f32_i8_naive(const float* a, const int8_t* b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; ++i) s += a[i] * static_cast<float>(b[i]);
  return s;
}
void axpy_f32_i8_naive(float* y, float alpha, const int8_t* x, int n) {
  for (int i = 0; i < n; ++i) y[i] += alpha * static_cast<float>(x[i]);
}

KernelTable naive_kernels() {
  return {gemv_f32_naive,   gemv_q8_naive,      gemm_nt_naive,       gemm_nt_q8_naive,    gemv_q8a8_naive,
          gemm_nt_q8a8_naive, dot_f32_naive,    axpy_f32_naive,      dot_f32_i8_naive,    axpy_f32_i8_naive};
}

}  // namespace bm
