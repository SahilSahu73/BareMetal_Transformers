#include <cmath>

#include "kernels.hpp"

namespace bm {

void rmsnorm(const float* x, const float* gamma, float* y, int n, float eps) {
  float ss = 0.f;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) y[i] = gamma[i] * (x[i] * inv);
}

void softmax_inplace(float* x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; ++i) mx = x[i] > mx ? x[i] : mx;  // subtract the max: exp never overflows
  float sum = 0.f;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  const float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

void gelu_inplace(float* x, int n) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kC = 0.044715f;
  for (int i = 0; i < n; ++i) {
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + std::tanh(kSqrt2OverPi * (v + kC * v * v * v)));
  }
}

void rope_inplace(float* head, const float* cos, const float* sin, int head_dim) {
  for (int i = 0; i < head_dim / 2; ++i) {
    const float a = head[2 * i], b = head[2 * i + 1];
    head[2 * i] = a * cos[i] - b * sin[i];
    head[2 * i + 1] = a * sin[i] + b * cos[i];
  }
}

}  // namespace bm
