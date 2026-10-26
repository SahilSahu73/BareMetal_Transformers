#include "lowbit_train.hpp"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#define BM_PRAGMA(x) _Pragma(#x)
#else
#define BM_PRAGMA(x)
#endif

namespace bm {
namespace {

inline uint64_t splitmix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

inline float round_value(float x, Rounding mode, uint64_t seed, uint64_t index) {
  return mode == Rounding::Stochastic ? stochastic_round(x, uniform01(seed, index)) : std::nearbyint(x);
}

}  // namespace

float uniform01(uint64_t seed, uint64_t index) {
  const uint64_t h = splitmix64(seed ^ splitmix64(index));
  return static_cast<float>(h >> 40) * (1.0f / 16777216.0f);  // top 24 bits -> [0, 1)
}

float stochastic_round(float x, float u) { return std::floor(x + u); }

void stochastic_round_array(const float* x, size_t n, uint64_t seed, float* out) {
  BM_PRAGMA(omp parallel for schedule(static) if (n >= 65536))
  for (long long i = 0; i < static_cast<long long>(n); ++i) out[i] = stochastic_round(x[i], uniform01(seed, static_cast<uint64_t>(i)));
}

void quantize_to_grid_i8(const float* x, size_t n, float scale, int bits, Rounding mode, uint64_t seed, int8_t* q) {
  const float qmax = static_cast<float>((1 << (bits - 1)) - 1);
  const float inv = 1.0f / scale;
  BM_PRAGMA(omp parallel for schedule(static) if (n >= 65536))
  for (long long i = 0; i < static_cast<long long>(n); ++i) {
    const float r = round_value(x[i] * inv, mode, seed, static_cast<uint64_t>(i));
    q[i] = static_cast<int8_t>(std::min(qmax, std::max(-qmax, r)));
  }
}

void bfp_quantize(const float* x, size_t n, int block, int bits, Rounding mode, uint64_t seed, int8_t* mant, int8_t* exps) {
  const float qmax = static_cast<float>((1 << (bits - 1)) - 1);
  const size_t nblocks = (n + block - 1) / block;
  BM_PRAGMA(omp parallel for schedule(static) if (n >= 65536))
  for (long long b = 0; b < static_cast<long long>(nblocks); ++b) {
    const size_t lo = static_cast<size_t>(b) * block, hi = std::min(n, lo + block);
    float amax = 0.f;
    for (size_t i = lo; i < hi; ++i) amax = std::max(amax, std::fabs(x[i]));
    int e = 0;
    if (amax > 0.f) {
      int p;
      const float m = std::frexp(amax / qmax, &p);  // amax / qmax = m * 2^p, m in [0.5, 1)
      e = (m == 0.5f) ? p - 1 : p;                  // ceil(log2(amax / qmax)), exact at powers of two
      e = std::min(127, std::max(-127, e));
    }
    exps[b] = static_cast<int8_t>(e);
    const float inv = std::ldexp(1.0f, -e);
    for (size_t i = lo; i < hi; ++i) {
      const float r = round_value(x[i] * inv, mode, seed, i);
      mant[i] = static_cast<int8_t>(std::min(qmax, std::max(-qmax, r)));
    }
  }
}

void bfp_dequantize(const int8_t* mant, const int8_t* exps, size_t n, int block, float* out) {
  for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(mant[i]) * std::ldexp(1.0f, exps[i / block]);
}

}  // namespace bm
