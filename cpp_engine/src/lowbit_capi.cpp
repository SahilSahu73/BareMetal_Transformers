// Plain C interface over the low-precision kernels, built as libbmlowbit.so and driven from Python
// with ctypes (python_scratch/lowbit_native.py). No Python headers or binding generator needed.
#include <cstdint>

#include "lowbit_train.hpp"
#include "quantize.hpp"
#include "ternary.hpp"

#if defined(_WIN32)
#define BM_EXPORT extern "C" __declspec(dllexport)
#else
#define BM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

BM_EXPORT int bm_ternary_row_bytes(int K) { return bm::ternary_row_bytes(K); }
BM_EXPORT float bm_ternary_quantize_absmean(const float* W, size_t n, int8_t* Wq) { return bm::ternary_quantize_absmean(W, n, Wq); }
BM_EXPORT void bm_pack_ternary(const int8_t* Wq, int rows, int K, uint8_t* packed) { bm::pack_ternary(Wq, rows, K, packed); }
BM_EXPORT void bm_unpack_ternary(const uint8_t* packed, int rows, int K, int8_t* Wq) { bm::unpack_ternary(packed, rows, K, Wq); }
BM_EXPORT void bm_ternary_gemm_i32(int M, int N, int K, const int8_t* Xq, const uint8_t* Wp, int32_t* acc) { bm::ternary_gemm_i32(M, N, K, Xq, Wp, acc); }
BM_EXPORT void bm_ternary_gemm_f32(int M, int N, int K, const int8_t* Xq, const float* xs, const uint8_t* Wp, float gamma, float* Y) {
  bm::ternary_gemm_f32(M, N, K, Xq, xs, Wp, gamma, Y);
}

BM_EXPORT float bm_quantize_row_q8(const float* x, int n, int8_t* q) { return bm::quantize_row_q8(x, n, q); }

BM_EXPORT void bm_stochastic_round(const float* x, size_t n, uint64_t seed, float* out) { bm::stochastic_round_array(x, n, seed, out); }
BM_EXPORT void bm_quantize_to_grid_i8(const float* x, size_t n, float scale, int bits, int stochastic, uint64_t seed, int8_t* q) {
  bm::quantize_to_grid_i8(x, n, scale, bits, stochastic ? bm::Rounding::Stochastic : bm::Rounding::Nearest, seed, q);
}
BM_EXPORT void bm_bfp_quantize(const float* x, size_t n, int block, int bits, int stochastic, uint64_t seed, int8_t* mant, int8_t* exps) {
  bm::bfp_quantize(x, n, block, bits, stochastic ? bm::Rounding::Stochastic : bm::Rounding::Nearest, seed, mant, exps);
}
BM_EXPORT void bm_bfp_dequantize(const int8_t* mant, const int8_t* exps, size_t n, int block, float* out) { bm::bfp_dequantize(mant, exps, n, block, out); }
BM_EXPORT const char* bm_simd_isa() {
#if defined(__AVX2__)
  return "AVX2";
#elif defined(__ARM_NEON)
  return "NEON";
#else
  return "scalar";
#endif
}
