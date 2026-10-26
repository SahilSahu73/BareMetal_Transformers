// Ternary ({-1, 0, +1}) weights, 2 bits each, and a multiplier-less GEMM over INT8 activations.
//
// Format. A weight matrix W[rows, K] (rows = output channels, K = input features) is stored one
// row after another. Four weights share a byte, first weight in the low bits:
//
//     byte = w0 | w1 << 2 | w2 << 4 | w3 << 6        code: 00 -> 0,  01 -> +1,  10 -> -1,  11 -> reserved (reads as 0)
//
// Every row is padded with zero codes to a multiple of 32 weights (8 bytes), so a row starts on an
// 8-byte boundary and a SIMD step consumes exactly 8 bytes. That is 16x smaller than FP32.
//
// Arithmetic. With w in {-1, 0, +1} a multiply-accumulate degenerates into "add x, subtract x, or
// skip": acc += x, acc -= x. The kernels below never multiply; they select the activation with a
// mask and add/subtract it into an integer accumulator, the way a ternary processing element
// (an adder instead of a DSP multiplier) would. The single multiply per output happens afterwards,
// when the integer accumulator is rescaled by gamma * activation_scale.
#pragma once
#include <cstddef>
#include <cstdint>

namespace bm {

constexpr int kTernaryBlock = 32;  // weights per SIMD step and row padding granularity

inline int ternary_row_bytes(int K) { return (K + kTernaryBlock - 1) / kTernaryBlock * (kTernaryBlock / 4); }

// ---- quantisation and packing ---------------------------------------------------------------
// BitNet b1.58 "absmean": gamma = mean(|W|), Wq = clip(round(W / gamma), -1, 1). One gamma per
// tensor. Returns gamma (0 for an all-zero tensor, in which case Wq is all zero).
float ternary_quantize_absmean(const float* W, size_t n, int8_t* Wq);

void pack_ternary(const int8_t* Wq, int rows, int K, uint8_t* packed);  // Wq in {-1,0,+1}, row-major
void unpack_ternary(const uint8_t* packed, int rows, int K, int8_t* Wq);

// ---- multiplier-less integer kernels ---------------------------------------------------------
// acc[m, n] = sum_k Xq[m, k] * W[n, k]  with W ternary and packed. Xq is INT8 in [-127, 127]
// (-128 is outside the contract: negating it would overflow). Exact integer result.
void ternary_gemv_i32(int N, int K, const uint8_t* Wp, const int8_t* xq, int32_t* acc);
void ternary_gemm_i32(int M, int N, int K, const int8_t* Xq, const uint8_t* Wp, int32_t* acc);
// Straight-line scalar reference of the same computation (also add/subtract only).
void ternary_gemm_i32_ref(int M, int N, int K, const int8_t* Xq, const uint8_t* Wp, int32_t* acc);

// ---- floating-point wrappers: one rescale multiply per OUTPUT, none per weight --------------
// Y[m, n] = acc[m, n] * gamma * xscale[m]
void ternary_gemv_f32(int N, int K, const uint8_t* Wp, float gamma, const int8_t* xq, float xscale, float* y);
void ternary_gemm_f32(int M, int N, int K, const int8_t* Xq, const float* xscales, const uint8_t* Wp, float gamma, float* Y);

}  // namespace bm
