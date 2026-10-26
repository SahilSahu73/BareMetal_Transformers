// Primitives for low-precision training: unbiased stochastic rounding, block-floating-point
// (shared power-of-two exponent per block) INT8/INT4 gradient quantisation, and rounding of
// weights onto a coarse INT8 grid.
//
// Stochastic rounding of x:   floor(x + u),  u ~ U[0, 1)
//   P(ceil(x)) = x - floor(x)   =>   E[SR(x)] = x      (round-to-nearest is biased for every
//   update smaller than half an LSB: it always rounds to zero, so the update never accumulates)
//
// The uniform u for element i comes from a counter-based hash of (seed, i), not from a
// sequential generator, so the result is identical for any thread count and any call order.
#pragma once
#include <cstddef>
#include <cstdint>

namespace bm {

enum class Rounding { Nearest, Stochastic };

// u in [0, 1) with 24 random bits for element `index` of the stream `seed`.
float uniform01(uint64_t seed, uint64_t index);

// SR of one value / an array of floats to integers (stored as float, exactly representable).
float stochastic_round(float x, float u);
void stochastic_round_array(const float* x, size_t n, uint64_t seed, float* out);

// INT8 grid: q = clamp(round(x / scale), -qmax, qmax) with qmax = 2^(bits-1) - 1, bits <= 8.
void quantize_to_grid_i8(const float* x, size_t n, float scale, int bits, Rounding mode, uint64_t seed, int8_t* q);

// Block floating point: blocks of `block` consecutive values share one exponent e (scale 2^e),
// chosen as the smallest e for which the block's largest magnitude fits in `bits`-bit mantissas.
//   mant[i] = clamp(round(x[i] / 2^e), -qmax, qmax),   value = mant * 2^e
// exps has ceil(n / block) entries. An all-zero block gets e = 0.
void bfp_quantize(const float* x, size_t n, int block, int bits, Rounding mode, uint64_t seed, int8_t* mant, int8_t* exps);
void bfp_dequantize(const int8_t* mant, const int8_t* exps, size_t n, int block, float* out);

}  // namespace bm
