// Symmetric INT8 quantisation: scale = max|x| / 127, q = round(x / scale) clamped to [-127, 127].
// "Symmetric" means there is no zero point, so an integer dot product needs no correction terms.
// Rounding is round-half-to-even, matching numpy.rint / torch.round in the Python tooling.
#pragma once
#include <cstdint>

namespace bm {

// Quantises one vector with a single scale; returns the scale (1.0 for an all-zero vector).
float quantize_row_q8(const float* x, int n, int8_t* q);
void dequantize_row_q8(const int8_t* q, float scale, int n, float* x);

// Per-output-channel matrix quantisation: one scale per row (what the exporter does for weights).
void quantize_matrix_q8(const float* W, int rows, int cols, int8_t* Q, float* scales);

}  // namespace bm
