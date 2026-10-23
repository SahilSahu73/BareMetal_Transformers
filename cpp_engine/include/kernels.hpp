// Compute kernels. Kernel 0 (naive) lives here; later phases add tiled / SIMD / INT8 variants
// behind the same signatures so the model code never changes.
#pragma once
#include <cstddef>
#include <cstddef>
#include <cstdint>

namespace bm {

// ---- matrix products (row-major, contiguous) --------------------------------------------
// C[M,N] = A[M,K] * B[K,N]
// Kernel 0: textbook i-j-k triple loop. B is walked down a column, one cache line per element.
void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C);
// Kernel 1a: i-k-j loop order. The inner loop streams contiguous rows of B and C.
void gemm_reordered(int M, int N, int K, const float* A, const float* B, float* C);
// Kernel 1b: i-k-j plus block x block x block tiling so a tile of B is reused from L1/L2.
void gemm_tiled(int M, int N, int K, const float* A, const float* B, float* C, int block = 64);

// Kernel 2: packed, register-blocked micro-kernel (AVX2+FMA 6x16 / NEON 8x12) + OpenMP.
// Cache blocking parameters of the GotoBLAS/BLIS algorithm; mc is rounded to the micro-tile
// height and nc to its width.
struct GemmBlocking {
  int mc = 72;    // rows of A packed per block (sized for L2)
  int kc = 256;   // shared dimension per block (a B micro-panel kc x NR fits L1)
  int nc = 4080;  // columns of B packed per block (sized for L3)
};
// With b_transposed, B is stored [N,K] (a weight matrix W[out,in]) and C = A * B^T, which lets
// the model multiply by its weights without a transpose copy. lda/ldb/ldc are row strides.
void gemm_simd_ld(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                  bool b_transposed, float* C, int ldc, const GemmBlocking& blk = {});
void gemm_simd(int M, int N, int K, const float* A, const float* B, float* C,
               bool b_transposed = false, const GemmBlocking& blk = {});

// Y[M,N] = X[M,K] * W[N,K]^T: a batch of M tokens through one weight matrix (prefill).
using GemmNtF32 = void (*)(int M, int N, int K, const float* X, const float* W, float* Y);
void gemm_nt_naive(int M, int N, int K, const float* X, const float* W, float* Y);
void gemm_nt_simd(int M, int N, int K, const float* X, const float* W, float* Y);

// Name of the instruction set the SIMD kernels were compiled for ("AVX2+FMA", "NEON", "scalar").
const char* simd_isa();
int micro_tile_m();
int micro_tile_n();
// Thread count used by the OpenMP kernels (no-op without OpenMP).
void set_num_threads(int n);
int num_threads();
// Physical core count (SMT siblings counted once; falls back to hardware threads). FMA-bound kernels
// gain nothing from the second hardware thread of a core, so this is the sensible default.
int physical_cores();

// y[N] = W[N,K] * x[K]. W is stored [out, in], so each output is a contiguous dot product.
using GemvF32 = void (*)(int N, int K, const float* W, const float* x, float* y);
void gemv_f32_naive(int N, int K, const float* W, const float* x, float* y);
// Four rows at a time share each load of x; independent accumulators hide FMA latency; large
// matrices are split across threads. Decoding is memory bound, so this is about streaming W.
void gemv_f32_simd(int N, int K, const float* W, const float* x, float* y);
// Matrices with fewer weights than this run single-threaded (tunable; see benchmark --gemv).
size_t& gemv_parallel_threshold();

// INT8 weights with one FP32 scale per output row, FP32 activations:
//   y[n] = scales[n] * sum_k Q[n,k] * x[k]
using GemvQ8 = void (*)(int N, int K, const int8_t* Q, const float* scales, const float* x, float* y);
void gemv_q8_naive(int N, int K, const int8_t* Q, const float* scales, const float* x, float* y);

// ---- INT8 ------------------------------------------------------------------------------------
// W8A32 batch: Y[M,N] = X[M,K] * dequant(Q[N,K])^T. The SIMD version dequantises while packing B
// into the FP32 micro-kernel's panels, so it runs at FP32 GEMM speed with 4x less weight traffic.
using GemmNtQ8 = void (*)(int M, int N, int K, const float* X, const int8_t* Q, const float* wscales, float* Y);
void gemm_nt_q8_naive(int M, int N, int K, const float* X, const int8_t* Q, const float* wscales, float* Y);
void gemm_nt_q8_simd(int M, int N, int K, const float* X, const int8_t* Q, const float* wscales, float* Y);
void gemv_q8_simd(int N, int K, const int8_t* Q, const float* scales, const float* x, float* y);

// W8A8: activations are quantised too (one scale per vector / token row), so the inner loop is a
// pure integer dot product accumulated in int32 and rescaled once per output:
//   y[n] = wscales[n] * xscale * sum_k Q[n,k] * xq[k]
using GemvQ8A8 = void (*)(int N, int K, const int8_t* Q, const float* wscales, const int8_t* xq, float xscale, float* y);
using GemmNtQ8A8 = void (*)(int M, int N, int K, const int8_t* Xq, const float* xscales, const int8_t* Q,
                            const float* wscales, float* Y);
void gemv_q8a8_naive(int N, int K, const int8_t* Q, const float* wscales, const int8_t* xq, float xscale, float* y);
void gemv_q8a8_simd(int N, int K, const int8_t* Q, const float* wscales, const int8_t* xq, float xscale, float* y);
void gemm_nt_q8a8_naive(int M, int N, int K, const int8_t* Xq, const float* xscales, const int8_t* Q,
                        const float* wscales, float* Y);
void gemm_nt_q8a8_simd(int M, int N, int K, const int8_t* Xq, const float* xscales, const int8_t* Q,
                       const float* wscales, float* Y);

// Attention inner loops over one head (n = head_dim): scores = q . k and out += weight * v,
// for FP32 and for INT8 cached vectors (the caller applies the per-vector scale).
using DotF32 = float (*)(const float* a, const float* b, int n);
using AxpyF32 = void (*)(float* y, float alpha, const float* x, int n);  // y += alpha * x
using DotF32I8 = float (*)(const float* a, const int8_t* b, int n);
using AxpyF32I8 = void (*)(float* y, float alpha, const int8_t* x, int n);
float dot_f32_naive(const float* a, const float* b, int n);
void axpy_f32_naive(float* y, float alpha, const float* x, int n);
float dot_f32_i8_naive(const float* a, const int8_t* b, int n);
void axpy_f32_i8_naive(float* y, float alpha, const int8_t* x, int n);
float dot_f32(const float* a, const float* b, int n);
void axpy_f32(float* y, float alpha, const float* x, int n);
float dot_f32_i8(const float* a, const int8_t* b, int n);
void axpy_f32_i8(float* y, float alpha, const int8_t* x, int n);

// The model calls matrix-vector products through this table, so a benchmark can swap
// kernel implementations without touching the transformer.
struct KernelTable {
  GemvF32 gemv_f32;
  GemvQ8 gemv_q8;
  GemmNtF32 gemm_nt_f32;
  GemmNtQ8 gemm_nt_q8;
  GemvQ8A8 gemv_q8a8;
  GemmNtQ8A8 gemm_nt_q8a8;
  DotF32 dot_f32;
  AxpyF32 axpy_f32;
  DotF32I8 dot_f32_i8;
  AxpyF32I8 axpy_f32_i8;
};
KernelTable naive_kernels();
KernelTable optimized_kernels();  // SIMD kernels for every precision

// ---- element-wise / row-wise ops ---------------------------------------------------------
// y = gamma * x / sqrt(mean(x^2) + eps)
void rmsnorm(const float* x, const float* gamma, float* y, int n, float eps);
void softmax_inplace(float* x, int n);
void gelu_inplace(float* x, int n);  // tanh approximation, same as the training code

// Rotary embedding of one head vector (interleaved pairs) at one position.
// cos/sin point at that position's row of the precomputed table (head_dim/2 entries).
void rope_inplace(float* head, const float* cos, const float* sin, int head_dim);

}  // namespace bm
