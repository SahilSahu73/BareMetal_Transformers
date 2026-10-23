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

// The model calls matrix-vector products through this table, so a benchmark can swap
// kernel implementations without touching the transformer.
struct KernelTable {
  GemvF32 gemv_f32;
  GemvQ8 gemv_q8;
  GemmNtF32 gemm_nt_f32;
};
KernelTable naive_kernels();
KernelTable optimized_kernels();  // SIMD FP32 kernels (INT8 stays scalar until the INT8 kernels land)

// ---- element-wise / row-wise ops ---------------------------------------------------------
// y = gamma * x / sqrt(mean(x^2) + eps)
void rmsnorm(const float* x, const float* gamma, float* y, int n, float eps);
void softmax_inplace(float* x, int n);
void gelu_inplace(float* x, int n);  // tanh approximation, same as the training code

// Rotary embedding of one head vector (interleaved pairs) at one position.
// cos/sin point at that position's row of the precomputed table (head_dim/2 entries).
void rope_inplace(float* head, const float* cos, const float* sin, int head_dim);

}  // namespace bm
