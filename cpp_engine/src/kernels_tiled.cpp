// Kernel 1: fixing the memory access pattern of the triple loop, in two steps.
#include <algorithm>
#include <cstddef>
#include <cstring>

#include "kernels.hpp"

namespace bm {

void gemm_reordered(int M, int N, int K, const float* A, const float* B, float* C) {
  std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      const float a = A[static_cast<size_t>(i) * K + k];
      const float* b = B + static_cast<size_t>(k) * N;
      float* c = C + static_cast<size_t>(i) * N;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];  // contiguous in b and c
    }
}

void gemm_tiled(int M, int N, int K, const float* A, const float* B, float* C, int block) {
  std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
  const int bs = std::max(block, 1);
  // A bs x bs tile of B is reused by every row of the A tile before it is evicted, instead of
  // streaming the whole of B from L3/DRAM once per row of A.
  for (int i0 = 0; i0 < M; i0 += bs)
    for (int k0 = 0; k0 < K; k0 += bs)
      for (int j0 = 0; j0 < N; j0 += bs) {
        const int i1 = std::min(i0 + bs, M), k1 = std::min(k0 + bs, K), j1 = std::min(j0 + bs, N);
        for (int i = i0; i < i1; ++i)
          for (int k = k0; k < k1; ++k) {
            const float a = A[static_cast<size_t>(i) * K + k];
            const float* b = B + static_cast<size_t>(k) * N;
            float* c = C + static_cast<size_t>(i) * N;
            for (int j = j0; j < j1; ++j) c[j] += a * b[j];
          }
      }
}

}  // namespace bm
