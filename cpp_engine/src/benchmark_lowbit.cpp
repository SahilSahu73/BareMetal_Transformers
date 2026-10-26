// Empirical kernel comparison for the low-precision extension: FP32 vs INT8 (W8A8) vs ternary
// (2-bit weights, multiplier-less) on the two shapes that matter.
//
//   decode  : y = W x          one token, a matrix-vector product, bound by weight bytes from DRAM
//   prefill : Y = X W^T        128 tokens, a matrix-matrix product, bound by arithmetic
//
// "cold" runs rotate through enough copies of the weights to defeat every cache, which is what a
// real model larger than the last-level cache looks like.
//   benchmark_lowbit [--threads T] [--reps R] [--csv FILE]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "kernels.hpp"
#include "quantize.hpp"
#include "tensor.hpp"
#include "ternary.hpp"

using namespace bm;
using clk = std::chrono::steady_clock;

namespace {

FILE* g_csv = nullptr;

std::vector<float> randn(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d;
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

// minimum over reps (a laptop CPU only ever gets slower than its best run), after one warm-up
double best_ms(const std::function<void(int)>& fn, int reps) {
  fn(0);
  double best = 1e30;
  for (int i = 1; i <= reps; ++i) {
    const auto a = clk::now();
    fn(i);
    best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - a).count());
  }
  return best;
}

struct Case {
  const char* name;
  double bytes_per_weight;
  std::function<void(int)> run;  // argument: which weight copy to read
};

}  // namespace

int main(int argc, char** argv) {
  int threads = physical_cores(), reps = 30;
  std::string csv;
  for (int i = 1; i < argc; ++i) {
    auto next = [&] { return i + 1 < argc ? argv[++i] : "0"; };
    if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(next());
    else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(next());
    else if (!std::strcmp(argv[i], "--csv")) csv = next();
    else { std::fprintf(stderr, "usage: benchmark_lowbit [--threads T] [--reps R] [--csv FILE]\n"); return 1; }
  }
  if (!csv.empty()) {
    g_csv = std::fopen(csv.c_str(), "w");
    if (g_csv) std::fprintf(g_csv, "phase,shape,kernel,threads,bytes_per_weight,ms,weight_gbs,gmacs\n");
  }
  set_num_threads(threads);
  std::printf("low-precision kernels | ISA %s | %d threads | best of %d runs\n", simd_isa(), threads, reps);

  struct Shape { int n, k; };
  for (const char* phase : {"decode", "prefill"}) {
    const bool decode = phase[0] == 'd';
    const int M = decode ? 1 : 128;
    std::printf("\n== %s (%s, M = %d tokens) ==\n", phase, decode ? "matrix-vector, weights streamed from DRAM" : "matrix-matrix", M);
    std::printf("%-12s %-30s %6s | %9s %10s %10s | %s\n", "N x K", "kernel", "B/wt", "time us", "weight GB/s", "GMAC/s", "check");
    for (auto s : {Shape{512, 512}, Shape{2048, 512}, Shape{512, 2048}, Shape{4096, 4096}}) {
      const int N = s.n, K = s.k;
      const size_t elems = size_t(N) * K;
      auto Wf = randn(elems, 3);
      auto X = randn(size_t(M) * K, 4);

      std::vector<int8_t> Wq8(elems), Wt(elems), Xq(size_t(M) * K);
      std::vector<float> ws(N), xs(M);
      quantize_matrix_q8(Wf.data(), N, K, Wq8.data(), ws.data());
      const float gamma = ternary_quantize_absmean(Wf.data(), elems, Wt.data());
      const int rb = ternary_row_bytes(K);
      std::vector<uint8_t> Wp(size_t(N) * rb);
      pack_ternary(Wt.data(), N, K, Wp.data());
      for (int m = 0; m < M; ++m) xs[m] = quantize_row_q8(X.data() + size_t(m) * K, K, Xq.data() + size_t(m) * K);

      // `cold` = enough copies of each weight format that consecutive calls never hit cache
      auto copies = [&](double bytes) { return int(std::max(2.0, std::ceil(192e6 / (elems * bytes)))); };
      const int c32 = copies(4), c8 = copies(1), c2 = copies(0.25);
      std::vector<float> big32(elems * c32);
      std::vector<int8_t> big8(elems * c8);
      std::vector<uint8_t> big2(Wp.size() * c2);
      for (int c = 0; c < c32; ++c) std::memcpy(big32.data() + elems * c, Wf.data(), elems * 4);
      for (int c = 0; c < c8; ++c) std::memcpy(big8.data() + elems * c, Wq8.data(), elems);
      for (int c = 0; c < c2; ++c) std::memcpy(big2.data() + Wp.size() * c, Wp.data(), Wp.size());

      std::vector<float> Y(size_t(M) * N), ref(size_t(M) * N);
      std::vector<int32_t> acc(size_t(M) * N);
      const float sx0 = xs[0];
      auto rescale = [&] {  // the single multiply per output that follows the integer accumulation
        for (int m = 0; m < M; ++m)
          for (int n = 0; n < N; ++n) Y[size_t(m) * N + n] = float(acc[size_t(m) * N + n]) * (gamma * xs[m]);
      };

      std::vector<Case> cases;
      if (decode) {
        cases = {
            {"fp32 SIMD", 4, [&](int i) { gemv_f32_simd(N, K, big32.data() + elems * (i % c32), X.data(), Y.data()); }},
            {"int8 W8A8 (madd)", 1, [&](int i) { gemv_q8a8_simd(N, K, big8.data() + elems * (i % c8), ws.data(), Xq.data(), sx0, Y.data()); }},
            {"ternary 2-bit (add/sub only)", 0.25, [&](int i) { ternary_gemv_i32(N, K, big2.data() + Wp.size() * (i % c2), Xq.data(), acc.data()); rescale(); }},
        };
      } else {
        cases = {
            {"fp32 SIMD", 4, [&](int i) { gemm_nt_simd(M, N, K, X.data(), big32.data() + elems * (i % c32), Y.data()); }},
            {"int8 W8A8 (madd)", 1, [&](int i) { gemm_nt_q8a8_simd(M, N, K, Xq.data(), xs.data(), big8.data() + elems * (i % c8), ws.data(), Y.data()); }},
            {"ternary 2-bit (add/sub only)", 0.25, [&](int i) { ternary_gemm_i32(M, N, K, Xq.data(), big2.data() + Wp.size() * (i % c2), acc.data()); rescale(); }},
        };
      }

      // exact check of the ternary kernel against a plain multiply-accumulate on the same integers
      std::vector<int32_t> exact(size_t(M) * N);
      ternary_gemm_i32_ref(M, N, K, Xq.data(), Wp.data(), exact.data());
      ternary_gemm_i32(M, N, K, Xq.data(), Wp.data(), acc.data());
      const bool ok = exact == acc;

      char shape[32];
      std::snprintf(shape, sizeof shape, "%dx%d", N, K);
      double fp32_ms = 0;
      for (size_t c = 0; c < cases.size(); ++c) {
        const int r = decode ? reps : std::max(5, reps / 3);
        const double ms = best_ms(cases[c].run, r);
        if (c == 0) fp32_ms = ms;
        const double gbs = elems * cases[c].bytes_per_weight / ms / 1e6;
        const double gmacs = double(M) * N * K / ms / 1e6;
        std::printf("%-12s %-30s %6.2f | %9.1f %10.1f %10.1f | %s%s\n", shape, cases[c].name, cases[c].bytes_per_weight, ms * 1000, gbs, gmacs,
                    c == 0 ? "" : (std::to_string(fp32_ms / ms).substr(0, 4) + "x vs fp32").c_str(), (c == 2 && !ok) ? "  MISMATCH" : "");
        if (g_csv) std::fprintf(g_csv, "%s,%s,%s,%d,%.2f,%.5f,%.3f,%.3f\n", phase, shape, cases[c].name, threads, cases[c].bytes_per_weight, ms, gbs, gmacs);
      }
    }
  }
  if (g_csv) std::fclose(g_csv);
  return 0;
}
