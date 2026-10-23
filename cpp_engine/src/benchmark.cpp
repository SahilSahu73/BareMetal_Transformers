// Microbenchmark suite: ceilings first (what the CPU can do), then each kernel against them.
//   benchmark_transformer [--all] [--peak] [--gemm] [--gemv] [--scaling] [--sweep]
//                         [--size N] [--threads T] [--reps R] [--naive-max N] [--csv FILE]
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
#include "tensor.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bm;
using clk = std::chrono::steady_clock;

namespace {

struct Options {
  bool peak = false, gemm = false, gemv = false, scaling = false, sweep = false;
  int size = 1024, reps = 7;
  int naive_max = 1024;
  int threads = 0;  // 0 = physical cores
  std::string csv;
};

FILE* g_csv = nullptr;
void csv_row(const char* bench, const std::string& name, const std::string& params, int threads,
             double ms, double gflops, double gbs) {
  if (g_csv) std::fprintf(g_csv, "%s,%s,%s,%d,%.4f,%.3f,%.3f\n", bench, name.c_str(), params.c_str(), threads, ms, gflops, gbs);
}

std::vector<float> randn(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d;
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

// Best (minimum) wall time in ms over `reps` runs after one warm-up. Minimum rather than median:
// on a laptop CPU, other work and clock changes only ever make a run slower, never faster.
double time_ms(const std::function<void()>& fn, int reps) {
  fn();
  std::vector<double> t;
  double total = 0;
  for (int i = 0; i < reps && (i < 3 || total < 3000.0); ++i) {
    const auto a = clk::now();
    fn();
    const double ms = std::chrono::duration<double, std::milli>(clk::now() - a).count();
    t.push_back(ms);
    total += ms;
  }
  std::sort(t.begin(), t.end());
  return t.front();
}

// ---- hardware ceilings -----------------------------------------------------------------------
// Peak FP32 FMA throughput: independent accumulator chains kept in registers, no memory traffic.
double peak_gflops(int threads) {
  const long iters = 4000000;
  double best = 0;
  for (int rep = 0; rep < 3; ++rep) {
    volatile float sink = 0;
    const auto a = clk::now();
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#if defined(__AVX2__) && defined(__FMA__)
      __m256 acc[12];
      for (int i = 0; i < 12; ++i) acc[i] = _mm256_set1_ps(1.0f + i);
      const __m256 m = _mm256_set1_ps(1.0000001f), c = _mm256_set1_ps(1e-7f);
      for (long it = 0; it < iters; ++it)
#pragma GCC unroll 12
        for (int i = 0; i < 12; ++i) acc[i] = _mm256_fmadd_ps(acc[i], m, c);
      __m256 s = acc[0];
      for (int i = 1; i < 12; ++i) s = _mm256_add_ps(s, acc[i]);
      alignas(32) float out[8];
      _mm256_store_ps(out, s);
      sink = sink + out[0];
#elif defined(__ARM_NEON)
      float32x4_t acc[16];
      for (int i = 0; i < 16; ++i) acc[i] = vdupq_n_f32(1.0f + i);
      const float32x4_t m = vdupq_n_f32(1.0000001f), c = vdupq_n_f32(1e-7f);
      for (long it = 0; it < iters; ++it)
        for (int i = 0; i < 16; ++i) acc[i] = vfmaq_f32(c, acc[i], m);
      float32x4_t s = acc[0];
      for (int i = 1; i < 16; ++i) s = vaddq_f32(s, acc[i]);
      sink = sink + vgetq_lane_f32(s, 0);
#endif
    }
    const double sec = std::chrono::duration<double>(clk::now() - a).count();
#if defined(__AVX2__) && defined(__FMA__)
    const double flops = double(threads) * iters * 12 * 8 * 2;
#elif defined(__ARM_NEON)
    const double flops = double(threads) * iters * 16 * 4 * 2;
#else
    const double flops = 0;
#endif
    best = std::max(best, flops / sec / 1e9);
  }
  return best;
}

// Sustained read bandwidth from DRAM: each thread streams its own slice of a buffer far
// larger than the last-level cache.
double read_bandwidth_once(int threads) {
  static std::vector<float> buf(size_t(64) << 20, 1.0f);  // 256 MB
  const size_t n = buf.size();
  double best = 0;
  for (int rep = 0; rep < 4; ++rep) {
    volatile float sink = 0;
    const auto a = clk::now();
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
      const size_t t = omp_get_thread_num(), nt = omp_get_num_threads();
#else
      const size_t t = 0, nt = 1;
#endif
      const size_t lo = n / nt * t / 8 * 8, hi = (t + 1 == nt) ? n : n / nt * (t + 1) / 8 * 8;
      const float* p = buf.data();
      float total = 0;
#if defined(__AVX2__)
      __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
      size_t i = lo;
      for (; i + 32 <= hi; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + i));
        a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + i + 8));
        a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + i + 16));
        a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + i + 24));
      }
      alignas(32) float o[8];
      _mm256_store_ps(o, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
      total = o[0];
#else
      for (size_t i = lo; i < hi; ++i) total += p[i];
#endif
      sink = sink + total;
    }
    const double sec = std::chrono::duration<double>(clk::now() - a).count();
    best = std::max(best, double(n) * 4 / sec / 1e9);
  }
  return best;
}

double read_bandwidth_gbs(int max_threads) {
  double best = 0;
  for (int t : {1, 2, 4, 8, 16})
    if (t <= std::max(max_threads, 1) || t == 1) best = std::max(best, read_bandwidth_once(t));
  return best;
}

// Clocks on a laptop CPU swing by 2x with thermals and power limits, so "peak" is the highest
// FMA throughput observed at that thread count at any point during the run.
double g_peak[2] = {0, 0};  // [0] = 1 thread, [1] = N threads
void sample_peak(int nthreads) {
  g_peak[0] = std::max(g_peak[0], peak_gflops(1));
  g_peak[1] = std::max(g_peak[1], peak_gflops(nthreads));
}

// ---- GEMM ------------------------------------------------------------------------------------
double max_rel_err(const std::vector<float>& a, const std::vector<float>& ref) {
  double mx = 0, scale = 1e-9;
  for (size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, double(std::fabs(a[i] - ref[i])));
    scale = std::max(scale, double(std::fabs(ref[i])));
  }
  return mx / scale;
}

struct GemmCase {
  std::string name;
  int threads;
  std::function<void(int, int, int, const float*, const float*, float*)> fn;
  bool slow;  // skipped above --naive-max
};

void run_gemm(const Options& o, int nthreads) {
  const int N = o.size;
  const double flops = 2.0 * N * N * N;
  std::printf("\n== GEMM %dx%dx%d  (FP32, %s, tile %dx%d) ==\n", N, N, N, simd_isa(), micro_tile_m(), micro_tile_n());
  auto A = randn(size_t(N) * N, 1), B = randn(size_t(N) * N, 2);
  std::vector<float> C(size_t(N) * N), ref(size_t(N) * N);
  set_num_threads(1);
  gemm_simd(N, N, N, A.data(), B.data(), ref.data());

  const std::vector<GemmCase> cases = {
      {"1. naive i-j-k", 1, [](int M, int N, int K, const float* a, const float* b, float* c) { gemm_naive(M, N, K, a, b, c); }, true},
      {"2. reordered i-k-j", 1, [](int M, int N, int K, const float* a, const float* b, float* c) { gemm_reordered(M, N, K, a, b, c); }, true},
      {"3. reordered + tiled 64", 1, [](int M, int N, int K, const float* a, const float* b, float* c) { gemm_tiled(M, N, K, a, b, c, 64); }, false},
      {"4. packed micro-kernel (1 thread)", 1, [](int M, int N, int K, const float* a, const float* b, float* c) { gemm_simd(M, N, K, a, b, c); }, false},
      {"5. packed micro-kernel + OpenMP", nthreads, [](int M, int N, int K, const float* a, const float* b, float* c) { gemm_simd(M, N, K, a, b, c); }, false},
  };
  struct Row { std::string name; int threads; double ms, gf; bool ok; };
  std::vector<Row> rows;
  for (const auto& k : cases) {
    if (k.slow && N > o.naive_max && k.name[0] == '1') continue;
    sample_peak(nthreads);
    set_num_threads(k.threads);
    const double ms = time_ms([&] { k.fn(N, N, N, A.data(), B.data(), C.data()); }, k.slow ? 3 : o.reps);
    sample_peak(nthreads);
    rows.push_back({k.name, k.threads, ms, flops / ms / 1e6, max_rel_err(C, ref) < 1e-4});
  }
  std::printf("%-36s %8s %10s %10s %9s %9s %s\n", "kernel", "threads", "time (ms)", "GFLOPS", "speedup", "% peak", "check");
  for (const auto& r : rows) {
    const double peak = g_peak[r.threads == 1 ? 0 : 1];
    std::printf("%-36s %8d %10.2f %10.2f %8.1fx %8.1f%% %s\n", r.name.c_str(), r.threads, r.ms, r.gf,
                rows[0].ms / r.ms, 100.0 * r.gf / peak, r.ok ? "ok" : "MISMATCH");
    csv_row("gemm", r.name, std::to_string(N), r.threads, r.ms, r.gf, 0);
  }
  std::printf("  (peak = highest FMA throughput observed: %.1f GFLOPS x1 thread, %.1f x%d threads)\n", g_peak[0], g_peak[1], nthreads);
  set_num_threads(nthreads);
}

void run_scaling(const Options& o) {
  const int N = std::max(o.size, 1024);
  auto A = randn(size_t(N) * N, 1), B = randn(size_t(N) * N, 2);
  std::vector<float> C(size_t(N) * N);
  std::printf("\n== thread scaling, packed GEMM %d^3 ==\n%8s %10s %10s %10s\n", N, "threads", "ms", "GFLOPS", "efficiency");
  double g1 = 0;
  for (int t : {1, 2, 4, 8, 16}) {
    if (t > std::max(1, num_threads()) && t != 1) { if (t > 16) break; }
    set_num_threads(t);
    const double ms = time_ms([&] { gemm_simd(N, N, N, A.data(), B.data(), C.data()); }, o.reps);
    const double gf = 2.0 * N * N * N / ms / 1e6;
    if (t == 1) g1 = gf;
    std::printf("%8d %10.2f %10.2f %9.0f%%\n", t, ms, gf, 100.0 * gf / (g1 * t));
    csv_row("scaling", "gemm_simd", std::to_string(N), t, ms, gf, 0);
  }
}

// ---- GEMV (the decode path) ------------------------------------------------------------------
void run_gemv(const Options& o, double bw, int nthreads) {
  std::printf("\n== GEMV  y = W x  (the decode path), DRAM read bandwidth %.1f GB/s ==\n", bw);
  std::printf("hot  = the same matrix every call (stays in L2/L3); cold = rotating through >=160 MB of\n"
              "matrices so every call streams from DRAM, like a real model whose weights exceed the cache.\n");
  std::printf("%-10s %-15s %4s | %9s %8s | %9s %8s %7s\n", "N x K", "kernel", "thr", "hot us", "GB/s", "cold us", "GB/s", "%DRAM");
  struct Shape { int n, k; };
  for (auto s : {Shape{512, 512}, Shape{2048, 512}, Shape{512, 2048}, Shape{8192, 8192}}) {
    const size_t elems = size_t(s.n) * s.k;
    const int copies = int(std::max<size_t>(2, (size_t(160) << 20) / (elems * 4)));
    std::vector<float> big(elems * copies);
    auto first = randn(elems, 3);
    for (int c = 0; c < copies; ++c) std::memcpy(big.data() + elems * c, first.data(), elems * 4);
    auto x = randn(s.k, 4);
    std::vector<float> y(s.n), ref(s.n);
    gemv_f32_naive(s.n, s.k, first.data(), x.data(), ref.data());
    struct K { const char* name; int threads; GemvF32 fn; };
    for (auto k : {K{"naive", 1, gemv_f32_naive}, K{"simd", 1, gemv_f32_simd}, K{"simd + OpenMP", nthreads, gemv_f32_simd}}) {
      set_num_threads(k.threads);
      const int reps = (elems * 4 >= (64u << 20) && k.threads == 1 && k.name[1] == 'a') ? 3 : o.reps * 4;
      const double hot = time_ms([&] { k.fn(s.n, s.k, first.data(), x.data(), y.data()); }, reps);
      int idx = 0;
      const double cold = time_ms([&] { k.fn(s.n, s.k, big.data() + elems * (idx++ % copies), x.data(), y.data()); }, reps);
      const double ghot = elems * 4 / hot / 1e6, gcold = elems * 4 / cold / 1e6;
      char shape[32];
      std::snprintf(shape, sizeof shape, "%dx%d", s.n, s.k);
      std::printf("%-10s %-15s %4d | %9.1f %8.1f | %9.1f %8.1f %6.0f%% %s\n", shape, k.name, k.threads, hot * 1000, ghot,
                  cold * 1000, gcold, 100 * gcold / bw, max_rel_err(y, ref) < 1e-4 ? "" : "MISMATCH");
      csv_row("gemv_hot", k.name, shape, k.threads, hot, 0, ghot);
      csv_row("gemv_cold", k.name, shape, k.threads, cold, 0, gcold);
    }
  }

  std::printf("\n-- thread count vs. GEMV, 2048x512 (4 MB: hot) --\n");
  auto W = randn(size_t(2048) * 512, 5), x = randn(512, 6);
  std::vector<float> y(2048);
  const size_t saved = gemv_parallel_threshold();
  gemv_parallel_threshold() = 0;
  for (int t : {1, 2, 4, 8, 16}) {
    set_num_threads(t);
    const double ms = time_ms([&] { gemv_f32_simd(2048, 512, W.data(), x.data(), y.data()); }, o.reps * 4);
    std::printf("  %2d threads: %8.1f us  (%.1f GB/s)\n", t, ms * 1000, 2048.0 * 512 * 4 / ms / 1e6);
  }
  gemv_parallel_threshold() = saved;
  set_num_threads(nthreads);
}

// ---- tile-size sweeps (graphs for the README) -------------------------------------------------
void run_sweep(const Options& o) {
  const int N = o.size;
  auto A = randn(size_t(N) * N, 1), B = randn(size_t(N) * N, 2);
  std::vector<float> C(size_t(N) * N);
  const double flops = 2.0 * N * N * N;
  set_num_threads(1);
  std::printf("\n== Kernel 1 tile size sweep (%d^3, 1 thread): GFLOPS ==\n", N);
  for (int bs : {8, 16, 32, 64, 128, 256, 512}) {
    const double ms = time_ms([&] { gemm_tiled(N, N, N, A.data(), B.data(), C.data(), bs); }, 3);
    std::printf("  block %4d: %7.2f GFLOPS\n", bs, flops / ms / 1e6);
    csv_row("sweep_tiled", "block", std::to_string(bs), 1, ms, flops / ms / 1e6, 0);
  }
  std::printf("\n== Kernel 2 blocking sweep (%d^3, 1 thread): GFLOPS, rows = mc, cols = kc ==\n      ", N);
  const int kcs[] = {32, 64, 128, 256, 512, 1024}, mcs[] = {12, 36, 72, 144, 288, 576};
  for (int kc : kcs) std::printf("%8d", kc);
  std::printf("\n");
  for (int mc : mcs) {
    std::printf("%5d ", mc);
    for (int kc : kcs) {
      GemmBlocking blk;
      blk.mc = mc;
      blk.kc = kc;
      const double ms = time_ms([&] { gemm_simd(N, N, N, A.data(), B.data(), C.data(), false, blk); }, 3);
      std::printf("%8.1f", flops / ms / 1e6);
      csv_row("sweep_packed", "mc_kc", std::to_string(mc) + "x" + std::to_string(kc), 1, ms, flops / ms / 1e6, 0);
    }
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  bool all = argc == 1;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* s) { return std::strcmp(argv[i], s) == 0; };
    auto next = [&]() { return i + 1 < argc ? argv[++i] : "0"; };
    if (is("--all")) all = true;
    else if (is("--peak")) o.peak = true;
    else if (is("--gemm")) o.gemm = true;
    else if (is("--gemv")) o.gemv = true;
    else if (is("--scaling")) o.scaling = true;
    else if (is("--sweep")) o.sweep = true;
    else if (is("--size")) o.size = std::atoi(next());
    else if (is("--threads")) o.threads = std::atoi(next());
    else if (is("--reps")) o.reps = std::atoi(next());
    else if (is("--naive-max")) o.naive_max = std::atoi(next());
    else if (is("--csv")) o.csv = next();
    else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
  }
  if (all) o.peak = o.gemm = o.gemv = o.scaling = true;
  if (!o.csv.empty()) {
    g_csv = std::fopen(o.csv.c_str(), "w");
    if (g_csv) std::fprintf(g_csv, "bench,name,params,threads,ms,gflops,gbs\n");
  }
  const int hw = num_threads();
  const int nthreads = o.threads > 0 ? o.threads : physical_cores();
  std::printf("BareMetal benchmark | ISA %s | %d hardware threads, using %d | reps %d\n", simd_isa(), hw, nthreads, o.reps);

  double bw = 0;
  sample_peak(nthreads);
  if (o.peak || o.gemv) bw = read_bandwidth_gbs(nthreads);
  if (o.peak) {
    std::printf("\n== hardware ceilings (measured) ==\n");
    std::printf("  FP32 FMA peak, 1 thread : %8.1f GFLOPS\n  FP32 FMA peak, %2d threads: %8.1f GFLOPS\n", g_peak[0], nthreads, g_peak[1]);
    std::printf("  DRAM read bandwidth     : %8.1f GB/s  (best over 1..16 threads)\n", bw);
    std::printf("  => FP32 decode floor for a 75 MB model: %.2f ms/token; INT8 (19 MB): %.2f ms/token\n", 75.0 / bw, 19.0 / bw);
    csv_row("ceiling", "fma_peak", "", 1, 0, g_peak[0], 0);
    csv_row("ceiling", "fma_peak", "", nthreads, 0, g_peak[1], 0);
    csv_row("ceiling", "dram_read", "", nthreads, 0, 0, bw);
  }
  set_num_threads(nthreads);
  if (o.gemm) run_gemm(o, nthreads);
  if (o.gemv) run_gemv(o, bw, nthreads);
  if (o.scaling) run_scaling(o);
  if (o.sweep) run_sweep(o);
  if (g_csv) std::fclose(g_csv);
  return 0;
}
