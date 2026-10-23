// End-to-end checks of the C++ engine against PyTorch (reference.bin) for every numeric mode:
// FP32, INT8 weights (W8A32), INT8 weights + INT8 activations (W8A8), and the INT8 KV-cache.
//   test_reference <model_fp32.bin> <model_int8.bin> <reference.bin> [--smooth-only]
// --smooth-only skips the modes that round activations or K/V (W8A8, INT8 KV-cache): use it for
// random-weight models, whose chaotic activations flip rounding ties far more often than a trained one.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "quantize.hpp"
#include "testing.hpp"
#include "transformer.hpp"

using namespace bm;

static const char *g_fp32, *g_int8, *g_ref;
static bool g_smooth_only = false;

// Statistics over a set of position logits: the largest and the mean absolute difference, and
// how often the arg-max token agrees.
struct Diff {
  double max = 0, mean = 0, top1 = 1;
};

static bool exists(const char* p) { return std::ifstream(p).good(); }

// Agreement required between two implementations of one mode.
//
// FP32 and W8A32 are smooth functions of the weights: they must match PyTorch to ~1e-5, and the
// bound below is 100x that. Quantised activations / KV vectors are different: x / scale rounds to
// an integer, so two computations that differ at the 1e-7 level (float32 here, float64 in the
// reference, or just a different summation order) occasionally land on opposite sides of a
// rounding tie and differ by a whole quantisation step in one element. That is intrinsic: PyTorch
// float32 vs PyTorch float64 on the same W8A8 model already differs by max 8.5e-2 / mean 8.4e-3
// (measured; the injected quantisation error itself is max 1.6e-1 / mean 3.3e-2). The bounds for
// those modes sit just above that noise floor; a wrong scale or index would blow far past them.
// Exact correctness of the integer kernels and the quantiser is asserted bit for bit elsewhere
// (test_kernels, test_cpp_quantiser_matches_exporter), and aggregate agreement by eval_perplexity.
struct Tol {
  double max, mean, top1;
};
static const Tol kSmooth = {1e-3, 1e-4, 1.0};
static const Tol kKv8 = {3e-2, 5e-3, 0.9};
static const Tol kW8A8 = {0.3, 5e-2, 0.9};

struct Mode {
  const char* name;
  bool int8_model;
  ActMode act;
  bool kv8;
  RefVariant ref;
  Tol tol;
};

static const Mode kModes[] = {
    {"FP32", false, ActMode::F32, false, kRefFp32, kSmooth},
    {"INT8 weights (W8A32)", true, ActMode::F32, false, kRefW8, kSmooth},
    {"INT8 W8A8", true, ActMode::Q8, false, kRefW8A8, kW8A8},
    {"INT8 + INT8 KV-cache", true, ActMode::F32, true, kRefW8Kv8, kKv8},
    {"FP32 + INT8 KV-cache", false, ActMode::F32, true, kRefFp32Kv8, kKv8},
};

static bool skipped(const Mode& m) { return g_smooth_only && (m.act == ActMode::Q8 || m.kv8); }

static bool within(const Tol& t, const Diff& d) { return d.max < t.max && d.mean < t.mean && d.top1 >= t.top1; }

static TransformerOptions options(const Mode& m, KernelTable k) {
  TransformerOptions o;
  o.kernels = k;
  o.act = m.act;
  o.kv_int8 = m.kv8;
  return o;
}

static Diff compare(const float* got, const float* want, int positions, int V) {
  Diff d;
  int agree = 0;
  for (int t = 0; t < positions; ++t) {
    const float* g = got + size_t(t) * V;
    const float* w = want + size_t(t) * V;
    for (int v = 0; v < V; ++v) {
      const double e = std::fabs(g[v] - w[v]);
      d.max = std::max(d.max, e);
      d.mean += e;
    }
    agree += std::max_element(g, g + V) - g == std::max_element(w, w + V) - w;
  }
  d.mean /= double(positions) * V;
  d.top1 = double(agree) / positions;
  return d;
}

// Decode path (one token at a time, GEMV kernels) and prefill path (all tokens, GEMM kernels).
static void check_mode(const Mode& m, bool naive) {
  Reference ref = load_reference(g_ref);
  ModelFile model(m.int8_model ? g_int8 : g_fp32);
  const auto& want = ref.logits[m.ref];
  const int n = static_cast<int>(ref.ids.size()), V = ref.vocab;
  const KernelTable kt = naive ? naive_kernels() : optimized_kernels();

  Transformer step(model, options(m, kt));
  std::vector<float> dec_all;
  for (int t = 0; t < n; ++t) { const float* l = step.forward(ref.ids[t]); dec_all.insert(dec_all.end(), l, l + V); }
  const Diff dec = compare(dec_all.data(), want.data(), n, V);

  Transformer batch(model, options(m, kt));
  std::vector<float> all(size_t(n) * V);
  batch.prefill_all(ref.ids.data(), n, all.data());
  const Diff pre = compare(all.data(), want.data(), n, V);

  std::printf("      %-24s %-9s decode max %.1e mean %.1e top1 %.0f%% | prefill max %.1e mean %.1e top1 %.0f%%\n", m.name,
              naive ? "naive" : "optimised", dec.max, dec.mean, 100 * dec.top1, pre.max, pre.mean, 100 * pre.top1);
  CHECK(within(m.tol, dec));
  CHECK(within(m.tol, pre));
}

static void test_all_modes_match_pytorch_naive_kernels() {
  for (const auto& m : kModes)
    if (!skipped(m)) check_mode(m, true);
}
static void test_all_modes_match_pytorch_optimised_kernels() {
  for (int threads : {1, 4}) {
    set_num_threads(threads);
    for (const auto& m : kModes)
      if (!skipped(m)) check_mode(m, false);
  }
  set_num_threads(num_threads());
}

static void test_kv_cache_is_stateless_across_resets() {
  Reference ref = load_reference(g_ref);
  ModelFile model(g_fp32);
  for (bool kv8 : {false, true}) {
    TransformerOptions o;
    o.kv_int8 = kv8;
    Transformer net(model, o);
    std::vector<float> first, second;
    for (int pass = 0; pass < 2; ++pass) {
      net.reset();
      for (int id : ref.ids) {
        const float* l = net.forward(id);
        (pass ? second : first).insert((pass ? second : first).end(), l, l + ref.vocab);
      }
    }
    CHECK(first == second);  // bit-identical: no stale cache entries leak between sequences
  }
}

// Batched prefill (GEMMs) must equal feeding the same tokens one at a time (GEMVs), in every mode,
// and generation must continue seamlessly afterwards.
static void test_prefill_matches_stepwise_forward() {
  Reference ref = load_reference(g_ref);
  const int n = static_cast<int>(ref.ids.size()), V = ref.vocab;
  for (const auto& m : kModes) {
    if (skipped(m)) continue;
    ModelFile model(m.int8_model ? g_int8 : g_fp32);
    Transformer step(model, options(m, naive_kernels()));
    std::vector<float> want;
    for (int i = 0; i < n; ++i) { const float* l = step.forward(ref.ids[i]); want.insert(want.end(), l, l + V); }

    for (int split : {n, n / 2, 1}) {  // the whole prompt, or a prefix and then decode the rest
      Transformer net(model, options(m, optimized_kernels()));
      std::vector<float> got(want.size());
      const float* l = net.prefill(ref.ids.data(), split);
      std::copy(l, l + V, got.begin() + size_t(split - 1) * V);
      for (int i = split; i < n; ++i) { l = net.forward(ref.ids[i]); std::copy(l, l + V, got.begin() + size_t(i) * V); }
      CHECK(net.pos() == n);
      // compare only the positions we actually obtained (last prefill position, then the decoded ones)
      std::vector<float> g2(got.begin() + size_t(split - 1) * V, got.end()), w2(want.begin() + size_t(split - 1) * V, want.end());
      CHECK(within(m.tol, compare(g2.data(), w2.data(), n - split + 1, V)));
    }
    Transformer net(model, options(m, optimized_kernels()));  // a second prefill chunk appended to a sequence
    net.prefill(ref.ids.data(), n / 2);
    const float* l = net.prefill(ref.ids.data() + n / 2, n - n / 2);
    CHECK(within(m.tol, compare(l, &want[size_t(n - 1) * V], 1, V)));
  }
}

// The C++ quantiser must reproduce the exporter's INT8 file bit for bit from the FP32 file.
static void test_cpp_quantiser_matches_exporter() {
  ModelFile f(g_fp32), q(g_int8);
  auto check = [&](const Matrix& a, const Matrix& b) {
    std::vector<int8_t> Q(size_t(a.rows) * a.cols);
    std::vector<float> sc(a.rows);
    quantize_matrix_q8(a.f32, a.rows, a.cols, Q.data(), sc.data());
    CHECK(std::memcmp(sc.data(), b.scales, sc.size() * 4) == 0);
    CHECK(std::memcmp(Q.data(), b.q8, Q.size()) == 0);
  };
  check(f.emb, q.emb);
  for (size_t l = 0; l < f.layers.size(); ++l) {
    check(f.layers[l].wq, q.layers[l].wq); check(f.layers[l].wk, q.layers[l].wk);
    check(f.layers[l].wv, q.layers[l].wv); check(f.layers[l].wo, q.layers[l].wo);
    check(f.layers[l].w1, q.layers[l].w1); check(f.layers[l].w2, q.layers[l].w2);
  }
}

static void test_kv_int8_shrinks_cache() {
  ModelFile model(g_fp32);
  TransformerOptions o;
  Transformer fp(model, o);
  o.kv_int8 = true;
  Transformer q8(model, o);
  const double ratio = double(fp.kv_cache_bytes()) / q8.kv_cache_bytes();
  std::printf("      KV-cache: %.2f MB fp32 -> %.2f MB int8 (%.2fx)\n", fp.kv_cache_bytes() / 1e6, q8.kv_cache_bytes() / 1e6, ratio);
  CHECK(ratio > 3.0);
}

static void test_context_limit_and_bad_token() {
  ModelFile model(g_fp32);
  Transformer net(model);
  bool bad_token = false, overflow = false;
  try { net.forward(model.cfg.vocab); } catch (const std::out_of_range&) { bad_token = true; }
  CHECK(bad_token);
  for (int i = 0; i < model.cfg.max_seq; ++i) net.forward(i % model.cfg.vocab);
  try { net.forward(0); } catch (const std::runtime_error&) { overflow = true; }
  CHECK(overflow);
}

static void test_rejects_bad_files() {
  bool threw = false;
  try { ModelFile m("/nonexistent/model.bin"); } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { ModelFile m(g_ref); } catch (const std::runtime_error&) { threw = true; }  // wrong magic
  CHECK(threw);
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) { std::fprintf(stderr, "usage: test_reference fp32.bin int8.bin reference.bin [--smooth-only]\n"); return 1; }
  g_fp32 = argv[1]; g_int8 = argv[2]; g_ref = argv[3];
  g_smooth_only = argc == 5 && std::strcmp(argv[4], "--smooth-only") == 0;
  if (!exists(g_fp32) || !exists(g_int8) || !exists(g_ref)) {
    std::printf("SKIP: run python_scratch/export_weights.py first\n");
    return 77;
  }
  RUN(test_all_modes_match_pytorch_naive_kernels);
  RUN(test_all_modes_match_pytorch_optimised_kernels);
  RUN(test_kv_cache_is_stateless_across_resets);
  RUN(test_prefill_matches_stepwise_forward);
  RUN(test_cpp_quantiser_matches_exporter);
  RUN(test_kv_int8_shrinks_cache);
  RUN(test_context_limit_and_bad_token);
  RUN(test_rejects_bad_files);
  return finish();
}
