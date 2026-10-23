// End-to-end check: the C++ forward pass must reproduce the logits PyTorch produced
// (reference.bin) for the same weights, for the FP32 and INT8 model files.
//   test_reference <model_fp32.bin> <model_int8.bin> <reference.bin>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "testing.hpp"
#include "transformer.hpp"

using namespace bm;

static bool exists(const char* p) { return std::ifstream(p).good(); }

static double max_logit_diff(const char* model_path, const Reference& ref, const std::vector<float>& want) {
  ModelFile model(model_path);
  Transformer net(model);
  double worst = 0;
  for (size_t t = 0; t < ref.ids.size(); ++t) {
    const float* got = net.forward(ref.ids[t]);
    for (int v = 0; v < ref.vocab; ++v)
      worst = std::max(worst, double(std::fabs(got[v] - want[t * ref.vocab + v])));
  }
  return worst;
}

static const char *g_fp32, *g_int8, *g_ref;

static void test_fp32_logits_match_pytorch() {
  Reference ref = load_reference(g_ref);
  const double d = max_logit_diff(g_fp32, ref, ref.logits_fp32);
  std::printf("      FP32 max |logit diff| over %zu positions: %.3g\n", ref.ids.size(), d);
  CHECK(d < 1e-3);
}

static void test_int8_logits_match_pytorch_fake_quant() {
  Reference ref = load_reference(g_ref);
  const double d = max_logit_diff(g_int8, ref, ref.logits_int8);
  std::printf("      INT8 max |logit diff| vs dequantised-weights reference: %.3g\n", d);
  CHECK(d < 1e-3);
}

static void test_kv_cache_is_stateless_across_resets() {
  Reference ref = load_reference(g_ref);
  ModelFile model(g_fp32);
  Transformer net(model);
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

// Batched prefill (GEMMs) must equal feeding the same tokens one at a time (GEMVs), for both
// kernel sets and for the INT8 model, and generation must continue seamlessly afterwards.
static void check_prefill_matches_stepwise(const char* path, KernelTable kernels, double tol) {
  Reference ref = load_reference(g_ref);
  ModelFile model(path);
  const int n = static_cast<int>(ref.ids.size());
  Transformer step(model, naive_kernels());
  std::vector<std::vector<float>> want;
  for (int i = 0; i < n; ++i) { const float* l = step.forward(ref.ids[i]); want.emplace_back(l, l + ref.vocab); }

  for (int split : {n, n / 2, 1}) {  // prefill the whole prompt, or a prefix and then continue
    Transformer net(model, kernels);
    const float* l = net.prefill(ref.ids.data(), split);
    double worst = 0;
    for (int v = 0; v < ref.vocab; ++v) worst = std::max(worst, double(std::fabs(l[v] - want[split - 1][v])));
    for (int i = split; i < n; ++i) {  // the cache written by prefill must support decoding
      l = net.forward(ref.ids[i]);
      for (int v = 0; v < ref.vocab; ++v) worst = std::max(worst, double(std::fabs(l[v] - want[i][v])));
    }
    CHECK(net.pos() == n);
    CHECK(worst < tol);
  }
  // a second prefill chunk appended to an existing sequence
  Transformer net(model, kernels);
  net.prefill(ref.ids.data(), n / 2);
  const float* l = net.prefill(ref.ids.data() + n / 2, n - n / 2);
  double worst = 0;
  for (int v = 0; v < ref.vocab; ++v) worst = std::max(worst, double(std::fabs(l[v] - want[n - 1][v])));
  CHECK(worst < tol);
}

static void test_prefill_matches_stepwise_forward() {
  check_prefill_matches_stepwise(g_fp32, naive_kernels(), 1e-4);
  check_prefill_matches_stepwise(g_fp32, optimized_kernels(), 1e-3);
  check_prefill_matches_stepwise(g_int8, optimized_kernels(), 1e-3);
}

static void test_optimized_kernels_match_pytorch() {
  Reference ref = load_reference(g_ref);
  for (int threads : {1, 4}) {
    set_num_threads(threads);
    ModelFile model(g_fp32);
    Transformer net(model, optimized_kernels());
    double worst = 0;
    for (size_t t = 0; t < ref.ids.size(); ++t) {
      const float* got = net.forward(ref.ids[t]);
      for (int v = 0; v < ref.vocab; ++v) worst = std::max(worst, double(std::fabs(got[v] - ref.logits_fp32[t * ref.vocab + v])));
    }
    std::printf("      optimised kernels, %d thread(s): max |logit diff| %.3g\n", threads, worst);
    CHECK(worst < 1e-3);
  }
  set_num_threads(num_threads());
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
  if (argc != 4) { std::fprintf(stderr, "usage: test_reference fp32.bin int8.bin reference.bin\n"); return 1; }
  g_fp32 = argv[1]; g_int8 = argv[2]; g_ref = argv[3];
  if (!exists(g_fp32) || !exists(g_int8) || !exists(g_ref)) {
    std::printf("SKIP: run python_scratch/export_weights.py first\n");
    return 77;
  }
  RUN(test_fp32_logits_match_pytorch);
  RUN(test_int8_logits_match_pytorch_fake_quant);
  RUN(test_kv_cache_is_stateless_across_resets);
  RUN(test_prefill_matches_stepwise_forward);
  RUN(test_optimized_kernels_match_pytorch);
  RUN(test_context_limit_and_bad_token);
  RUN(test_rejects_bad_files);
  return finish();
}
