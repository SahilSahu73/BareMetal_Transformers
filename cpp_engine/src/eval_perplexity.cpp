// Validation perplexity of a model file under the engine's own numerics.
//   eval_perplexity --weights model.bin --val checkpoints/val_tokens.bin [--act w8a8] [--kv-int8]
// Uses the same non-overlapping max_seq windows as python_scratch/eval_quant.py, so the two
// numbers are directly comparable (the Python side fake-quantises the same way).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "transformer.hpp"

int main(int argc, char** argv) {
  std::string weights, val_path;
  bool kv_int8 = false, naive = false;
  bm::ActMode act = bm::ActMode::F32;
  int max_windows = 0, threads = bm::physical_cores();
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* n) { return std::strcmp(argv[i], n) == 0 && i + 1 < argc; };
    if (arg("--weights")) weights = argv[++i];
    else if (arg("--val")) val_path = argv[++i];
    else if (arg("--windows")) max_windows = std::atoi(argv[++i]);
    else if (arg("--threads")) threads = std::atoi(argv[++i]);
    else if (arg("--act")) act = std::strcmp(argv[++i], "w8a8") == 0 ? bm::ActMode::Q8 : bm::ActMode::F32;
    else if (arg("--kernels")) naive = std::strcmp(argv[++i], "naive") == 0;
    else if (std::strcmp(argv[i], "--kv-int8") == 0) kv_int8 = true;
    else {
      std::fprintf(stderr, "usage: eval_perplexity --weights FILE --val val_tokens.bin [--windows N] [--act w8a32|w8a8] [--kv-int8]\n");
      return 1;
    }
  }
  if (weights.empty() || val_path.empty()) { std::fprintf(stderr, "--weights and --val are required\n"); return 1; }

  try {
    std::ifstream f(val_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + val_path);
    const std::vector<unsigned char> val((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    bm::ModelFile model(weights);
    bm::set_num_threads(threads);
    bm::TransformerOptions opt;
    opt.kernels = naive ? bm::naive_kernels() : bm::optimized_kernels();
    opt.act = act;
    opt.kv_int8 = kv_int8;
    bm::Transformer net(model, opt);

    const int T = model.cfg.max_seq, V = model.cfg.vocab;
    int windows = static_cast<int>((val.size() - 1) / T);
    if (max_windows > 0) windows = std::min(windows, max_windows);
    std::vector<int> ids(T);
    std::vector<float> logits(static_cast<size_t>(T) * V);

    double nll = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int w = 0; w < windows; ++w) {
      const size_t base = static_cast<size_t>(w) * T;
      for (int i = 0; i < T; ++i) ids[i] = val[base + i];
      net.reset();
      net.prefill_all(ids.data(), T, logits.data());
      for (int i = 0; i < T; ++i) {
        const float* l = logits.data() + static_cast<size_t>(i) * V;
        float mx = l[0];
        for (int v = 1; v < V; ++v) mx = std::max(mx, l[v]);
        double sum = 0;
        for (int v = 0; v < V; ++v) sum += std::exp(static_cast<double>(l[v] - mx));
        nll += std::log(sum) + mx - l[val[base + i + 1]];
      }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double tokens = static_cast<double>(windows) * T;
    std::printf("%s %s%s | %d windows, %.0f tokens | loss %.5f | perplexity %.4f | %.1fs\n",
                !model.int8 ? "FP32" : act == bm::ActMode::Q8 ? "INT8 W8A8" : "INT8 W8A32",
                naive ? "naive" : "fast", kv_int8 ? " +KV-int8" : "", windows, tokens, nll / tokens,
                std::exp(nll / tokens), secs);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
