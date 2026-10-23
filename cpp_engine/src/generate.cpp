// generate: prompt in, bytes out.  ./generate --weights model.bin --prompt "ROMEO:" -n 300
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "transformer.hpp"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: generate --weights FILE [--prompt TEXT] [-n TOKENS] [--temp T] [--top-k K]\n"
               "                [--top-p P] [--seed S] [--threads T] [--kernels fast|naive]\n"
               "  --temp 0 selects greedy decoding. Tokens are raw bytes (byte-level vocabulary).\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights, prompt = "\n";
  int n_new = 200, top_k = 0;
  float temp = 0.8f, top_p = 1.0f;
  uint64_t seed = 1;
  int threads = bm::physical_cores();
  bool naive = false;

  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0 && i + 1 < argc; };
    if (arg("--weights")) weights = argv[++i];
    else if (arg("--prompt")) prompt = argv[++i];
    else if (arg("-n") || arg("--n")) n_new = std::atoi(argv[++i]);
    else if (arg("--temp")) temp = static_cast<float>(std::atof(argv[++i]));
    else if (arg("--top-k")) top_k = std::atoi(argv[++i]);
    else if (arg("--top-p")) top_p = static_cast<float>(std::atof(argv[++i]));
    else if (arg("--seed")) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (arg("--threads")) threads = std::atoi(argv[++i]);
    else if (arg("--kernels")) naive = std::strcmp(argv[++i], "naive") == 0;
    else { usage(); return 1; }
  }
  if (weights.empty()) { usage(); return 1; }

  try {
    bm::ModelFile model(weights);
    bm::set_num_threads(threads);
    bm::Transformer net(model, naive ? bm::naive_kernels() : bm::optimized_kernels());
    bm::Sampler sampler(model.cfg.vocab, temp, top_k, top_p, seed);
    const int max_seq = model.cfg.max_seq;

    std::fprintf(stderr, "%s | %d layers, d=%d, %d heads | %.2f MB weights, %.2f MB KV-cache | %s kernels, %d threads\n",
                 model.int8 ? "INT8" : "FP32", model.cfg.n_layers, model.cfg.d_model, model.cfg.n_heads,
                 model.file_bytes() / 1e6, net.kv_cache_bytes() / 1e6, naive ? "naive" : bm::simd_isa(), threads);

    using clock = std::chrono::steady_clock;
    const float* logits = nullptr;
    std::fwrite(prompt.data(), 1, prompt.size(), stdout);

    const auto t0 = clock::now();
    std::vector<int> ids(prompt.begin(), prompt.end());
    for (int& t : ids) t &= 0xff;  // bytes are unsigned token ids
    if (static_cast<int>(ids.size()) > max_seq) ids.resize(max_seq);
    if (!ids.empty()) logits = net.prefill(ids.data(), static_cast<int>(ids.size()));
    const auto t1 = clock::now();

    int generated = 0;
    for (; generated < n_new && net.pos() < max_seq; ++generated) {
      const int tok = sampler.sample(logits);
      std::fputc(tok, stdout);
      logits = net.forward(tok);
    }
    const auto t2 = clock::now();
    std::fputc('\n', stdout);

    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::fprintf(stderr, "prefill %zu tokens: %.2f ms | decode %d tokens: %.2f ms (%.3f ms/token, %.1f tok/s)%s\n",
                 prompt.size(), ms(t0, t1), generated, ms(t1, t2), generated ? ms(t1, t2) / generated : 0.0,
                 generated ? 1000.0 * generated / ms(t1, t2) : 0.0,
                 generated < n_new ? " [stopped at max_seq]" : "");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
