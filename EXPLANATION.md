# BareMetal-Transformer — What We're Building and How It Works

This document explains the whole project before any code is written: what each piece is, the math behind it, the hardware reasoning behind the C++ engine, how we verify correctness, and the order things get built in. It also works as a study guide. Interviewers will ask about exactly these internals, so every section below is something you should be able to explain on a whiteboard.

Section 10 lists the decisions I need you to confirm before building starts.

---

## 0. The One-Paragraph Version

We train a small GPT-style (decoder-only) Transformer on text. First we write every layer in **pure NumPy**, deriving each backward pass by hand, and check those gradients against PyTorch autograd. A **PyTorch** version of the *identical* architecture trains the larger model quickly, optionally on a Colab/Kaggle GPU. The trained weights are exported to a **custom binary file**, in FP32 and in INT8. A **zero-dependency C++17 engine** loads that file and generates text token by token using a KV-cache. Its matrix multiplication is optimized in four steps (naive → cache-tiled → AVX2/NEON SIMD + OpenMP → INT8), and each step is benchmarked. The point of the project is to show *why* each optimization helps, with your own measurements.

```
            ┌──────────────── python_scratch/ ────────────────┐
 text  ──►  │ NumPy layers (manual fwd/bwd) ── gradcheck ──► PyTorch autograd
            │            │                                     │
            │   NumPy tiny model trains            PyTorch model trains (CPU/Colab)
            │                                                  │
            │                                   export_weights.py (FP32 + INT8)
            └──────────────────────────────────────────────────┼──────────────┘
                                                               ▼
                                                    checkpoints/model_*.bin
                                                               │
            ┌──────────────── cpp_engine/ ─────────────────────▼──────────────┐
            │ mmap/load → Tensor views → Transformer forward (KV-cache)       │
            │                 └── GEMM/GEMV kernels: naive│tiled│SIMD│INT8    │
            │ sampler → generated text        benchmark CLI → GFLOPS, ms/tok  │
            └─────────────────────────────────────────────────────────────────┘
```

---

## 1. Notes on the Current README

Most of the README describes the target accurately. Three things need to change before this goes on a resume:

1. **The benchmark numbers are placeholders.** Nobody has measured them yet. A resume link to a repo with made-up numbers is the fastest way to lose credibility in an interview ("how did you get 135 GFLOPS from INT8 on that CPU?"). Once the engine works, we'll replace every number with **measured results from your machine**, and state the CPU, compiler, and flags next to them.
2. **"INT8 GEMM is faster than FP32 SIMD GEMM" isn't automatically true.** For a large, compute-bound GEMM (1024³), W8A32 (INT8 weights, FP32 activations) has to convert the weights back to float before the FMA. That adds instructions, so it's usually *slower* than a good FP32 kernel. INT8 helps in two places:
   - **Decoding (GEMV, batch = 1)**, which is **memory-bandwidth bound**. Moving 4× fewer bytes gives close to 4× speedup. This is the headline result for an inference engine.
   - **W8A8**, where activations are also quantized to INT8 and multiplied with integer instructions (`_mm256_madd_epi16`). This can beat FP32 in compute-bound GEMM too.

   We'll implement both and report honestly which wins where. That's a much stronger story than a single big number.
3. **The NumPy baseline isn't "pure".** NumPy's `@` calls an optimized BLAS (OpenBLAS/MKL). So "C++ vs NumPy" compares your hand-written kernels against a professional BLAS plus Python overhead. That's fine, but the writeup should say so. Our real baseline is Kernel 0 (naive C++).

**About the GPU:** this project barely needs one. The C++ engine is CPU-only by design, and the NumPy model is CPU-only. The GPU is only useful for training the 6-layer/512-dim PyTorch model faster (≈20 min on a Colab T4 vs. a few hours on your CPU). We'll provide a Colab notebook for that, but CPU training will also work.

**Your machine:** AMD Ryzen 7 7730U (Zen 3, 8 cores/16 threads, AVX2 + FMA, **no AVX-512, no VNNI**), 32 KB L1d/core, 512 KB L2/core, 16 MB shared L3, GCC 16. All tuning numbers below are for this CPU. NumPy and PyTorch aren't installed yet; we'll use a venv.

---

## 2. The Model: A Small GPT

### 2.1 Architecture (decoder-only, pre-norm)

```
tokens (T,)  ──►  Embedding  E[vocab, d]  ──► x (T, d)
for each of L layers:
    h = RMSNorm(x)
    x = x + W_o · MultiHeadCausalAttention(RoPE(W_q h), RoPE(W_k h), W_v h)
    h = RMSNorm(x)
    x = x + W_2 · GELU(W_1 h)            # W_1: d→4d, W_2: 4d→d
x = RMSNorm(x)
logits = x · Eᵀ                          # tied embeddings (output = input matrix)
```

Why these choices:
- **Pre-norm** (normalize *before* each sublayer) keeps the residual stream clean, so training is stable without careful warmup tuning. GPT-2 onwards and LLaMA all do this.
- **RMSNorm** instead of LayerNorm: it skips mean-centering, so it's cheaper and works just as well (LLaMA uses it). We still *implement and test both*, as the README promises, and the model uses RMSNorm.
- **RoPE** (rotary position embeddings) instead of learned positions: the position is encoded by rotating Q and K, so it has no parameters and works naturally with the KV-cache.
- **GELU MLP** (not SwiGLU): it's what the README lists and keeps the backward pass simpler. We use the **tanh approximation** and match it with `torch.nn.functional.gelu(x, approximate='tanh')`.
- **Tied embeddings**: one fewer big matrix to train, export, and quantize.
- **No biases** in linear layers (modern convention, simpler export). The `Linear` layer will still support a bias and test its gradient.

### 2.2 Tokenizer: byte-level (proposed)

Each UTF-8 byte is a token, so `vocab = 256`. There's no vocab file, the C++ side needs no tokenizer code (a byte *is* the token id), and any text works. A BPE tokenizer would be a separate project.

### 2.3 Dataset

**TinyShakespeare** (~1 MB). It's the standard small benchmark, trains fast, and the generated output is recognizable enough to show off. Split is 90% train / 10% validation, and we report **validation perplexity** = exp(mean cross-entropy).

### 2.4 Sizes

| Config | L | d | heads | context | params (non-emb) | Trained with |
|---|---|---|---|---|---|---|
| `tiny` | 2 | 64 | 4 | 64 | ~0.1M | Pure NumPy (proves the from-scratch stack learns) |
| `small` | 6 | 512 | 8 | 256 | ~18.9M | PyTorch (CPU or Colab); used for C++ benchmarks |

Parameter count per layer ≈ 12·d² (4d² for attention Q/K/V/O, 8d² for the MLP), so 12 · 512² · 6 ≈ 18.9M. That's **~75 MB in FP32 and ~19 MB in INT8**. Remember these numbers; Section 6 relies on them.

---

## 3. Stage 1: Pure NumPy with Manual Backprop (`python_scratch/`)

### 3.1 Design pattern

Every layer is a class with `forward(x)`, which caches what backward needs, and `backward(dout)`, which returns `dx` and stores `self.grads`. There is no autograd and no computation graph. The model's `backward` calls the layers' `backward` methods in reverse order. This mirrors how framework-free training code is actually written.

```python
class Linear:
    def forward(self, x):            # x: (..., in)
        self.x = x
        return x @ self.W + self.b
    def backward(self, dy):          # dy: (..., out)
        x2, dy2 = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.dW = x2.T @ dy2
        self.db = dy2.sum(0)
        return dy @ self.W.T
```

### 3.2 The derivations you need to own

Notation: $g$ = upstream gradient, all ops are per-row (over the feature dim $d$) unless stated otherwise.

**Linear** $y = xW + b$:
$\;\partial x = g W^\top,\quad \partial W = x^\top g,\quad \partial b = \textstyle\sum_{\text{rows}} g$

**LayerNorm** $\hat x = (x-\mu)/\sigma,\; y = \gamma \hat x + \beta$, with $\sigma=\sqrt{\text{var}+\epsilon}$. Let $\hat g = g\odot\gamma$:
$$\partial x = \frac{1}{\sigma}\Big(\hat g - \text{mean}(\hat g) - \hat x \cdot \text{mean}(\hat g \odot \hat x)\Big),\quad \partial\gamma = \textstyle\sum g\odot\hat x,\quad \partial\beta=\textstyle\sum g$$
*Intuition:* the two subtracted terms remove the gradient components along the directions normalization takes away, namely shifting the mean and scaling the variance.

**RMSNorm** $r=\sqrt{\text{mean}(x^2)+\epsilon},\; \hat x = x/r,\; y=\gamma\hat x$. Let $\hat g = g\odot\gamma$:
$$\partial x = \frac{1}{r}\Big(\hat g - \hat x\cdot\text{mean}(\hat g\odot\hat x)\Big)$$
It's the LayerNorm result without the mean term. That's literally the only difference between the two.

**Softmax** $s=\text{softmax}(z)$. The Jacobian is $J = \text{diag}(s) - ss^\top$, but we never build it. Instead:
$$\partial z = s\odot\big(g - \langle g, s\rangle\big)$$
That's O(d) instead of O(d²). Forward always subtracts $\max(z)$ first for numerical stability.

**Softmax + Cross-Entropy fused** (over $N$ tokens, targets $t$): $\;\partial z = (s - \text{onehot}(t))/N$. Computing the two separately would be unstable and wasteful. We use log-sum-exp in the forward pass.

**GELU (tanh approx)** $y = \tfrac12 x(1+\tanh u),\; u=\sqrt{2/\pi}\,(x+0.044715x^3)$:
$$\partial x = g\cdot\Big[\tfrac12(1+\tanh u) + \tfrac12 x(1-\tanh^2 u)\sqrt{2/\pi}\,(1+3\cdot0.044715x^2)\Big]$$

**Scaled dot-product causal attention**, per head, with $S = QK^\top/\sqrt{d_h} + M$ ($M$ = $-\infty$ above the diagonal), $P=\text{softmax}(S)$, $O = PV$:
$$\partial V = P^\top \partial O,\quad \partial P = \partial O\, V^\top,\quad \partial S = P\odot\big(\partial P - \text{rowsum}(\partial P\odot P)\big)$$
$$\partial Q = \partial S\,K/\sqrt{d_h},\quad \partial K = \partial S^\top Q/\sqrt{d_h}$$
Masked positions have $P=0$, so they automatically get zero gradient. Multi-head is just reshaping $(T, d) \to (h, T, d_h)$ and back, and its backward is the inverse reshape.

**RoPE.** For each pair of dims $(x_{2i}, x_{2i+1})$ at position $p$, rotate by angle $\theta_{p,i} = p\cdot 10000^{-2i/d_h}$:
$$\begin{pmatrix}x'_{2i}\\x'_{2i+1}\end{pmatrix}=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}\begin{pmatrix}x_{2i}\\x_{2i+1}\end{pmatrix}$$
A rotation matrix is orthogonal, so **the backward pass is rotation by $-\theta$**. Why RoPE encodes *relative* position: $\langle R_p q, R_{p'} k\rangle = \langle q, R_{p'-p}k\rangle$.

**Residual** $y = x + f(x)$: $\partial x = g + \partial f$. Gradients simply add, which is why deep residual nets train.

**Embedding**: forward is a row gather. Backward is a scatter-add (`np.add.at`), because repeated tokens must accumulate gradient. With tied weights, the embedding gradient = scatter-add from the input side + $g_{\text{logits}}^\top x$ from the output side.

### 3.3 Optimizers and scheduler (`optimizers.py`)

- **SGD + momentum:** $v \leftarrow \mu v + g,\; \theta \leftarrow \theta - \eta v$
- **AdamW:** $m \leftarrow \beta_1 m + (1-\beta_1)g$, $v\leftarrow\beta_2 v+(1-\beta_2)g^2$, $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$, then
  $\theta \leftarrow \theta - \eta\big(\hat m/(\sqrt{\hat v}+\epsilon) + \lambda\theta\big)$.
  *Decoupled* means the weight decay $\lambda\theta$ is applied directly and is **not** added to $g$. If it were, it would get divided by $\sqrt{\hat v}$, and parameters with large gradients would barely decay. That's the bug in "Adam + L2" that AdamW fixed. We apply decay only to matrices, not to norm gains.
- **Cosine with linear warmup:** $\eta_t = \eta_{\max}\cdot t/T_w$ for $t<T_w$, otherwise $\eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})(1+\cos(\pi\frac{t-T_w}{T-T_w}))$.
- **Gradient clipping** by global norm (essential for transformer stability).

All three get tested step by step against `torch.optim.SGD/AdamW` and `LambdaLR`.

### 3.4 Verification (`test_gradients.py`, pytest)

For each layer and for the full model: generate random inputs in **float64**, run our forward/backward, run the same thing in PyTorch with `requires_grad=True`, and compare outputs and every gradient. The `max_diff < 1e-6` claim only holds in float64; in float32, rounding alone is ~1e-6. A finite-difference check $(f(x+h)-f(x-h))/2h$ on a few elements serves as a second, framework-independent check.

### 3.5 `transformer.py`: two implementations, one architecture

- `NumpyGPT` is built from the layers above and trains the `tiny` config with our AdamW. This proves the from-scratch stack actually learns: the loss falls and samples look like Shakespeare-ish gibberish.
- `TorchGPT` is the same architecture in `torch.nn`, with **identical parameter names and layouts**, used to train `small`. A test loads the same weights into both and checks that the logits match. That test ties the two implementations together.

---

## 4. Export Format (`export_weights.py`)

The C++ side shouldn't parse pickles or depend on anything, so we use a flat binary file:

```
Header (64 bytes):  magic "BMTF" | version u32 | dtype u32 (0=FP32, 1=INT8)
                    vocab, d_model, n_layers, n_heads, d_ff, max_seq u32 | rope_theta f32 | pad
Tensors, fixed order (emb, then per layer: attn_norm, wq, wk, wv, wo, mlp_norm, w1, w2; final_norm),
each starting at a 64-byte-aligned offset:
    FP32 tensor:  float32[rows*cols]
    INT8 tensor:  float32 scales[rows]  then  int8[rows*cols]     (per-output-channel)
Norm gains always stay FP32 (tiny and precision-sensitive).
```

Weights are stored **row-major as [out_features, in_features]**, so each output channel's weights are contiguous. This matters for the GEMV kernel (Section 6) and makes per-channel scales natural.

The exporter also writes `reference.bin`: a fixed prompt plus the PyTorch logits for it. The C++ test suite compares its own output against that file.

### INT8 symmetric per-channel quantization

For each output row $w$: $\;s = \max|w|/127,\;\; q = \text{round}(w/s) \in [-127,127]$, and the dequantized value is $\hat w = s\,q$.
- **Symmetric** means there's no zero-point, so the integer dot product needs no correction terms.
- **Per-channel** (one scale per output row), not per-tensor, because one outlier row would otherwise wreck the resolution of every other row.
- The max rounding error per weight is $s/2$. We'll measure the effect as **perplexity delta** (FP32 vs. INT8 on the validation set), computed in PyTorch with fake quantization *and* in C++. The two should agree.
- Optional extra: group-wise scales (e.g., one scale per 64 weights). This is the README's "block-wise" and what llama.cpp's Q8_0 does.

---

## 5. Stage 2: The C++17 Engine (`cpp_engine/`)

### 5.1 Memory: arena plus tensor views (`tensor.hpp`)

- **Arena:** one big `std::aligned_alloc(64, bytes)` at startup holds weights, activations, and KV-cache. There are no allocations in the generation loop. Alignment to 64 bytes means a cache line never splits a SIMD load, and AVX2 aligned loads work.
- **Tensor:** a *non-owning view* `{float* data; int rows, cols, stride;}`, similar to a tiny `std::span` with a shape. Cheap to copy and trivially sliceable (e.g., one head's slice of Q).
- Weights are loaded with **`mmap`** on Linux: the OS pages them in lazily, and load time ≈ 0. This falls back to `fread` elsewhere.

### 5.2 Forward pass for one token (`transformer.cpp`)

Generation processes **one new token at a time**. For position $p$:

```
x = emb[token]                                   # (d,)
for l in layers:
    h = rmsnorm(x, g_attn[l])
    q, k, v = Wq h, Wk h, Wv h                   # three GEMVs (matrix × vector)
    rope(q, p); rope(k, p)
    K_cache[l][p] = k;  V_cache[l][p] = v        # append to KV-cache
    for each head:  scores[0..p] = q_h · K_cache[l][0..p]_h / √d_h
                    softmax(scores);  out_h = Σ scores[i] · V_cache[l][i]_h
    x += Wo · out
    h = rmsnorm(x, g_mlp[l]);  x += W2 · gelu(W1 h)
logits = emb · rmsnorm(x, g_final)               # (vocab,)
next = sample(logits)
```

**The KV-cache.** K and V for past positions never change, so we store them instead of recomputing. Without it, generating token $p$ costs a full forward pass over $p$ tokens, which makes generating $T$ tokens O(T²) layer passes. With it, each step is one token's worth of matmul plus an O(p) attention dot product. Cache size = 2 · L · max_seq · d floats = 2·6·256·512·4 B ≈ **6 MB** for `small`. An optional extra is an INT8 KV-cache (the README mentions it).

**Prefill vs. decode.** The prompt can be processed all at once (a real **GEMM**, compute-bound), and each generated token is a **GEMV** (memory-bound). The benchmark reports both, because they stress the hardware in completely different ways. That contrast is one of the most important insights of the project.

**Sampler:** greedy, temperature, top-k, and top-p, with a seeded xorshift RNG so runs are reproducible.

---

## 6. The Hardware Story: Why Each Kernel Is Faster

This is the core of the project. **Roofline thinking:** every kernel is limited either by compute (FLOP/s) or by memory bandwidth (bytes/s). *Arithmetic intensity* = FLOPs per byte moved decides which.

**Your CPU's ceilings (approximate; we'll measure them):**
- Compute: 8 cores × 2 FMA units × 8 floats × 2 FLOP ≈ 32 FLOP/cycle/core. At an all-core ~3 GHz that's **~770 GFLOPS FP32 peak** (one core ≈ 100–140 GFLOPS at boost).
- DRAM bandwidth: roughly 30–50 GB/s (LPDDR4x). We'll measure it with a simple STREAM-like copy loop.
- Caches: L1d 32 KB / L2 512 KB per core, L3 16 MB shared.

**GEMM** C[M×N] = A[M×K]·B[K×N] does 2MNK FLOPs on 4(MK+KN+MN) bytes. At 1024³ that's 2.1 GFLOP on 12 MB, so intensity is high and it's **compute-bound if we reuse data from cache**. Getting that reuse is the whole game.

### Kernel 0: naive i-j-k
```cpp
for i: for j: { float s=0; for k: s += A[i*K+k]*B[k*N+j]; C[i*N+j]=s; }
```
The inner loop walks B **down a column** with stride N·4 bytes, so every access hits a new cache line, and only 4 of its 64 bytes get used. The compiler can't vectorize it well either. Expect ~1–2 GFLOPS.

### Kernel 1: loop reorder (i-k-j) plus cache tiling
```cpp
for i: for k: { float a=A[i*K+k]; for j: C[i*N+j] += a*B[k*N+j]; }   // inner loop contiguous
```
Now the inner loop streams contiguous rows of B and C, which uses whole cache lines and auto-vectorizes. **Tiling** then splits the loops into blocks, so a block of B (e.g. 64×64 floats = 16 KB) stays in L1/L2 and is reused by many rows of A before it gets evicted. Without tiling, B (4 MB) is streamed from L3/DRAM again for every row of A. We'll sweep tile sizes and plot the result, which is a nice graph for the README.

### Kernel 2: SIMD micro-kernel, packing, OpenMP (the BLIS/GotoBLAS approach)
This is how real BLAS libraries are built, scaled down to something you can explain line by line:

1. **Register-blocked micro-kernel (6×16).** AVX2 has 16 `ymm` registers of 8 floats. We keep a 6×16 tile of C in **12 accumulator registers** for the entire K loop. Each step loads 2 vectors of B (16 floats), broadcasts 6 scalars of A, and issues 12 `_mm256_fmadd_ps`. That's 12 FMAs per 8 loads: enough independent FMAs to cover the 4–5 cycle FMA latency on both FMA ports. Accumulators only go back to memory once at the end. **This is where most of the speedup comes from.**
2. **Packing.** Before the micro-kernel runs, the A and B blocks are copied into contiguous "panels" laid out in exactly the order the micro-kernel reads them. Every load then hits sequential, aligned memory with no TLB misses, and the cost is amortized over the reuse.
3. **Three cache levels of blocking** (KC, MC, NC). The B panel is sized for L1 (KC×16 floats), the A block for L2 (MC×KC), and the B block for L3 (KC×NC). Values are tuned for Zen 3, roughly KC=256, MC=72, NC=4080, and we'll sweep them.
4. **OpenMP** `#pragma omp parallel for` over the MC/NC blocks, so each core gets its own C tiles and there's no false sharing.
5. **ARM NEON** version: same structure with 128-bit `float32x4_t` and `vfmaq_f32`. AArch64 has 32 vector registers, so an 8×12 micro-kernel fits. The code is selected at compile time with `#if defined(__AVX2__) / __ARM_NEON` and has a scalar fallback. **Testing it:** your machine is x86, so we'll cross-compile with `aarch64-linux-gnu-g++` and run the tests under `qemu-aarch64` (correctness only), and/or use a free GitHub Actions ARM runner. Benchmarks on real ARM hardware (e.g., a friend's Mac/Raspberry Pi) are optional.

Target: **~50%+ of peak** on 8 cores, a few hundred GFLOPS. A comparison against OpenBLAS or NumPy's GEMM on the same machine gives an honest reference point.

### Kernel 3: INT8
Two variants, both per-channel symmetric:
- **W8A32 GEMV (the decode path).** This is the important one. Decode is a matrix × vector product: 2 FLOPs per weight, and each weight is read once. With 2 FLOPs per 4 bytes in FP32, **memory bandwidth is the limit, not compute.** Per token we must read all ~75 MB of FP32 weights. At ~40 GB/s that's a **floor of ~1.9 ms/token** no matter how good the SIMD is. INT8 weights are 19 MB, so the floor drops to ~0.5 ms, and 19 MB nearly fits in the 16 MB L3. Kernel: load 32 int8 weights → `_mm256_cvtepi8_epi32` (8 at a time) → `_mm256_cvtepi32_ps` → FMA with the FP32 activation, then multiply the row sum by its scale once at the end.
- **W8A8 GEMM (prefill/compute-bound).** Quantize the activations dynamically, per token row, then do integer dot products: sign-extend int8→int16 (`_mm256_cvtepi8_epi16`), `_mm256_madd_epi16` (16 multiplies + pairwise add into int32), accumulate in int32, and rescale at the end by $s_a \cdot s_w$. Without VNNI on Zen 3 this is roughly 1.5–2× FP32 throughput in the best case. We'll measure whether it actually is.

We'll also show an **FP32 GEMV** optimized the same way, so the INT8 speedup is measured against a strong baseline, not a weak one.

### What the benchmark suite reports (`benchmark.cpp`)
- GEMM: M=N=K ∈ {128, 256, 512, 1024, 2048} for every kernel: median of N runs after warmup, GFLOPS = 2MNK/t, % of measured peak, plus a correctness check against Kernel 0 for every result.
- GEMV: GB/s achieved vs. measured DRAM bandwidth.
- End-to-end: prefill tokens/s and decode ms/token for FP32 vs. INT8, with thread scaling (1, 2, 4, 8, 16 threads).
- Memory footprint and INT8 perplexity delta.
- Output: a table plus CSV. A small Python script turns the CSV into plots for the README.

---

## 7. Correctness Chain (How We Know Everything Is Right)

```
NumPy layer grads ══ PyTorch autograd          (float64, < 1e-6, pytest)
NumpyGPT logits   ══ TorchGPT logits           (same weights)
C++ FP32 logits   ══ TorchGPT logits           (reference.bin, < 1e-4 rel, ctest)
every fast kernel ══ naive kernel              (randomized shapes incl. non-multiples of tile size)
C++ INT8 ppl      ≈  PyTorch fake-quant ppl    (validation set)
```

Edge cases the kernel tests must cover: sizes that aren't multiples of 6/16/tile size (remainder handling is where bugs hide), M=1 (GEMV), and unaligned leading dimensions. The build also runs the tests under `-fsanitize=address,undefined` in Debug.

---

## 8. Final Repository Layout

This is the README's layout plus the pieces needed to make it real (tests, a Colab notebook, plots):

```
BareMetal-Transformer/
├── python_scratch/
│   ├── nn_layers.py  attention.py  optimizers.py  transformer.py
│   ├── data.py                  # TinyShakespeare download + byte tokenization
│   ├── train_numpy.py           # trains `tiny` with the pure-NumPy stack
│   ├── train_torch.py           # trains `small` (CPU or GPU)
│   ├── export_weights.py        # FP32 + INT8 .bin + reference.bin
│   ├── eval_quant.py            # FP32 vs INT8 perplexity
│   └── test_gradients.py  test_optimizers.py  test_model_parity.py
├── cpp_engine/
│   ├── include/  tensor.hpp  kernels.hpp  quantize.hpp  transformer.hpp  model_io.hpp
│   ├── src/      kernels_naive.cpp  kernels_tiled.cpp  kernels_simd.cpp  quantize.cpp
│   │             transformer.cpp  model_io.cpp  generate.cpp  benchmark.cpp
│   ├── tests/    test_kernels.cpp  test_reference.cpp      # tiny assert harness, no gtest
│   └── CMakeLists.txt
├── notebooks/colab_train.ipynb  # clone repo → train small → download checkpoint
├── scripts/  run_all_benchmarks.sh  plot_benchmarks.py
├── docs/     EXPLANATION.md (this file) + benchmark plots
└── README.md                    # rewritten with real, measured numbers
```

Executables: `generate` (prompt → text, with `--temp/--top-k/--threads`) and `benchmark_transformer`.

---

## 9. Build Plan (Phases)

Each phase ends with something runnable and a commit, so the history shows the project actually developing step by step.

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | NumPy layers + gradient tests | all layer gradchecks pass vs. PyTorch |
| 2 | Attention + RoPE + full NumpyGPT, optimizers + tests | model gradcheck passes; `tiny` loss falls on Shakespeare |
| 3 | TorchGPT + parity test, training script, Colab notebook | logits match NumpyGPT; `small` trained, val ppl recorded |
| 4 | Exporter (FP32/INT8) + quant eval | INT8 ppl delta measured in PyTorch |
| 5 | C++ arena, loader, naive kernels, FP32 forward + KV-cache, sampler | C++ logits match `reference.bin`; `generate` produces text |
| 6 | Tiled → SIMD/packing/OpenMP GEMM + optimized GEMV | all kernels pass tests; GFLOPS table |
| 7 | INT8 GEMV + W8A8 GEMM, INT8 model path | INT8 logits ≈ fake-quant; ms/token table |
| 8 | NEON path (cross-compile + qemu tests), CI | ARM tests pass |
| 9 | Benchmarks, plots, README rewrite with real numbers | reproducible via `run_all_benchmarks.sh` |

**On it looking like your work:** the code will follow one consistent, plain style: clear names, comments only where something is non-obvious (e.g., *why* the micro-kernel is 6×16), and no boilerplate. More important than how it looks: after each phase you should read the code against this document until you can rebuild the derivation or explain the kernel yourself. That's what holds up in an interview.

---

## 10. Decisions to Confirm Before I Start

1. **Tokenizer:** byte-level (vocab 256), proposed above. Or would you prefer char-level (vocab ~65)?
2. **Dataset:** TinyShakespeare OK?
3. **Model sizes:** `tiny` (NumPy) and `small` 6L/512d/8h (PyTorch, for benchmarks), as in the README?
4. **Training compute:** CPU is fine for everything, with Colab optional for `small`. Should I still build the Colab notebook?
5. **README numbers:** OK to replace all placeholder numbers with measured ones (and to reframe INT8's benefit around decode/GEMV, as explained in Section 1)?
6. **Python env:** create a `.venv` in the repo with `numpy`, `torch` (CPU build), `pytest`, `matplotlib`?
7. **Commits:** should I commit at the end of each phase (as you), or leave all git operations to you?

Once you've read this and confirmed (or changed) these points, I'll start Phase 1.

---

## 11. Extension: Low-Precision Edge Training and a Ternary Accelerator Model

This section was added after the nine phases. It asks a different question from the inference engine: *what does it take to **train** with 2-bit weights and INT8 everything else, and what would that save in silicon?* Code: `python_scratch/low_bit_training.py`, `hardware_profiler.py`, `lowbit_native.py`; `cpp_engine/{include/ternary.hpp, src/kernels_ternary_accum.cpp, src/ternary.cpp, src/lowbit_train.cpp, src/lowbit_capi.cpp, src/benchmark_lowbit.cpp}`; tests in `tests/test_lowbit_training.py` and `cpp_engine/tests/test_ternary.cpp`.

### 11.1 Why training is harder than inference

Inference quantisation freezes the weights. Training breaks three things:

1. **The quantiser has zero gradient almost everywhere** (round() is a staircase). Fix: the **straight-through estimator**: forward with $\hat W = Q(W)$, backward as if $Q$ were the identity, $\partial L/\partial W := \partial L/\partial \hat W$. The gradient is applied to a latent higher-precision copy $W$ (the *master weights*), and only the forward pass sees $\hat W$.
2. **Updates are tiny, grids are coarse.** An update $\Delta w$ smaller than half a grid step rounds back to the same value every time: training silently stalls. Fix: **stochastic rounding** (below).
3. **Activations must be remembered.** Backprop needs every layer's forward input, so activation memory grows with batch × depth and dominates training RAM. Fix: **cache them as INT8**, dequantise during backward.

### 11.2 Ternary weights (BitNet b1.58 "absmean")

$\gamma = \tfrac1n\sum|W_{ij}|,\qquad \hat W = \operatorname{clip}(\operatorname{round}(W/\gamma),-1,1)\in\{-1,0,+1\}$, used as $y \approx \gamma\,(\hat W x)$. Three values need $\log_2 3 \approx 1.58$ bits; we store 2: four weights per byte (`w0 | w1<<2 | w2<<4 | w3<<6`, codes `00=0, 01=+1, 10=-1`), 16× smaller than FP32.

### 11.3 Why the matmul needs no multiplier

With $w\in\{-1,0,+1\}$: $\;w\cdot x \;=\; \{+x,\,-x,\,0\}$. A dot product becomes "add some activations, subtract others, skip the zeros". On an FPGA that is an adder/subtractor in a LUT instead of a DSP multiplier. In the AVX2 kernel each 32-weight step does: expand the 8 packed bytes to two byte masks (`pshufb`, `and`, `cmpeq`), then `d = (x & pos) − (x & neg)`, then add the 32 bytes with `psadbw` (after a `^0x80` bias so they are unsigned). No `pmaddubsw`, no `fmadd`. `scripts/check_multiplierless.sh` disassembles `kernels_ternary_accum.cpp.o` in CTest and fails if any multiplier-class instruction appears; a negative control proves the check can fail. After the integer sum there is **one** rescale multiply per *output* ($\gamma\cdot s_x$), O(N) instead of O(NK).

### 11.4 Stochastic rounding is unbiased

$\operatorname{SR}(x)=\lfloor x+u\rfloor,\; u\sim U[0,1)$. Write $x=n+f$ with $n=\lfloor x\rfloor$, $f\in[0,1)$. Then $\operatorname{SR}(x)=n+1$ exactly when $u\ge 1-f$, probability $f$, else $n$. So $\mathbb E[\operatorname{SR}(x)] = n(1-f)+(n+1)f = n+f = x$. Round-to-nearest of $x=0.3$ is always 0, so a thousand updates of $0.3$ LSB add up to nothing; SR returns 1 with probability 0.3 and the sum is about 300. The cost is variance: $\operatorname{Var}=f(1-f)\le\tfrac14$ LSB². Whether the bias or the variance matters more is an empirical question, and the experiments show both outcomes (README).

### 11.5 Block floating point gradients

Each block of 32 gradient values shares a power-of-two exponent $e=\lceil\log_2(\max|g|/q_{\max})\rceil$ and stores signed mantissas $m=\operatorname{SR}(g/2^e)$ in 8 (or 4) bits. Dequantisation is a shift. Storage ≈ 1.03 bytes per gradient instead of 4. The exponent is computed with `frexp` so it is exact at powers of two (a naive `ceil(log2())` is off by one there; a test pins that).

### 11.6 INT8 activation caching

The INT8 tensor that the forward matmul already consumed (one scale per row) is simply kept instead of the FP32 input: caching is free. GELU, RMSNorm, and attention (q, k, v as INT8, the softmax matrix as UINT8) are compressed the same way. During backward, $dW = \hat x^\top\,dy$ uses $\hat x = q\cdot s$, dequantised on the fly, so one layer's FP32 copy exists at a time.

### 11.7 What the hardware profiler does and does not claim

`hardware_profiler.py` computes, for a given model/batch: (a) bytes crossing the DRAM boundary per training step by a tiling model with a finite SRAM; (b) resident memory by tensor class; (c) DSP/LUT counts for 16×16, 32×32, 64×64 PE arrays from per-PE assumptions listed in the code. These are **analytical models with stated parameters, not synthesis results**. The footprint model is validated against the real training loop (it predicts the measured activation cache to within 1.5%). Everything labelled *measured* in the README was measured on this machine by running the code.
