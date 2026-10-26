# BareMetal-Transformer: Neural Networks & Transformers from Scratch + C++17 SIMD/INT8 Inference Engine

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://isocpp.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Pure%20Autograd-013243.svg)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Reference-ee4c2c.svg)](https://pytorch.org/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20NEON-success.svg)]()

**BareMetal-Transformer** is a ground-up implementation of deep learning primitives and Transformer architectures built in two tightly coupled stages:
1. **Mathematical & Training Foundations (`python_scratch/`):** Pure **NumPy** (manual forward/backward analytical gradients, optimizers, and attention) alongside a reference **PyTorch** training pipeline.
2. **Hardware-Aware C++17 Inference Engine (`cpp_engine/`):** A zero-dependency **C++17** runtime executing Transformer forward passes using **L1/L2 cache-tiled matrix multiplication**, **explicit SIMD vectorization (AVX2 / ARM NEON)**, **OpenMP multi-threading**, and **per-channel INT8 weight quantization**.

---

## Key Highlights

- **Analytical Backpropagation in Pure NumPy:** Full manual forward and backward pass derivations for Linear projections, LayerNorm, RMSNorm, Softmax Jacobian, Cross-Entropy Loss, GELU, and Scaled Dot-Product Multi-Head Attention—verified against PyTorch `autograd` (`max_diff < 1e-6`).
- **From-Scratch Optimizers & Schedulers:** Custom implementations of `SGD + Momentum`, `AdamW` with decoupled weight decay, and Cosine Annealing with Linear Warmup.
- **Zero-Dependency C++17 Transformer Runtime:** Exports trained checkpoint weights into a compact binary format and executes autoregressive token generation in pure C++17 without external BLAS libraries.
- **Progressive Hardware Kernel Optimization:** Step-by-step GEMM and attention optimization demonstrating the impact of memory hierarchy, cache locality, instruction-level parallelism (SIMD), and low-bit quantization:
  1. `Kernel 0 (Naive FP32)`: Standard $O(N^3)$ triple-loop baseline.
  2. `Kernel 1 (Cache-Blocked + Loop-Reordered)`: $64 \times 64$ tile blocking to eliminate L1/L2 cache thrashing.
  3. `Kernel 2 (SIMD Vectorized + OpenMP)`: Explicit 256-bit `AVX2` (`_mm256_fmadd_ps`) and 128-bit `ARM NEON` (`vfmaq_f32`) intrinsics with register accumulation.
  4. `Kernel 3 (INT8 Symmetric Quantization)`: $4\times$ model memory footprint compression via block/channel-wise `INT8` weight quantization and fast integer dot-product accumulation.
- **Low-Precision Edge Training & Ternary Accelerator Co-Design:** 2-bit `{-1,0,+1}` weights, a multiplier-less C++ GEMM (add/subtract only, verified on the machine code), INT8 activation caching, and block-floating-point gradients with stochastic rounding, plus an analytical PE-array / DRAM-bandwidth profiler. [Jump to the section](#low-precision-edge-training--ternary-accelerator-co-design).

---

## Repository Architecture

```text
BareMetal-Transformer/
├── python_scratch/
│   ├── nn_layers.py            # Pure NumPy Linear, LayerNorm, RMSNorm, GELU, Softmax (fwd + bwd)
│   ├── attention.py            # Multi-Head Causal Self-Attention & RoPE in NumPy/PyTorch
│   ├── optimizers.py           # From-scratch SGD, AdamW, and Cosine LR Scheduler
│   ├── transformer.py          # Complete Decoder-Only Transformer training pipeline
│   ├── export_weights.py       # Serializes FP32 & calibrated INT8 weights to binary format
│   ├── test_gradients.py       # Unit tests verifying NumPy gradients vs. PyTorch Autograd
│   ├── low_bit_training.py     # Ternary/INT8 STE training, INT8 activation cache, BFP gradients + stochastic rounding
│   ├── lowbit_native.py        # ctypes bridge to the C++ low-bit kernels (cross-checks the NumPy reference)
│   └── hardware_profiler.py    # PE-array (DSP/LUT) and DRAM-traffic model + measured convergence/memory report
├── tests/
│   └── test_lowbit_training.py # Bit-packing, stochastic-rounding bias, INT8 cache, convergence vs FP32
├── cpp_engine/
│   ├── include/
│   │   ├── tensor.hpp          # Contiguous 64-byte aligned memory arena & Tensor views
│   │   ├── kernels.hpp         # Naive, Cache-Tiled, SIMD (AVX2/NEON), & INT8 GEMM headers
│   │   ├── transformer.hpp     # C++17 Transformer block, KV-Cache, & token sampler
│   │   ├── ternary.hpp         # 2-bit packed ternary weights + multiplier-less GEMM interface
│   │   └── lowbit_train.hpp    # Stochastic rounding, block-floating-point gradient quantisation
│   ├── src/
│   │   ├── kernels_simd.cpp    # Hardware-specific SIMD intrinsics & cache-tiling routines
│   │   ├── quantize.cpp        # Symmetric INT8 quantization & dequantization routines
│   │   ├── transformer.cpp     # End-to-end C++ forward pass & KV-Cache management
│   │   ├── benchmark.cpp       # CLI microbenchmark suite (GFLOPS, ms/token, memory)
│   │   ├── kernels_ternary_accum.cpp  # Add/subtract-only ternary accumulate kernel (AVX2 / NEON)
│   │   ├── benchmark_lowbit.cpp       # FP32 vs INT8 vs ternary kernel timings
│   │   └── lowbit_capi.cpp            # C API -> libbmlowbit.so for Python
│   └── CMakeLists.txt          # Cross-platform build config (-O3 -march=native -fopenmp)
└── scripts/
    ├── run_all_benchmarks.sh   # Runs gradient tests, exports weights, & benchmarks C++ kernels
    └── check_multiplierless.sh # Disassembles a kernel and fails if it contains a data-path multiplier
```

## Performance Benchmarks (C++17 Engine vs. Naive Baseline)

### 1. Matrix Multiplication (GEMM 1024×1024×1024)
| Kernel Implementation | Precision | Execution Time (ms) | Throughput (GFLOPS) | Relative Speedup | Memory Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Naive 3-Loop (i-j-k) | FP32 | 1,420.5 ms | 1.51 GFLOPS | 1.00x | 4.00 MB |
| 2. Loop-Reordered + Cache-Tiled (64×64) | FP32 | 168.4 ms | 12.75 GFLOPS | 8.43x | 4.00 MB |
| 3. SIMD Vectorized (AVX2 / NEON) + OpenMP | FP32 | 24.6 ms | 87.29 GFLOPS | 57.74x | 4.00 MB |
| 4. Quantized SIMD Dot-Product (W8A32) | INT8 | 15.8 ms | 135.91 GFLOPS (eff.) | 89.90x | 1.00 MB (4x smaller) |


2. End-to-End Transformer Forward Pass (6 Layers, dmodel​=512, nheads​=8)

 - Pure NumPy Baseline: 48.2 ms / token (20.7 tokens/sec)
 - C++17 Cache-Tiled + SIMD Engine (FP32): 6.4 ms / token (156.2 tokens/sec)
 - C++17 Quantized Engine (INT8 Weights + KV-Cache): 3.9 ms / token (256.4 tokens/sec, <0.4% perplexity delta)

## Low-Precision Edge Training & Ternary Accelerator Co-Design

An extension that asks: *what does it take to **train**, not just run, a transformer with 2-bit weights and INT8 everything else, and what would that save in an accelerator?* It combines a C++17 multiplier-less kernel, a NumPy training loop with the memory tricks edge training needs, and a hardware model of processing-element arrays and DRAM traffic. Every number below was produced by running the code in this repository (`python python_scratch/hardware_profiler.py`); numbers marked **analytical** come from an explicit model with stated parameters, not from synthesis or a board.

### What was built

- **2-bit packed ternary weights** (`ternary.hpp`): BitNet *absmean* quantisation, four weights per byte (`w0 | w1<<2 | w2<<4 | w3<<6`), 16x smaller than FP32.
- **Multiplier-less GEMM** (`kernels_ternary_accum.cpp`, AVX2 and NEON, OpenMP): the multiply-accumulate is replaced by `acc += x` / `acc -= x` selected with byte masks; one rescale multiply per *output*, none per weight. CTest disassembles the compiled kernel and fails on any multiplier-class instruction (x86 and AArch64); a negative control proves the check can fail.
- **Low-precision training** (`low_bit_training.py`): straight-through estimator for ternary/INT8 weights; every cached activation (linear inputs, GELU, RMSNorm, q/k/v, softmax matrix) held as INT8 and dequantised on the fly in the backward pass; weight gradients stored as INT8/INT4 **block floating point** (one power-of-two exponent per 32 values) with **unbiased stochastic rounding**; optional INT8 master weights.
- **Hardware profiler** (`hardware_profiler.py`): per-step DRAM traffic through a tiled-GEMM model with finite SRAM, resident-memory footprint, and DSP/LUT cost of 16x16, 32x32 and 64x64 PE arrays.

### Results

<!-- LOWBIT-RESULTS:BEGIN -->

#### Summary

`small` model (19M parameters), batch 4 x 256 tokens for the memory and bandwidth columns; `tiny` model (115k parameters), 200 AdamW steps, 3 seeds for the loss columns. Details, assumptions and the measured-vs-analytical split follow.

| Configuration | Training memory, MB (analytical) | Peak RAM, MB (measured) | Activation cache, MB (measured) | Off-chip traffic, MB/step | Off-chip bandwidth saved | Multipliers / DSP (32x32 PE array) | Val loss, 200 steps | vs FP32 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FP32 training | 611 | 428 | 307 | 2,381 | - | 1,024 / 3,072 | 2.287 | - |
| INT8 training (FP32 master) | 336 (1.82x) | 232 | 69 (4.4x) | 983 | 59% | 1,024 / 1,024 | 2.288 | +0.000 |
| Ternary + INT8 act (FP32 master) | 322 (1.90x) | 228 | 69 (4.4x) | 941 | 60% | 0 / 0 | 2.404 | +0.117 |
| Ternary + INT8 act (INT8 master) | 265 (2.31x) | 228 | 69 (4.4x) | 827 | 65% | 0 / 0 | 2.403 | +0.116 |

#### Off-chip memory traffic per training step (analytical)

`small` model (6 layers, d=512, 8 heads, ~19M parameters), batch 4 x context 256 tokens, 32x32 PE array, 2 MiB on-chip SRAM, ideal-streaming DRAM model. MB per step.

| Configuration | Weights | Activations | Gradients | Optimizer state | Total | Bandwidth saved |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| FP32 training | 555.8 | 1,368.6 | 152.1 | 304.2 | **2,380.7** | - |
| INT8 training (FP32 master) | 209.8 | 429.1 | 40.0 | 304.2 | **983.1** | 59% |
| Ternary + INT8 act (FP32 master) | 167.3 | 429.1 | 40.0 | 304.2 | **940.6** | 60% |
| Ternary + INT8 act (INT8 master) | 54.1 | 429.1 | 40.0 | 304.2 | **827.4** | 65% |

**Sensitivity to the PE array size and SRAM budget** (total MB per step; the bigger the buffer, the less operands are re-fetched):

| PE array | SRAM | FP32 training | INT8 training (FP32 master) | Ternary + INT8 act (FP32 master) | Ternary + INT8 act (INT8 master) | Ternary saving |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| 16x16 | 512 KiB | 3,998.6 | 1,091.3 | 985.9 | 872.6 | 75% |
| 16x16 | 2048 KiB | 2,380.7 | 983.1 | 940.6 | 827.4 | 60% |
| 16x16 | 8192 KiB | 1,978.0 | 983.1 | 940.6 | 827.4 | 52% |
| 32x32 | 512 KiB | 4,652.9 | 1,091.3 | 989.0 | 875.8 | 79% |
| 32x32 | 2048 KiB | 2,380.7 | 983.1 | 940.6 | 827.4 | 60% |
| 32x32 | 8192 KiB | 1,978.0 | 983.1 | 940.6 | 827.4 | 52% |
| 64x64 | 512 KiB | 4,202.0 | 1,093.5 | 991.1 | 877.9 | 76% |
| 64x64 | 2048 KiB | 2,380.7 | 983.1 | 940.6 | 827.4 | 60% |
| 64x64 | 8192 KiB | 1,978.0 | 983.1 | 940.6 | 827.4 | 52% |

#### Training memory footprint

Resident bytes during a step at the same shape (MB). *Analytical* = from the tensor inventory above; *measured* = bytes actually held by this repository's NumPy training code (NumPy integer GEMMs) for one real forward+backward.

| Configuration (analytical) | Master weights | Compute weight copy | Weight gradients | Optimizer state | Cached activations | Total | Reduction |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FP32 training | 76.0 | 0.0 | 76.0 | 152.1 | 307.2 | **611.4** | - |
| INT8 training (FP32 master) | 76.0 | 18.9 | 20.0 | 152.1 | 68.9 | **336.0** | 1.82x |
| Ternary + INT8 act (FP32 master) | 76.0 | 4.7 | 20.0 | 152.1 | 68.9 | **321.8** | 1.90x |
| Ternary + INT8 act (INT8 master) | 19.4 | 4.7 | 20.0 | 152.1 | 68.9 | **265.2** | 2.31x |

**Measured on the real `small` model** (one forward + backward):

| Configuration | Activation cache held after forward | (analytical) | Cache reduction | Peak RAM during step (tracemalloc) | Step time |
| --- | :---: | :---: | :---: | :---: | :---: |
| FP32 training | 307.3 | 307.2 | - | 427.9 | 1.05 s |
| INT8 training (FP32 master) | 69.2 | 68.9 | 4.44x | 232.4 | 2.12 s |
| Ternary + INT8 act (FP32 master) | 69.2 | 68.9 | 4.44x | 228.1 | 1.93 s |
| Ternary + INT8 act (INT8 master) | 69.2 | 68.9 | 4.44x | 228.1 | 1.99 s |

#### Processing-element array cost (analytical)

Multipliers (DSP slices) versus add/subtract units for the weight x activation matmul engine. Per-PE assumptions: FP32 MAC = 3 DSP + 420 LUT; INT8 MAC = 1 DSP + 10 LUT; ternary PE = 0 DSP + 32 LUT (sign gate + 24-bit add/sub). Estimates for planning, not synthesis results.

| Array | FP32 PE array | INT8 PE array | Ternary PE array | DSP reduction vs FP32 | Peak GMAC/s @ 300 MHz |
| --- | :---: | :---: | :---: | :---: | :---: |
| 16x16 (256) | 256 FP32 mul / 768 DSP / 107,520 LUT | 256 INT8 mul / 256 DSP / 2,560 LUT | **0 mul / 0 DSP** / 8,192 LUT (256 add/sub) | 768 -> 0 | 77 |
| 32x32 (1024) | 1,024 FP32 mul / 3,072 DSP / 430,080 LUT | 1,024 INT8 mul / 1,024 DSP / 10,240 LUT | **0 mul / 0 DSP** / 32,768 LUT (1,024 add/sub) | 3,072 -> 0 | 307 |
| 64x64 (4096) | 4,096 FP32 mul / 12,288 DSP / 1,720,320 LUT | 4,096 INT8 mul / 4,096 DSP / 40,960 LUT | **0 mul / 0 DSP** / 131,072 LUT (4,096 add/sub) | 12,288 -> 0 | 1229 |

Scope: in the `small` model 95% of the per-token MACs are weight x activation products (the ternary-capable linear layers), 4% are the activation x activation attention products (QK^T, PV), and 1% are the FP32 output head. Attention and the head still need real multipliers; only the linear layers become multiplier-less.


#### Convergence and execution speed: tiny transformer, 200 steps (measured)

`tiny` model (115,008 parameters), TinyShakespeare bytes, batch 16 x 64, AdamW, cosine schedule, 3 seed(s); mean (std) across seeds. Validation loss is measured on the quantised network itself.

| Configuration | Train loss (last 20) | Val loss | vs FP32 | Val perplexity | Grad cosine vs FP32 | ms / step | Time vs FP32 | Cache MB | Peak RAM MB |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FP32 training | 2.279 (0.002) | **2.287** (0.002) | +0.000 | 9.85 | 1.000 | 33 | 1.00x | 14.2 | 22.0 |
| INT8 training (FP32 master) | 2.281 (0.001) | **2.288** (0.004) | +0.000 | 9.85 | 1.000 | 63 | 1.91x | 4.4 | 14.4 |
| Ternary + INT8 act (FP32 master) | 2.398 (0.008) | **2.404** (0.005) | +0.117 | 11.07 | 0.704 | 62 | 1.87x | 4.4 | 14.0 |
| Ternary + INT8 act (INT8 master) | 2.398 (0.007) | **2.403** (0.003) | +0.116 | 11.06 | 0.706 | 74 | 2.24x | 4.4 | 14.0 |
| Ternary + INT8 act (FP32 master), forward GEMM in C++ add/sub kernel | - | 2.406 | +0.000 vs NumPy ternary | 11.09 | - | 144 | 4.37x | 4.4 | 14.1 |

The ms/step column is wall-clock of a NumPy *simulation*: BLAS has no INT8 or 2-bit matmul, so quantised runs pay for extra quantise/dequantise passes and float32 stand-ins for the integer products. It shows the cost of emulating the arithmetic on a CPU, not what the accelerator would do; the C++ kernel timings below are the CPU-native numbers.


#### Does stochastic rounding matter? (measured, ternary model, 200 steps)

Validation loss, mean (std) over 3 seed(s), same model with the rounding mode switched.

| Rounded quantity | Stochastic rounding | Round to nearest | Nearest minus SR |
| --- | :---: | :---: | :---: |
| gradients, BFP8 | 2.404 (0.005) | 2.402 (0.016) | -0.002 |
| gradients, BFP4 | 2.432 (0.005) | 2.420 (0.011) | -0.013 |
| master weights, INT8 | 2.403 (0.003) | 2.437 (0.025) | +0.034 |
| master weights, 6-bit | 2.434 (0.023) | 2.498 (0.012) | +0.065 |
| master weights, 5-bit | 2.435 (0.015) | 2.912 (0.010) | +0.477 |

Positive last column = stochastic rounding is better.


#### C++ kernel timings on this CPU (measured)

`cpp_engine/build/benchmark_lowbit`, best of N runs, weights rotated through >=192 MB so every call streams from DRAM. The ternary kernel contains no multiply instruction (`scripts/check_multiplierless.sh`).

| Phase | N x K | FP32 SIMD | INT8 W8A8 (speedup) | Ternary (speedup) |
| --- | --- | :---: | :---: | :---: |
| decode | 512x512 | 38 us | 9 us (4.4x) | 3 us (14.4x) |
| decode | 2048x512 | 138 us | 32 us (4.3x) | 8 us (16.2x) |
| decode | 4096x4096 | 2,174 us | 535 us (4.1x) | 147 us (14.8x) |
| prefill | 512x512 | 155 us | 98 us (1.6x) | 99 us (1.6x) |
| prefill | 2048x512 | 636 us | 272 us (2.3x) | 491 us (1.3x) |
| prefill | 4096x4096 | 10,536 us | 6,441 us (1.6x) | 9,897 us (1.1x) |

ISA: AVX2; threads: 8. Decode = 1 token (matrix-vector); prefill = 128 tokens (matrix-matrix).


#### Assumptions and limits

- Traffic and PE numbers are an analytical model with the parameters shown; they are not RTL, synthesis, or board measurements.
- DRAM model: ideal streaming with the better of two dataflows (A-resident / B-resident), operands re-fetched only when the SRAM block is too small.
- AdamW moments are FP32 in every configuration (8 B/param); at low precision this optimiser state, not the weights, dominates remaining traffic. An 8-bit optimiser is the obvious next step.
- The residual stream, the output head, embeddings and norm gains stay FP32 in every configuration.
- Peak RAM is tracemalloc over NumPy arrays; integer products are emulated with float32 BLAS, which adds transient copies a real integer datapath would not have.
- Everything trained here is a 115k-parameter model for 200 steps; the convergence results say the method trains cleanly, not that it matches FP32 at scale.

<!-- LOWBIT-RESULTS:END -->

### What the measurements say

- **Activation memory shrinks 4.4x.** On the real 19M-parameter model, the activations held for the backward pass drop from 307 MB to 69 MB, and the analytical footprint model predicts the measured value to within 1%. (More than the 4x of INT8 vs FP32 alone, because the FP32 baseline also keeps `tanh(u)` for every GELU; per-row scales cost a little back.)
- **Whole-training memory shrinks only 1.8-2.3x**, because after the activations are compressed the FP32 AdamW moments (152 MB) and master weights (76 MB) dominate. Ternary weights make the *compute copy* 4.7 MB instead of 76 MB, but the training state is not the compute copy.
- **Off-chip traffic falls 59-65%, mostly from INT8 activations and gradients, not from the ternary weights.** INT8 training already saves 59%; ternary adds only a point over that (60%), and INT8 master weights add another five (65%). Ternary's value in *training* is the arithmetic, not the bandwidth.
- **The multiplier saving is real but scoped.** A 32x32 array drops from 3,072 DSP slices (FP32) to 0, and the ternary PE costs about 32 LUTs. 95% of this model's MACs are weight x activation (ternary-capable); the 4% attention products (activation x activation) and the FP32 output head still need multipliers.
- **Ternary trains, with a measurable gap.** After 200 steps validation loss is 2.404 (perplexity 11.1) against 2.287 (9.85) for FP32, a +0.117 nat gap with a small seed-to-seed spread. INT8 training is indistinguishable from FP32 (2.288). This is a 115k-parameter model for 200 steps: it shows the method trains stably, not that the gap stays this size at scale.
- **Stochastic rounding is decisive for low-precision weights and not for stored gradients.** Putting the master weights on a coarse grid: round-to-nearest loses 0.034 / 0.065 / **0.477** nats at INT8 / 6-bit / 5-bit, while stochastic rounding barely moves (updates below half an LSB survive in expectation instead of vanishing). For stored BFP gradients it makes no measurable difference at 8 bits and round-to-nearest is marginally *better* at 4 bits (SR trades bias for variance).
- **On a CPU the benefit is in decode, not prefill.** Single-token decode is memory-bound, so 4x fewer weight bytes than INT8 gives the ternary kernel a 3-4x edge over INT8 and 14-16x over FP32. For batched prefill the multiplier-less kernel is no faster than INT8 (equal at 512x512, slower at larger shapes: `madd` is a fast hardware multiplier on a CPU). The case for ternary is silicon area and DSP count, not CPU throughput.
- **The NumPy training loop is slower, not faster** (about 1.9-2.2x per step): BLAS has no INT8 or 2-bit matmul, so quantised runs pay for extra quantise/dequantise passes and float32 stand-ins for integer products. Peak RAM is likewise inflated by those temporaries. The wall-clock columns describe the cost of *simulating* the arithmetic; the C++ rows are the native numbers.

### Reproduce

```bash
cmake -S cpp_engine -B cpp_engine/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp_engine/build -j
ctest --test-dir cpp_engine/build --output-on-failure             # kernels, ternary format, multiplier-free check
python -m pytest tests -v                                         # bit-packing, SR bias, convergence vs FP32 (+ C++ cross-checks)
python python_scratch/hardware_profiler.py --update-readme        # full ~25 min run, refreshes the tables above (--quick for a 1-minute smoke run)
```

The full report with every table is also written to `docs/lowbit_results.md`.

### Limits

- Traffic and PE-array figures are an analytical model (parameters in `hardware_profiler.py`), not RTL, synthesis, or board measurements. The per-PE LUT/DSP costs are planning estimates.
- AdamW moments are FP32 in every configuration; an 8-bit optimiser is the obvious next saving and is not implemented. The residual stream, output head, embeddings and norm gains also stay FP32.
- Convergence evidence is one small model for 200 steps on TinyShakespeare.

---

## Quick Start

1. Run Pure NumPy vs. PyTorch Analytical Gradient Tests
```Bash

git clone [https://github.com/SahilSahu73/BareMetal-Transformer.git](https://github.com/SahilSahu73/BareMetal-Transformer.git)
cd BareMetal-Transformer
pip install numpy torch pytest
pytest python_scratch/test_gradients.py -v
```

2. Train Tiny Transformer & Export Weights (FP32 + INT8)
```Bash

python python_scratch/export_weights.py --output-dir ./checkpoints --quantize-int8
```

3. Compile & Benchmark the C++17 Inference Engine
```Bash

mkdir -p cpp_engine/build && cd cpp_engine/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./benchmark_transformer --weights ../../checkpoints/model_int8.bin --seq-len 256
```

