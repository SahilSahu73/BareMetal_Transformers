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
│   └── test_gradients.py       # Unit tests verifying NumPy gradients vs. PyTorch Autograd
├── cpp_engine/
│   ├── include/
│   │   ├── tensor.hpp          # Contiguous 64-byte aligned memory arena & Tensor views
│   │   ├── kernels.hpp         # Naive, Cache-Tiled, SIMD (AVX2/NEON), & INT8 GEMM headers
│   │   └── transformer.hpp     # C++17 Transformer block, KV-Cache, & token sampler
│   ├── src/
│   │   ├── kernels_simd.cpp    # Hardware-specific SIMD intrinsics & cache-tiling routines
│   │   ├── quantize.cpp        # Symmetric INT8 quantization & dequantization routines
│   │   ├── transformer.cpp     # End-to-end C++ forward pass & KV-Cache management
│   │   └── benchmark.cpp       # CLI microbenchmark suite (GFLOPS, ms/token, memory)
│   └── CMakeLists.txt          # Cross-platform build config (-O3 -march=native -fopenmp)
└── scripts/
    └── run_all_benchmarks.sh   # Runs gradient tests, exports weights, & benchmarks C++ kernels
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

