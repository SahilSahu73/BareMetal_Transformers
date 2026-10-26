"""Hardware processing-element and memory-bandwidth profiler for low-precision training.

Compares three ways of training the same transformer:

    FP32            FP32 weights, activations and gradients
    INT8            INT8 weights / activations / error signals, BFP8 weight gradients, FP32 master weights
    Ternary + INT8  2-bit {-1,0,+1} weights, INT8 activations, BFP8 gradients (FP32 or INT8 master weights)

Two kinds of numbers are produced and they are kept apart:

  ANALYTICAL  (explicit model, every assumption is a named parameter below): bytes crossing the
              off-chip DRAM <-> on-chip SRAM boundary per training step, split into weights,
              activations, gradients and optimiser state; and the multipliers / DSP slices / LUTs of a
              16x16, 32x32 and 64x64 processing-element (PE) array. These are first-order planning
              estimates, NOT synthesis results.
  EMPIRICAL   (measured here): convergence of the NumPy training loop over 200 steps, wall-clock per
              step, bytes actually retained by the cached activations / stored state, peak RAM, and
              the C++ kernel timings from cpp_engine/build/benchmark_lowbit.

    python python_scratch/hardware_profiler.py                 # full run, writes docs/lowbit_results.md
    python python_scratch/hardware_profiler.py --quick         # 40 steps, 1 seed (smoke test)
    python python_scratch/hardware_profiler.py --update-readme # also refresh the README section
"""
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

import low_bit_training as L
import lowbit_native
from transformer import CONFIGS, Config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MB = 1e6


# ============================================================================== precision configs
@dataclass(frozen=True)
class Precision:
    """Bytes per element for each tensor class (scales/exponents are added where they are material)."""

    name: str
    label: str
    w_compute: float   # weight copy read by the forward / backward matmuls
    w_refresh: float   # bytes written to refresh that copy after each optimiser step (0: it IS the master)
    master: float      # master (latent) weights, read + written by the optimiser
    act: float         # activations between layers
    err: float         # error signals (dL/dx) between layers
    cache: float       # activations kept for the backward pass
    grad: float        # stored weight gradient
    mac: str           # datapath of the matmul PEs: 'fp32' | 'int8' | 'ternary'
    train_preset: str  # the matching low_bit_training preset


BFP8 = 1 + 1 / 32  # INT8 mantissa + one shared exponent byte per block of 32

PRECISIONS = [
    Precision("fp32", "FP32 training", 4, 0, 4, 4, 4, 4, 4, "fp32", "fp32"),
    Precision("int8", "INT8 training (FP32 master)", 1, 1, 4, 1, 1, 1, BFP8, "int8", "int8"),
    Precision("ternary", "Ternary + INT8 act (FP32 master)", 0.25, 0.25, 4, 1, 1, 1, BFP8, "ternary", "ternary"),
    Precision("ternary_m8", "Ternary + INT8 act (INT8 master)", 0.25, 0.25, 1, 1, 1, 1, BFP8, "ternary", "ternary_int8_master"),
]
MOMENT_BYTES = 8  # AdamW m and v, FP32, in every configuration (as implemented; see "Limits" in the README)


# ============================================================================== DRAM traffic model
@dataclass(frozen=True)
class Hardware:
    pe: int = 32                 # PE array is pe x pe
    sram_bytes: int = 2 << 20    # on-chip buffer (BRAM/URAM) available to the matmul engine
    panel: int = 64              # K-panel streamed through the double buffer
    acc_bytes: int = 4           # on-chip accumulator width (32-bit)


def gemm_traffic(M, N, K, a_b, b_b, c_b, hw):
    """DRAM bytes (A reads, B reads, C writes) for C[M,N] = A[M,K] B[K,N] on a PE array with a finite SRAM.

    Two dataflows are tried and the cheaper one is taken:
      A-resident: a block of Tm rows of A stays on chip while all of B streams past it
                  -> A read once, B re-read ceil(M/Tm) times
      B-resident: a block of Tn columns of B stays on chip while all of A streams past it
                  -> B read once, A re-read ceil(N/Tn) times
    Tm / Tn are the largest multiples of the array size whose working set (resident block + a double-
    buffered K-panel of the streamed operand + 32-bit accumulators) fits the SRAM.
    """
    pe, sram = hw.pe, hw.sram_bytes

    def block(resident_b, streamed_b, extent, per_resident):
        t = int((sram - 2 * hw.panel * pe * streamed_b) // (per_resident * resident_b + pe * hw.acc_bytes))
        t = max(pe, t // pe * pe)  # a multiple of the array size, at least one array wide
        return min(t, math.ceil(extent / pe) * pe)

    tm = block(a_b, b_b, M, K)
    tn = block(b_b, a_b, N, K)
    a_res = (M * K * a_b, math.ceil(M / tm) * K * N * b_b, M * N * c_b)
    b_res = (math.ceil(N / tn) * M * K * a_b, K * N * b_b, M * N * c_b)
    return min(a_res, b_res, key=sum)


def _row(b, rowlen):
    """Bytes per element of a quantised tensor whose rows share one 4-byte scale."""
    return b + (4 / rowlen if b == 1 else 0)


def linear_shapes(c: Config):
    d, ff = c.d_model, c.d_ff
    return [("wq", d, d), ("wk", d, d), ("wv", d, d), ("wo", d, d), ("w1", d, ff), ("w2", ff, d)]  # (name, K, N)


def traffic_per_step(c: Config, batch, ctx, p: Precision, hw: Hardware):
    """Bytes across the DRAM boundary for one forward + backward + optimiser step, by tensor class."""
    M = batch * ctx
    out = {"weights": 0.0, "activations": 0.0, "gradients": 0.0, "optimizer": 0.0}
    L_, d, ff, h, T = c.n_layers, c.d_model, c.d_ff, c.n_heads, ctx
    dh = d // h

    # ---- the quantised linear layers: three GEMMs each (forward, error signal, weight gradient)
    n_lin_params = 0
    for _, K, N in linear_shapes(c):
        n_lin_params += L_ * K * N
        a, b, cc = gemm_traffic(M, N, K, _row(p.act, K), p.w_compute, _row(p.act, N), hw)            # Y  = X  W
        out["activations"] += L_ * (a + cc); out["weights"] += L_ * b
        a, b, cc = gemm_traffic(M, K, N, _row(p.err, N), p.w_compute, _row(p.err, K), hw)            # dX = dY W^T
        out["activations"] += L_ * (a + cc); out["weights"] += L_ * b
        a, b, cc = gemm_traffic(K, N, M, _row(p.cache, K), _row(p.err, N), p.grad, hw)               # dW = X^T dY
        out["activations"] += L_ * (a + b); out["gradients"] += L_ * cc

    # ---- tensors cached for backward by non-GEMM ops: written once in forward, read once in backward
    cache_elems = M * L_ * (2 * d + ff + 3 * d)  # RMSNorm xhat x2, GELU pre-activation, q/k/v
    cache_b = _row(p.cache, 64)
    attn_p = M * L_ * h * (T + 1) / 2             # softmax matrix (causal: on average (T+1)/2 entries per row)
    out["activations"] += 2 * (cache_elems * cache_b + attn_p * (1 if p.cache == 1 else 4))
    # ---- attention matmuls read q,k,v / P and write the context; backward writes dq,dk,dv (activation x activation)
    out["activations"] += M * L_ * (3 * d + d) * p.act + M * L_ * (d + 3 * d) * p.err
    # ---- residual stream and norm/GELU streams stay FP32 in every configuration (as implemented)
    out["activations"] += M * L_ * 12 * d * 4

    # ---- tied FP32 output head + embedding (unquantised in every configuration)
    V = c.vocab
    for a, b, cc in (gemm_traffic(M, V, d, 4, 4, 4, hw), gemm_traffic(M, d, V, 4, 4, 4, hw)):
        out["activations"] += a + cc; out["weights"] += b
    a, b, cc = gemm_traffic(d, V, M, 4, 4, 4, hw)
    out["activations"] += a + b; out["gradients"] += cc
    n_fp32_params = V * d + (2 * L_ + 1) * d      # embedding table + norm gains

    # ---- optimiser: read gradient, read + write master, refresh compute copy, read + write moments
    out["gradients"] += n_lin_params * p.grad + n_fp32_params * 4          # dW read back by the optimiser
    out["weights"] += n_lin_params * (2 * p.master + p.w_refresh) + n_fp32_params * 8
    out["optimizer"] += (n_lin_params + n_fp32_params) * 2 * MOMENT_BYTES
    out["total"] = sum(out.values())
    return out


# ============================================================================== memory footprint model
def footprint(c: Config, batch, ctx, p: Precision):
    """Bytes resident during a training step (the quantity that decides whether it fits on the device)."""
    M, L_, d, ff, h, T, V = batch * ctx, c.n_layers, c.d_model, c.d_ff, c.n_heads, ctx, c.vocab
    n_lin = L_ * sum(K * N for _, K, N in linear_shapes(c))
    n_fp32 = V * d + (2 * L_ + 1) * d
    quantised = p.name != "fp32"
    cache_b = lambda rowlen: p.cache + (4 / rowlen if p.cache == 1 else 0)
    act = M * L_ * (d * cache_b(d)                 # shared q/k/v input
                    + d * cache_b(d)               # wo input
                    + d * cache_b(d)               # w1 input
                    + ff * cache_b(ff)             # w2 input
                    + ff * cache_b(ff)             # GELU pre-activation (FP32 baseline also keeps tanh)
                    + 2 * d * cache_b(d)           # two RMSNorm xhat
                    + 3 * d * cache_b(d // h))     # q, k, v
    act += M * L_ * h * T * (1 if p.cache == 1 else 4) + (4 / T * M * L_ * h if p.cache == 1 else 0)  # softmax matrix
    if not quantised:
        act += M * L_ * ff * 4                     # GELU also caches tanh(u) in the FP32 baseline
    unquantised_act = M * V * 4 + 2 * M * d * 4     # logits probabilities, final hidden state and final-norm xhat: FP32 everywhere
    return {
        "master weights": n_lin * p.master + n_fp32 * 4,
        "compute weight copy": n_lin * (p.w_compute if p.w_refresh else 0),
        "weight gradients": n_lin * p.grad + n_fp32 * 4,
        "optimizer state": (n_lin + n_fp32) * MOMENT_BYTES,
        "cached activations": act + unquantised_act,
    }


# ============================================================================== PE array resources
@dataclass(frozen=True)
class PEModel:
    """Per-PE resource assumptions (Xilinx UltraScale+ class device). ESTIMATES for planning, not synthesis.

    fp32:    one FP32 multiplier (3 DSP48E2, full-usage variant of the Floating-Point Operator) and one FP32
             adder built from logic (~350 LUT) plus ~70 LUT of multiplier glue; 2 x 32-bit pipeline registers.
    int8:    one 8x8 multiplier + 32-bit accumulate in one DSP48E2 (no operand packing assumed: packing two
             INT8 MACs per DSP would halve this), ~10 LUT of control.
    ternary: a sign/zero-gate on the INT8 operand (1 LUT per bit = 8) + 24-bit adder/subtractor
             (1 LUT per bit on the carry chain = 24), 0 DSP. The accumulator is 24 bits: |sum| <= 127 * K < 2^23
             for K <= 65,536.
    """

    fp32_mul_dsp: int = 3
    fp32_mul_lut: int = 70
    fp32_add_dsp: int = 0
    fp32_add_lut: int = 350
    int8_mac_dsp: int = 1
    int8_mac_lut: int = 10
    tern_gate_lut: int = 8
    tern_add_lut: int = 24
    clock_mhz: int = 300


def pe_costs(pe_dim, m: PEModel = PEModel()):
    n = pe_dim * pe_dim
    return {
        "fp32": {"PEs": n, "multipliers": n, "DSP": n * (m.fp32_mul_dsp + m.fp32_add_dsp),
                 "add/sub units": n, "LUT": n * (m.fp32_mul_lut + m.fp32_add_lut)},
        "int8": {"PEs": n, "multipliers": n, "DSP": n * m.int8_mac_dsp,
                 "add/sub units": n, "LUT": n * m.int8_mac_lut},
        "ternary": {"PEs": n, "multipliers": 0, "DSP": 0,
                    "add/sub units": n, "LUT": n * (m.tern_gate_lut + m.tern_add_lut)},
    }


def attention_mac_share(c: Config, ctx):
    """Fraction of per-token MACs that are weight x activation (ternary-capable) vs activation x activation."""
    lin = c.n_layers * sum(K * N for _, K, N in linear_shapes(c))
    attn = c.n_layers * 2 * c.d_model * (ctx + 1) / 2  # QK^T and PV, causal average
    head = c.vocab * c.d_model
    return lin / (lin + attn + head), attn / (lin + attn + head), head / (lin + attn + head)


# ============================================================================== empirical runs
def load_data():
    from data import load_tinyshakespeare
    return load_tinyshakespeare()


def log(msg):
    """Progress goes to stderr, flushed, so a long run is visibly alive."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def run_training(cfg_name, steps, seeds, batch, backend="numpy"):
    c = CONFIGS[cfg_name]
    data = load_data()
    rows = {}
    for p in PRECISIONS:
        preset = L.PRESETS[p.train_preset]
        if preset is not None and backend != "numpy":
            preset = replace(preset, backend=backend)
        runs = []
        for s in range(seeds):
            log(f"training {p.label}, seed {s + 1}/{seeds}, {steps} steps ...")
            runs.append(L.train(c, preset if preset is not None else "fp32", data, steps=steps, batch_size=batch, seed=s,
                                measure_memory=(s == 0)))
            log(f"   done: val loss {runs[-1].val_loss:.3f}, {runs[-1].ms_per_step:.0f} ms/step")
        rows[p.name] = runs
    if backend == "numpy" and lowbit_native.available():  # same model, forward integer GEMM through the C++ add/sub kernel
        t = replace(L.PRESETS["ternary"], backend="native")
        rows["ternary_native"] = [L.train(c, t, data, steps=steps, batch_size=batch, seed=0, measure_memory=True)]
    return rows


def run_ablation(cfg_name, steps, seeds, batch):
    """Rounding mode vs precision of the two things that are rounded: stored gradients and master weights."""
    c, data = CONFIGS[cfg_name], load_data()
    T = L.PRESETS["ternary"]
    grid = [("gradients, BFP8", dict()), ("gradients, BFP4", dict(grad_bits=4)),
            ("master weights, INT8", dict(master_bits=8)), ("master weights, 6-bit", dict(master_bits=6)),
            ("master weights, 5-bit", dict(master_bits=5))]
    out = []
    for name, kw in grid:
        res = {}
        for mode in ("stochastic", "nearest"):
            log(f"ablation: {name}, {mode} rounding, {seeds} seed(s) ...")
            vals = [L.train(c, replace(T, rounding=mode, **kw), data, steps=steps, batch_size=batch, seed=s,
                            measure_memory=False).val_loss for s in range(seeds)]
            res[mode] = (float(np.mean(vals)), float(np.std(vals)))
        out.append((name, res))
    return out


def measure_big(cfg_name, batch, ctx, backend="numpy"):
    """One real forward+backward at the `small` shape: retained cache bytes, stored state, tracemalloc peak."""
    import tracemalloc
    from data import get_batch
    c = replace(CONFIGS[cfg_name], max_seq=ctx)
    train_tok, _ = load_data()
    x, y = get_batch(train_tok, batch, ctx, np.random.default_rng(0))
    rows = {}
    for p in PRECISIONS:
        preset = L.PRESETS[p.train_preset]
        if preset is not None and backend != "numpy":
            preset = replace(preset, backend=backend)
        m = L.build_model(c, preset if preset is not None else "fp32", 0)
        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        m.loss(x, y)
        cache = L.activation_cache_bytes(m)
        m.backward()
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rows[p.name] = {"cache": cache, "peak": peak, "step_s": dt, "state": L.training_state_bytes(m)}
    return rows


def run_cpp_benchmark():
    exe = os.path.join(ROOT, "cpp_engine", "build", "benchmark_lowbit")
    if not os.path.exists(exe):
        return None
    path = os.path.join(ROOT, "docs", "lowbit_kernels.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    txt = subprocess.run([exe, "--csv", path], capture_output=True, text=True, timeout=1800).stdout
    with open(path) as f:
        return txt, list(csv.DictReader(f))


# ============================================================================== markdown
def fmt_mb(x):
    return f"{x / MB:,.1f}"


def md_table(header, rows, align=None):
    align = align or ["---"] + [":---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)



def build_summary(js):
    """The one-table overview: memory, bandwidth, multipliers and convergence per configuration."""
    rows = []
    fp = js["footprint"]["fp32"]
    for p in PRECISIONS:
        n = p.name
        foot = sum(js["footprint"][n].values())
        peak = js.get("measured_big", {}).get(n, {}).get("peak")
        cache = js.get("measured_big", {}).get(n, {}).get("cache")
        traf = js["traffic"][n]["total"]
        pe = js["pe"]["32"][p.mac]
        val = np.mean(js["train"][n]["val"]) if "train" in js else float("nan")
        base = np.mean(js["train"]["fp32"]["val"]) if "train" in js else float("nan")
        rows.append([
            p.label,
            f"{foot / MB:,.0f} ({sum(fp.values()) / foot:.2f}x)" if n != "fp32" else f"{foot / MB:,.0f}",
            f"{peak / MB:,.0f}" if peak else "-",
            f"{cache / MB:,.0f} ({js['measured_big']['fp32']['cache'] / cache:.1f}x)" if cache and n != "fp32" else (f"{cache / MB:,.0f}" if cache else "-"),
            f"{traf / MB:,.0f}", f"{100 * (1 - traf / js['traffic']['fp32']['total']):.0f}%" if n != "fp32" else "-",
            f"{pe['multipliers']:,} / {pe['DSP']:,}",
            f"{val:.3f}", f"{val - base:+.3f}" if n != "fp32" else "-"])
    head = ["Configuration", "Training memory, MB (analytical)", "Peak RAM, MB (measured)", "Activation cache, MB (measured)",
            "Off-chip traffic, MB/step", "Off-chip bandwidth saved", "Multipliers / DSP (32x32 PE array)",
            "Val loss, 200 steps", "vs FP32"]
    return ("## Summary\n\n`small` model (19M parameters), batch 4 x 256 tokens for the memory and bandwidth columns; `tiny` model (115k parameters), "
            "200 AdamW steps, 3 seeds for the loss columns. Details, assumptions and the measured-vs-analytical split follow.\n\n"
            + md_table(head, rows) + "\n\n")

def report(args):
    md, js = [], {}
    cfg = CONFIGS["small"]
    hw = Hardware(pe=32)

    # ------------------------------------------------------------------ traffic
    md.append(f"## Off-chip memory traffic per training step (analytical)\n")
    md.append(f"`small` model ({cfg.n_layers} layers, d={cfg.d_model}, {cfg.n_heads} heads, ~19M parameters), batch {args.big_batch} x context {cfg.max_seq} tokens, "
              f"32x32 PE array, {hw.sram_bytes >> 20} MiB on-chip SRAM, ideal-streaming DRAM model. MB per step.\n")
    base = traffic_per_step(cfg, args.big_batch, cfg.max_seq, PRECISIONS[0], hw)
    rows, js["traffic"] = [], {}
    for p in PRECISIONS:
        t = traffic_per_step(cfg, args.big_batch, cfg.max_seq, p, hw)
        js["traffic"][p.name] = t
        rows.append([p.label, fmt_mb(t["weights"]), fmt_mb(t["activations"]), fmt_mb(t["gradients"]), fmt_mb(t["optimizer"]),
                     f"**{fmt_mb(t['total'])}**", f"{100 * (1 - t['total'] / base['total']):.0f}%" if p.name != "fp32" else "-"])
    md.append(md_table(["Configuration", "Weights", "Activations", "Gradients", "Optimizer state", "Total", "Bandwidth saved"], rows))

    md.append("\n**Sensitivity to the PE array size and SRAM budget** (total MB per step; the bigger the buffer, the less operands are re-fetched):\n")
    rows = []
    for pe in (16, 32, 64):
        for sram in (512 << 10, 2 << 20, 8 << 20):
            h = Hardware(pe=pe, sram_bytes=sram)
            tots = [traffic_per_step(cfg, args.big_batch, cfg.max_seq, p, h)["total"] for p in PRECISIONS]
            rows.append([f"{pe}x{pe}", f"{sram >> 10} KiB"] + [fmt_mb(t) for t in tots] + [f"{100 * (1 - tots[2] / tots[0]):.0f}%"])
    md.append(md_table(["PE array", "SRAM"] + [p.label for p in PRECISIONS] + ["Ternary saving"], rows, ["---", "---"] + [":---:"] * 5))

    # ------------------------------------------------------------------ footprint (analytic + measured)
    md.append("\n## Training memory footprint\n")
    md.append(f"Resident bytes during a step at the same shape (MB). *Analytical* = from the tensor inventory above; "
              f"*measured* = bytes actually held by this repository's NumPy training code ({'native C++' if args.backend == 'native' else 'NumPy'} integer GEMMs) "
              f"for one real forward+backward.\n")
    foot = {p.name: footprint(cfg, args.big_batch, cfg.max_seq, p) for p in PRECISIONS}
    keys = list(foot["fp32"])
    rows = []
    for p in PRECISIONS:
        f = foot[p.name]
        rows.append([p.label] + [fmt_mb(f[k]) for k in keys] + [f"**{fmt_mb(sum(f.values()))}**",
                                                                 f"{sum(foot['fp32'].values()) / sum(f.values()):.2f}x" if p.name != "fp32" else "-"])
    md.append(md_table(["Configuration (analytical)"] + [k.capitalize() for k in keys] + ["Total", "Reduction"], rows))
    js["footprint"] = foot

    log("measuring one real forward+backward at the `small` shape ...")
    big = measure_big("small", args.big_batch, cfg.max_seq, args.backend) if not args.skip_big else None
    if big:
        md.append("\n**Measured on the real `small` model** (one forward + backward):\n")
        rows = []
        for p in PRECISIONS:
            m = big[p.name]
            ana = foot[p.name]["cached activations"]
            rows.append([p.label, fmt_mb(m["cache"]), fmt_mb(ana), f"{big['fp32']['cache'] / m['cache']:.2f}x" if p.name != "fp32" else "-",
                         fmt_mb(m["peak"]), f"{m['step_s']:.2f} s"])
        md.append(md_table(["Configuration", "Activation cache held after forward", "(analytical)", "Cache reduction", "Peak RAM during step (tracemalloc)", "Step time"], rows))
        js["measured_big"] = big

    # ------------------------------------------------------------------ PE arrays
    pm = PEModel()
    md.append("\n## Processing-element array cost (analytical)\n")
    md.append(f"Multipliers (DSP slices) versus add/subtract units for the weight x activation matmul engine. Per-PE assumptions: "
              f"FP32 MAC = {pm.fp32_mul_dsp} DSP + {pm.fp32_mul_lut + pm.fp32_add_lut} LUT; INT8 MAC = {pm.int8_mac_dsp} DSP + {pm.int8_mac_lut} LUT; "
              f"ternary PE = 0 DSP + {pm.tern_gate_lut + pm.tern_add_lut} LUT (sign gate + 24-bit add/sub). Estimates for planning, not synthesis results.\n")
    rows, js["pe"] = [], {}
    for dim in (16, 32, 64):
        c = pe_costs(dim, pm)
        js["pe"][dim] = c
        peak = lambda k: dim * dim * pm.clock_mhz * 1e6 / 1e9
        rows.append([f"{dim}x{dim} ({dim * dim})",
                     f"{c['fp32']['multipliers']:,} FP32 mul / {c['fp32']['DSP']:,} DSP / {c['fp32']['LUT']:,} LUT",
                     f"{c['int8']['multipliers']:,} INT8 mul / {c['int8']['DSP']:,} DSP / {c['int8']['LUT']:,} LUT",
                     f"**0 mul / 0 DSP** / {c['ternary']['LUT']:,} LUT ({c['ternary']['add/sub units']:,} add/sub)",
                     f"{c['fp32']['DSP']:,} -> 0",
                     f"{peak(0):.0f}"])
    md.append(md_table(["Array", "FP32 PE array", "INT8 PE array", "Ternary PE array", "DSP reduction vs FP32", f"Peak GMAC/s @ {pm.clock_mhz} MHz"], rows,
                       ["---", ":---:", ":---:", ":---:", ":---:", ":---:"]))
    lin, att, head = attention_mac_share(cfg, cfg.max_seq)
    md.append(f"\nScope: in the `small` model {100 * lin:.0f}% of the per-token MACs are weight x activation products "
              f"(the ternary-capable linear layers), {100 * att:.0f}% are the activation x activation attention products (QK^T, PV), "
              f"and {100 * head:.0f}% are the FP32 output head. Attention and the head still need real multipliers; only the linear layers become multiplier-less.\n")

    # ------------------------------------------------------------------ convergence
    steps, seeds = args.steps, args.seeds
    md.append(f"\n## Convergence and execution speed: tiny transformer, {steps} steps (measured)\n")
    runs = run_training("tiny", steps, seeds, args.batch, args.backend) if not args.skip_train else None
    if runs:
        c = CONFIGS["tiny"]
        md.append(f"`tiny` model ({L.build_model(c, 'fp32').num_params():,} parameters), TinyShakespeare bytes, batch {args.batch} x {c.max_seq}, AdamW, cosine schedule, "
                  f"{seeds} seed(s); mean (std) across seeds. Validation loss is measured on the quantised network itself.\n")
        rows, js["train"] = [], {}
        b = runs["fp32"]
        b_val = np.mean([r.val_loss for r in b])
        for p in PRECISIONS:
            rs = runs[p.name]
            val, trn = [r.val_loss for r in rs], [np.mean(r.losses[-20:]) for r in rs]
            r0 = rs[0]
            js["train"][p.name] = {"val": val, "train_last20": trn, "ms_per_step": r0.ms_per_step, "peak": r0.peak_bytes,
                                   "cache": r0.cache_bytes, "grad_cosine": r0.grad_cosine, "curve": r0.losses}
            rows.append([p.label, f"{np.mean(trn):.3f} ({np.std(trn):.3f})", f"**{np.mean(val):.3f}** ({np.std(val):.3f})",
                         f"{np.mean(val) - b_val:+.3f}", f"{math.exp(np.mean(val)):.2f}", f"{r0.grad_cosine:.3f}",
                         f"{r0.ms_per_step:.0f}", f"{r0.ms_per_step / b[0].ms_per_step:.2f}x",
                         fmt_mb(r0.cache_bytes), fmt_mb(r0.peak_bytes)])
        if "ternary_native" in runs:
            r0, tern = runs["ternary_native"][0], runs["ternary"][0]
            js["train"]["ternary_native"] = {"val": [r0.val_loss], "ms_per_step": r0.ms_per_step, "peak": r0.peak_bytes, "cache": r0.cache_bytes}
            rows.append(["Ternary + INT8 act (FP32 master), forward GEMM in C++ add/sub kernel", "-", f"{r0.val_loss:.3f}", f"{r0.val_loss - tern.val_loss:+.3f} vs NumPy ternary",
                         f"{math.exp(r0.val_loss):.2f}", "-", f"{r0.ms_per_step:.0f}", f"{r0.ms_per_step / b[0].ms_per_step:.2f}x", fmt_mb(r0.cache_bytes), fmt_mb(r0.peak_bytes)])
        md.append(md_table(["Configuration", "Train loss (last 20)", "Val loss", "vs FP32", "Val perplexity", "Grad cosine vs FP32",
                            "ms / step", "Time vs FP32", "Cache MB", "Peak RAM MB"], rows))
        md.append("\nThe ms/step column is wall-clock of a NumPy *simulation*: BLAS has no INT8 or 2-bit matmul, so quantised runs pay for extra quantise/dequantise "
                  "passes and float32 stand-ins for the integer products. It shows the cost of emulating the arithmetic on a CPU, not what the accelerator would do; "
                  "the C++ kernel timings below are the CPU-native numbers.\n")

    # ------------------------------------------------------------------ ablation
    if not args.skip_ablation and not args.skip_train:
        md.append(f"\n## Does stochastic rounding matter? (measured, ternary model, {steps} steps)\n")
        md.append(f"Validation loss, mean (std) over {seeds} seed(s), same model with the rounding mode switched.\n")
        ab = run_ablation("tiny", steps, seeds, args.batch)
        rows, js["ablation"] = [], {}
        for name, res in ab:
            s, n = res["stochastic"], res["nearest"]
            js["ablation"][name] = res
            rows.append([name, f"{s[0]:.3f} ({s[1]:.3f})", f"{n[0]:.3f} ({n[1]:.3f})", f"{n[0] - s[0]:+.3f}"])
        md.append(md_table(["Rounded quantity", "Stochastic rounding", "Round to nearest", "Nearest minus SR"], rows))
        md.append("\nPositive last column = stochastic rounding is better.\n")

    # ------------------------------------------------------------------ C++
    log("running the C++ kernel benchmark ...")
    cpp = run_cpp_benchmark() if not args.skip_cpp else None
    if cpp:
        txt, data = cpp
        md.append("\n## C++ kernel timings on this CPU (measured)\n")
        md.append(f"`cpp_engine/build/benchmark_lowbit`, best of N runs, weights rotated through >=192 MB so every call streams from DRAM. "
                  f"The ternary kernel contains no multiply instruction (`scripts/check_multiplierless.sh`).\n")
        rows = []
        for phase in ("decode", "prefill"):
            for shape in ("512x512", "2048x512", "4096x4096"):
                sel = {r["kernel"]: r for r in data if r["phase"] == phase and r["shape"] == shape}
                f = sel["fp32 SIMD"]
                row = [phase, shape]
                for k in ("fp32 SIMD", "int8 W8A8 (madd)", "ternary 2-bit (add/sub only)"):
                    ms = float(sel[k]["ms"])
                    row.append(f"{ms * 1000:,.0f} us" + ("" if k == "fp32 SIMD" else f" ({float(f['ms']) / ms:.1f}x)"))
                rows.append(row)
        md.append(md_table(["Phase", "N x K", "FP32 SIMD", "INT8 W8A8 (speedup)", "Ternary (speedup)"], rows, ["---", "---", ":---:", ":---:", ":---:"]))
        js["cpp"] = data
        md.append(f"\nISA: {lowbit_native.isa() if lowbit_native.available() else 'n/a'}; threads: {data[0]['threads']}. Decode = 1 token (matrix-vector); prefill = 128 tokens (matrix-matrix).\n")

    if "train" in js and "measured_big" in js:
        md.insert(0, build_summary(js))
    md.append("\n## Assumptions and limits\n")
    md.append("\n".join([
        "- Traffic and PE numbers are an analytical model with the parameters shown; they are not RTL, synthesis, or board measurements.",
        "- DRAM model: ideal streaming with the better of two dataflows (A-resident / B-resident), operands re-fetched only when the SRAM block is too small.",
        f"- AdamW moments are FP32 in every configuration ({MOMENT_BYTES} B/param); at low precision this optimiser state, not the weights, dominates remaining traffic. An 8-bit optimiser is the obvious next step.",
        "- The residual stream, the output head, embeddings and norm gains stay FP32 in every configuration.",
        "- Peak RAM is tracemalloc over NumPy arrays; integer products are emulated with float32 BLAS, which adds transient copies a real integer datapath would not have.",
        "- Everything trained here is a 115k-parameter model for 200 steps; the convergence results say the method trains cleanly, not that it matches FP32 at scale.",
    ]))
    return "\n".join(md) + "\n", js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--big-batch", type=int, default=4)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "native"])
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "lowbit_results.md"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-big", action="store_true")
    ap.add_argument("--skip-cpp", action="store_true")
    ap.add_argument("--update-readme", action="store_true", help="after a run, refresh the tables in README.md")
    ap.add_argument("--readme-from-results", action="store_true", help="only refresh README.md from the existing --out file")
    ap.add_argument("--add-summary", action="store_true", help="prepend the summary table to an existing --out file (from its .json) and stop")
    args = ap.parse_args()
    if args.add_summary:
        js = json.load(open(os.path.splitext(args.out)[0] + ".json"))
        text = open(args.out).read()
        if "## Summary" not in text:
            head, _, rest = text.partition("\n\n## ")
            open(args.out, "w").write(head + "\n\n" + build_summary(js) + "## " + rest)
        return print(f"summary added to {args.out}")
    if args.readme_from_results:
        body = open(args.out).read().split("\n", 3)[3]  # drop the title and the 'generated by' line
        return update_readme(body)
    if args.quick:
        args.steps, args.seeds, args.skip_ablation = 40, 1, True
    if args.backend == "native" and not lowbit_native.available():
        raise SystemExit("--backend native needs cpp_engine/build/libbmlowbit.so")

    log("start")
    text, js = report(args)
    log("finished")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Low-precision edge training: measured results\n\n_Generated by `python_scratch/hardware_profiler.py`._\n\n" + text)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(js, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(text)
    print(f"wrote {args.out}")
    if args.update_readme:
        update_readme(text)


def update_readme(tables_md):
    path = os.path.join(ROOT, "README.md")
    s = open(path).read()
    begin, end = "<!-- LOWBIT-RESULTS:BEGIN -->", "<!-- LOWBIT-RESULTS:END -->"
    if begin not in s or end not in s:
        raise SystemExit("README.md has no LOWBIT-RESULTS markers")
    body = re.sub(r"^## ", "#### ", tables_md, flags=re.M)
    s = re.sub(re.escape(begin) + ".*?" + re.escape(end), lambda _: begin + "\n" + body + "\n" + end, s, flags=re.S)
    open(path, "w").write(s)
    print("updated README.md")


if __name__ == "__main__":
    main()
