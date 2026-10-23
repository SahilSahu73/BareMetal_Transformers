"""Perplexity of the FP32 model vs. its INT8 (fake-quantised) versions on the validation split.

Windows are non-overlapping and deterministic, so the C++ engine can reproduce the number.
Per-tensor quantisation is included to show why the engine uses per-channel scales.
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from data import load_tinyshakespeare
from export_weights import (fake_quantize, is_matrix, load_checkpoint,
                            tensor_names, to_file_layout, from_file_layout, quantize_per_channel)
from torch_model import TorchGPT
from transformer import CONFIGS


def fake_quantize_per_tensor(params, cfg):
    out = dict(params)
    for name in tensor_names(cfg.n_layers):
        if is_matrix(name):
            w = to_file_layout(name, params[name])
            s = max(float(np.abs(w).max()) / 127.0, 1e-12)
            wq = np.clip(np.rint(w / s), -127, 127).astype(np.float32) * s
            out[name] = from_file_layout({name: wq})[name]
    return out


@torch.no_grad()
def perplexity(cfg, params, val, batch=16, act=False, kv=False):
    model = TorchGPT(cfg)
    model.load_state_dict({k: torch.tensor(v) for k, v in params.items()})
    model.set_quant(act=act, kv=kv)
    T = cfg.max_seq
    n = (len(val) - 1) // T
    x = torch.from_numpy(val[:n * T].reshape(n, T)).long()
    y = torch.from_numpy(val[1:n * T + 1].reshape(n, T)).long()
    total = 0.0
    for i in range(0, n, batch):
        logits = model(x[i:i + batch])
        total += F.cross_entropy(logits.flatten(0, 1), y[i:i + batch].flatten(), reduction="sum").item()
    return math.exp(total / (n * T)), n * T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/torch_small.pt")
    ap.add_argument("--config", default="tiny", choices=list(CONFIGS))
    args = ap.parse_args()
    cfg, params = load_checkpoint(args.ckpt, args.config)
    _, val = load_tinyshakespeare()

    q = fake_quantize(params, cfg)
    base, n_tok = perplexity(cfg, params, val)
    print(f"validation tokens: {n_tok:,}")
    rows = [
        ("FP32", params, {}),
        ("INT8 weights, per-channel (W8A32)", q, {}),
        ("INT8 weights, per-tensor", fake_quantize_per_tensor(params, cfg), {}),
        ("INT8 weights + INT8 activations (W8A8)", q, {"act": True}),
        ("FP32 weights + INT8 KV-cache", params, {"kv": True}),
        ("INT8 weights + INT8 KV-cache", q, {"kv": True}),
        ("W8A8 + INT8 KV-cache", q, {"act": True, "kv": True}),
    ]
    for label, p, kw in rows:
        ppl = base if label == "FP32" else perplexity(cfg, p, val, **kw)[0]
        delta = "" if label == "FP32" else f"   delta {100 * (ppl - base) / base:+.3f}%"
        print(f"{label:42s} ppl {ppl:8.4f}{delta}")

if __name__ == "__main__":
    main()
