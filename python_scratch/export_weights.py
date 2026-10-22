"""Serialise a trained checkpoint into the flat binary format read by the C++ engine.

File layout (little endian, every tensor starts on a 64-byte boundary):

    header (64 B): "BMTF" | version u32 | dtype u32 (0 = FP32, 1 = INT8)
                   vocab, d_model, n_layers, n_heads, d_ff, max_seq (u32 each)
                   rope_theta f32 | norm_eps f32 | zero padding
    tensors, in the order given by tensor_names():
        FP32 tensor : float32[rows * cols]
        INT8 matrix : float32 scales[rows] (padded to 64 B), then int8[rows * cols]
    norm gains are always FP32.

Matrices are stored row-major as [out_features, in_features]: every output channel's
weights are contiguous, which is what the GEMV kernels stream through and what makes
per-output-channel scales natural. emb.E is already [vocab, d] = [out, in] for the tied
output projection, so it is stored as is (one scale per vocabulary row).
"""
import argparse
import os
import struct

import numpy as np

from data import load_tinyshakespeare
from transformer import CONFIGS, Config

MAGIC, REF_MAGIC, VERSION = b"BMTF", b"BMRF", 1
FP32, INT8 = 0, 1
ALIGN = 64
HEADER_FMT = "<4sIIIIIIIIff"
BLOCK_TENSORS = ["attn_norm.gamma", "attn.wq.W", "attn.wk.W", "attn.wv.W", "attn.wo.W",
                 "mlp_norm.gamma", "mlp.w1.W", "mlp.w2.W"]


def tensor_names(n_layers):
    names = ["emb.E"]
    for i in range(n_layers):
        names += [f"blocks.{i}.{t}" for t in BLOCK_TENSORS]
    return names + ["final_norm.gamma"]


def is_matrix(name):
    return name == "emb.E" or name.endswith(".W")


def to_file_layout(name, arr):
    """NumPy-side Linear weights are (in, out); the file stores (out, in)."""
    return np.ascontiguousarray(arr.T if name.endswith(".W") else arr, dtype=np.float32)


def quantize_per_channel(w):
    """Symmetric per-row INT8: scale = max|w| / 127, q = round(w / scale) in [-127, 127]."""
    scales = np.abs(w).max(axis=1) / 127.0
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    q = np.clip(np.rint(w / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales


def dequantize(q, scales):
    return q.astype(np.float32) * scales[:, None]


def _pad(f):
    f.write(b"\0" * (-f.tell() % ALIGN))


def write_model(path, cfg, params, dtype):
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, dtype, cfg.vocab, cfg.d_model, cfg.n_layers,
                            cfg.n_heads, cfg.d_ff, cfg.max_seq, cfg.rope_theta, cfg.norm_eps))
        for name in tensor_names(cfg.n_layers):
            _pad(f)
            w = to_file_layout(name, params[name])
            if dtype == INT8 and is_matrix(name):
                q, scales = quantize_per_channel(w)
                f.write(scales.tobytes())
                _pad(f)
                f.write(q.tobytes())
            else:
                f.write(w.tobytes())
        _pad(f)


def tensor_shape(cfg, name):
    d, ff = cfg.d_model, cfg.d_ff
    if name == "emb.E":
        return (cfg.vocab, d)
    if name.endswith("gamma"):
        return (d,)
    if name.endswith("w1.W"):
        return (ff, d)
    if name.endswith("w2.W"):
        return (d, ff)
    return (d, d)


def read_model(path):
    """Parse a model file back into (cfg, dtype, tensors, quantised).

    tensors:   name -> float32 array in file layout ([out, in]); INT8 matrices dequantised.
    quantised: name -> (int8 array, float32 scales) for INT8 matrices only.
    """
    buf = open(path, "rb").read()
    magic, ver, dtype, vocab, d, L, h, ff, T, theta, eps = struct.unpack_from(HEADER_FMT, buf, 0)
    assert magic == MAGIC and ver == VERSION, "not a BMTF v1 file"
    cfg = Config(vocab=vocab, d_model=d, n_layers=L, n_heads=h, max_seq=T, rope_theta=theta, norm_eps=eps)
    assert cfg.d_ff == ff
    pos, tensors, quantised = ALIGN, {}, {}
    align = lambda p: p + (-p % ALIGN)
    for name in tensor_names(L):
        pos = align(pos)
        shape = tensor_shape(cfg, name)
        n = int(np.prod(shape))
        if dtype == INT8 and is_matrix(name):
            scales = np.frombuffer(buf, np.float32, shape[0], pos)
            pos = align(pos + 4 * shape[0])
            q = np.frombuffer(buf, np.int8, n, pos).reshape(shape)
            pos += n
            quantised[name] = (q, scales)
            tensors[name] = dequantize(q, scales)
        else:
            tensors[name] = np.frombuffer(buf, np.float32, n, pos).reshape(shape)
            pos += 4 * n
    return cfg, dtype, tensors, quantised


def from_file_layout(tensors):
    """Inverse of to_file_layout: back to the NumPy/Torch parameter layout."""
    return {k: (v.T if k.endswith(".W") else v) for k, v in tensors.items()}


def write_reference(path, ids, logits_fp32, logits_int8):
    """Reference for the C++ tests: token ids plus the logits PyTorch produces for them
    (FP32 weights, and INT8 weights dequantised = what W8A32 kernels should reproduce)."""
    n, v = logits_fp32.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", REF_MAGIC, n, v))
        f.write(np.asarray(ids, np.int32).tobytes())
        f.write(np.ascontiguousarray(logits_fp32, np.float32).tobytes())
        f.write(np.ascontiguousarray(logits_int8, np.float32).tobytes())


def load_checkpoint(path, config_name):
    """Returns (cfg, {name: float32 array}) from a torch .pt or a NumPy .npz checkpoint."""
    if path.endswith(".pt"):
        import torch
        ck = torch.load(path, map_location="cpu")
        cfg = Config(**ck["config"])
        return cfg, {k: v.float().numpy() for k, v in ck["model"].items()}
    with np.load(path) as z:
        return CONFIGS[config_name], {k: z[k].astype(np.float32) for k in z.files}


def fake_quantize(params, cfg):
    """Replace every quantisable matrix by dequantize(quantize(W)) - the INT8 model's exact weights."""
    out = dict(params)
    for name in tensor_names(cfg.n_layers):
        if is_matrix(name):
            q, s = quantize_per_channel(to_file_layout(name, params[name]))
            out[name] = from_file_layout({name: dequantize(q, s)})[name]
    return out


def reference_logits(cfg, params, ids):
    import torch
    from torch_model import TorchGPT
    model = TorchGPT(cfg).double()
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in params.items()})
    with torch.no_grad():
        return model(torch.tensor(ids)[None])[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/torch_small.pt")
    ap.add_argument("--config", default="tiny", choices=list(CONFIGS), help="only needed for .npz checkpoints")
    ap.add_argument("--output-dir", default="checkpoints")
    ap.add_argument("--name", default=None, help="file stem, default: checkpoint's stem")
    ap.add_argument("--quantize-int8", action="store_true")
    args = ap.parse_args()

    cfg, params = load_checkpoint(args.ckpt, args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = args.name or os.path.splitext(os.path.basename(args.ckpt))[0]

    p32 = os.path.join(args.output_dir, f"{stem}_fp32.bin")
    write_model(p32, cfg, params, FP32)
    print(f"{p32}: {os.path.getsize(p32) / 1e6:.2f} MB")
    if args.quantize_int8:
        p8 = os.path.join(args.output_dir, f"{stem}_int8.bin")
        write_model(p8, cfg, params, INT8)
        print(f"{p8}: {os.path.getsize(p8) / 1e6:.2f} MB "
              f"({os.path.getsize(p32) / os.path.getsize(p8):.2f}x smaller)")

    _, val = load_tinyshakespeare()
    ids = val[:min(32, cfg.max_seq)]
    ref_path = os.path.join(args.output_dir, f"{stem}_reference.bin")
    write_reference(ref_path, ids, reference_logits(cfg, params, ids),
                    reference_logits(cfg, fake_quantize(params, cfg), ids))
    val.astype(np.uint8).tofile(os.path.join(args.output_dir, "val_tokens.bin"))
    print(f"{ref_path}: {len(ids)} reference tokens (+ val_tokens.bin for perplexity checks)")


if __name__ == "__main__":
    main()
