"""Low-precision edge training on the from-scratch NumPy stack.

A transformer whose linear layers run on 2-bit ternary weights {-1, 0, +1} and INT8 activations,
trained with the memory-saving tricks an edge device (or an FPGA accelerator) needs:

  * Straight-through estimator (STE). The forward pass uses quantised weights; the backward pass
    treats the quantiser as the identity, so the gradient lands on a latent higher-precision copy
    ("master" weights) that accumulates the small updates the ternary weights could never hold.
  * INT8 activation caching. Backpropagation needs the forward activations, so they sit in memory
    until the backward pass reaches them: the dominant training-memory cost. Every cached tensor
    is compressed to INT8 (one scale per row) after the forward pass and dequantised on the fly
    during backward.
  * Block-floating-point gradients with stochastic rounding. Weight gradients are stored as INT8
    (or INT4) mantissas sharing a power-of-two exponent per block of 32. Stochastic rounding makes
    the quantiser unbiased - E[SR(x)] = x - so contributions smaller than one LSB still arrive in
    expectation instead of rounding to zero every time.
  * Optionally the master weights themselves live on an INT8 grid, updated with stochastic rounding.

Everything is plain NumPy on top of nn_layers.py: forward and backward are the same hand-derived
formulas, with quantisers inserted. The integer matrix products are exact in float32 (they stay
below 2^24), so the NumPy path is bit-identical to the C++ multiplier-less kernel; set
backend="native" to run the forward products through libbmlowbit (cpp_engine) instead.
"""
import math
import time
import tracemalloc
import weakref
from dataclasses import dataclass, replace

import numpy as np

import lowbit_native
from attention import MultiHeadCausalAttention
from nn_layers import GELU, Module, RMSNorm
from transformer import Config, NumpyGPT

TERNARY_BLOCK = 32  # row padding of the packed format, shared with cpp_engine/include/ternary.hpp
INT8_MAX = 127


# ------------------------------------------------------------------------------ quantisers
def ternary_quantize_absmean(W):
    """BitNet b1.58 absmean: gamma = mean|W|, Wq = clip(round(W / gamma), -1, 1). Returns (int8, gamma).

    Arithmetic mirrors cpp_engine/src/ternary.cpp (float32 multiply by 1/gamma, round half to even)
    so both implementations agree bit for bit.
    """
    W = np.asarray(W, dtype=np.float32)
    if W.size == 0:
        return np.zeros(W.shape, np.int8), np.float32(0)
    gamma = np.float32(np.abs(W).astype(np.float64).sum() / W.size)
    if gamma == 0:
        return np.zeros(W.shape, np.int8), np.float32(0)
    inv = np.float32(1.0) / gamma
    return np.clip(np.rint(W * inv), -1, 1).astype(np.int8), gamma


def packed_row_bytes(K):
    return (K + TERNARY_BLOCK - 1) // TERNARY_BLOCK * (TERNARY_BLOCK // 4)


def pack_ternary(Wq):
    """int8 (rows, K) in {-1, 0, +1} -> uint8 (rows, packed_row_bytes(K)); four weights per byte.

    byte = w0 | w1 << 2 | w2 << 4 | w3 << 6 with codes 0 -> 00, +1 -> 01, -1 -> 10. Rows are padded
    with zero codes to a multiple of 32 weights.
    """
    Wq = np.asarray(Wq, dtype=np.int8)
    rows, K = Wq.shape
    codes = np.where(Wq > 0, 1, np.where(Wq < 0, 2, 0)).astype(np.uint8)
    padded = np.zeros((rows, packed_row_bytes(K) * 4), dtype=np.uint8)
    padded[:, :K] = codes
    q = padded.reshape(rows, -1, 4)
    return q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)


def unpack_ternary(packed, K):
    """Inverse of pack_ternary. The reserved code 0b11 decodes as 0, like the C++ kernels."""
    packed = np.asarray(packed, dtype=np.uint8)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    codes = (packed[..., None] >> shifts) & 3
    codes = codes.reshape(packed.shape[0], -1)[:, :K]
    return np.where(codes == 1, 1, np.where(codes == 2, -1, 0)).astype(np.int8)


def stochastic_round(x, rng):
    """floor(x + u), u ~ U[0,1): rounds up with probability x - floor(x), so E[SR(x)] = x exactly."""
    x = np.asarray(x)
    return np.floor(x.astype(np.float64) + rng.random(x.shape)).astype(np.float32)


def _round(y, rounding, rng):
    return stochastic_round(y, rng) if rounding == "stochastic" else np.rint(y)


def quantize_int8_rows(x, rounding="nearest", rng=None):
    """Symmetric INT8 along the last axis, one scale per row: scale = max|x| / 127.

    Returns (q int8, scale float32 with shape x.shape[:-1] + (1,)). An all-zero row gets scale 1.
    """
    x = np.asarray(x, dtype=np.float32)
    amax = np.abs(x).max(axis=-1, keepdims=True)
    scale = np.where(amax > 0, amax / np.float32(INT8_MAX), np.float32(1)).astype(np.float32)
    q = np.clip(_round(x / scale, rounding, rng), -INT8_MAX, INT8_MAX).astype(np.int8)
    return q, scale


def dequantize_int8_rows(q, scale):
    return q.astype(np.float32) * scale


def quantize_uint8_rows(p):
    """Unsigned 8-bit for non-negative data (softmax probabilities): scale = max / 255 per row."""
    p = np.asarray(p, dtype=np.float32)
    scale = (p.max(axis=-1, keepdims=True) / np.float32(255)).astype(np.float32)
    scale = np.where(scale > 0, scale, np.float32(1))
    return np.clip(np.rint(p / scale), 0, 255).astype(np.uint8), scale


@dataclass
class BFPTensor:
    """Block floating point: int8 mantissas, one shared power-of-two exponent per block of values."""

    mant: np.ndarray  # int8, flat, trimmed to the original size
    exp: np.ndarray   # int8, one per block
    shape: tuple
    block: int
    bits: int

    def dequantize(self):
        scale = np.ldexp(np.float32(1), self.exp.astype(np.int32)).astype(np.float32)
        n = self.mant.size
        out = self.mant.astype(np.float32) * np.repeat(scale, self.block)[:n]
        return out.reshape(self.shape)

    @property
    def nbytes(self):
        """Storage if the mantissas were bit-packed: bits per value plus one exponent byte per block."""
        return (self.mant.size * self.bits + 7) // 8 + self.exp.size


def bfp_quantize(x, block=32, bits=8, rounding="stochastic", rng=None):
    """Quantise x (any shape, flattened row-major) to BFP. The shared exponent of a block is the smallest e
    such that max|x| / 2^e <= 2^(bits-1) - 1; mantissas are rounded (stochastically or to nearest).

    Matches cpp_engine/src/lowbit_train.cpp bit for bit in rounding="nearest".
    """
    x = np.asarray(x, dtype=np.float32)
    flat = x.reshape(-1)
    n = flat.size
    nb = (n + block - 1) // block
    padded = np.zeros(nb * block, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(nb, block)
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(blocks).max(axis=1)
    m, p = np.frexp(amax / np.float32(qmax))  # amax/qmax = m * 2^p, m in [0.5, 1)
    e = np.where(m == 0.5, p - 1, p)           # exact ceil(log2(.)) even at powers of two
    e = np.where(amax > 0, np.clip(e, -127, 127), 0).astype(np.int32)
    y = blocks * np.ldexp(np.float32(1), -e).astype(np.float32)[:, None]
    mant = np.clip(_round(y, rounding, rng), -qmax, qmax).astype(np.int8).reshape(-1)[:n]
    return BFPTensor(mant, e.astype(np.int8), x.shape, block, bits)


def quantize_to_grid(x, scale, bits=8, rounding="stochastic", rng=None):
    """Round x / scale onto the signed `bits`-bit integer grid (clamped)."""
    qmax = (1 << (bits - 1)) - 1
    return np.clip(_round(np.asarray(x, np.float32) / np.float32(scale), rounding, rng), -qmax, qmax).astype(np.int8)


# ------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class LowBitConfig:
    weights: str = "ternary"        # 'ternary' (absmean, 2-bit), 'int8' (per output channel) or 'fp32'
    act_bits: int = 8               # INT8 activations into every quantised linear layer
    cache: str = "int8"             # how forward activations are kept for backward: 'int8' | 'fp32'
    cache_attention: bool = True    # also compress q, k, v and the softmax matrix
    grad_bits: int = 8              # weight-gradient mantissa width (8 or 4); 0 keeps FP32 gradients
    grad_block: int = 32            # values per shared exponent
    rounding: str = "stochastic"    # for gradients and the INT8 error signal: 'stochastic' | 'nearest'
    quantize_dy: bool = True        # error signal into the backward matmul is INT8 as well
    master_bits: int = 0            # 0: FP32 latent weights; 8: latent weights live on an INT8 grid
    master_range_sigma: float = 6.0  # INT8 master grid covers +-6 sigma of the initial weights
    backend: str = "numpy"          # 'numpy' | 'native' (forward integer GEMM in C++)
    seed: int = 0


PRESETS = {
    "fp32": None,
    "int8": LowBitConfig(weights="int8"),
    "ternary": LowBitConfig(weights="ternary"),
    "ternary_int8_master": LowBitConfig(weights="ternary", master_bits=8),
}


# ------------------------------------------------------------------------------ layers
class _InputShare:
    """Lets wq / wk / wv (which all read the same tensor) quantise and cache that input once."""

    def __init__(self):
        self.ref = None
        self.value = None

    def get(self, x):
        if self.ref is not None and self.ref() is x:
            return self.value
        return None

    def put(self, x, value):
        self.ref = weakref.ref(x)  # weak: the FP32 input must not be kept alive by the cache
        self.value = value


class LowBitLinear(Module):
    """y = x W with quantised weights/activations, STE backward, INT8 activation cache, BFP gradients.

    W is stored (d_in, d_out) like nn_layers.Linear and holds the latent (master) weights.
    """

    def __init__(self, W, cfg, ctx, share=None):
        super().__init__()
        self.cfg, self.ctx, self.share = cfg, ctx, share
        self.params["W"] = np.array(W, dtype=np.float32)
        self.grad_bfp = None
        self._cache = None
        self._Wq = self._wscale = None
        if cfg.master_bits:
            qmax = (1 << (cfg.master_bits - 1)) - 1
            self.master_scale = np.float32(cfg.master_range_sigma * self.params["W"].std() / qmax)
            self.regrid_master(nearest=True)  # start from the nearest grid point; training uses SR

    # -- forward ------------------------------------------------------------------------
    def _quantize_weights(self):
        W = self.params["W"]
        if self.cfg.weights == "ternary":
            return ternary_quantize_absmean(W)
        if self.cfg.weights == "int8":
            amax = np.abs(W).max(axis=0, keepdims=True)
            s = np.where(amax > 0, amax / np.float32(INT8_MAX), np.float32(1)).astype(np.float32)
            return np.clip(np.rint(W / s), -INT8_MAX, INT8_MAX).astype(np.int8), s
        return None, None

    def _int_matmul(self, xq, Wq):
        """Exact integer product xq @ Wq. Ternary Wq can go through the C++ add/subtract kernel."""
        if self.cfg.backend == "native" and self.cfg.weights == "ternary":
            packed = lowbit_native.pack_ternary(np.ascontiguousarray(Wq.T))
            return lowbit_native.ternary_gemm_i32(xq, packed, Wq.shape[0]).astype(np.float32)
        return xq.astype(np.float32) @ Wq.astype(np.float32)  # |sum| < 2^24: exact in float32

    def forward(self, x):
        cfg = self.cfg
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        Wq, wscale = self._quantize_weights()
        self._Wq, self._wscale = Wq, wscale

        xq = None
        if cfg.act_bits:
            hit = self.share.get(x) if self.share is not None else None
            if hit is None:
                hit = quantize_int8_rows(x2)
                if self.share is not None:
                    self.share.put(x, hit)
            xq, sx = hit

        if Wq is not None and xq is not None:
            y = self._int_matmul(xq, Wq) * (sx * wscale)
        elif Wq is not None:
            y = (x2 @ Wq.astype(np.float32)) * wscale
        elif xq is not None:
            y = dequantize_int8_rows(xq, sx) @ self.params["W"]
        else:
            y = x2 @ self.params["W"]

        if cfg.cache == "int8" and xq is not None:
            self._cache = ("q8", xq, sx)  # the INT8 tensor the matmul already used: caching it is free
        else:
            self._cache = ("f32", x2)
        return y.reshape(*x.shape[:-1], -1)

    # -- backward -----------------------------------------------------------------------
    def backward(self, dy):
        cfg, rng = self.cfg, self.ctx.rng
        dy2 = dy.reshape(-1, dy.shape[-1]).astype(np.float32)
        kind = self._cache[0]
        x_hat = dequantize_int8_rows(self._cache[1], self._cache[2]) if kind == "q8" else self._cache[1]

        dW = x_hat.T @ dy2  # gradient w.r.t. the quantised weight; STE hands it to the latent weight
        if cfg.grad_bits:
            self.grad_bfp = bfp_quantize(dW, cfg.grad_block, cfg.grad_bits, cfg.rounding, rng)
            dW = self.grad_bfp.dequantize()
        self.grads["W"] = dW

        Wq, wscale = self._Wq, self._wscale
        if Wq is None:
            dx = dy2 @ self.params["W"].T
        elif cfg.quantize_dy:
            # dx[m, k] = sum_n dy[m, n] * s[n] * Wq[k, n]. A scalar scale (ternary gamma) factors out of the sum;
            # per-channel scales depend on n, so they are folded into the error signal before it is quantised.
            per_channel = np.ndim(wscale) > 0
            dyq, sdy = quantize_int8_rows(dy2 * wscale if per_channel else dy2, cfg.rounding, rng)
            dx = (dyq.astype(np.float32) @ Wq.T.astype(np.float32)) * (sdy if per_channel else sdy * wscale)
            dx = dx.astype(np.float32)
        else:
            dx = dy2 @ (Wq.astype(np.float32) * wscale).T
        self._cache = None
        return dx.reshape(*dy.shape[:-1], -1)

    # -- master weights on an INT8 grid ----------------------------------------------------
    def regrid_master(self, nearest=False):
        """Snap the latent weights onto the INT8 master grid after an optimiser step (SR keeps tiny updates alive)."""
        bits = self.cfg.master_bits
        mode = "nearest" if nearest else self.cfg.rounding
        q = quantize_to_grid(self.params["W"], self.master_scale, bits, mode, self.ctx.rng)
        self.params["W"][...] = q.astype(np.float32) * self.master_scale
        self.master_q = q


class CachedGELU(GELU):
    """GELU that keeps its input as INT8 for the backward pass and recomputes tanh there."""

    def __init__(self, cache="int8"):
        super().__init__()
        self.cache = cache

    def _tanh(self, x):
        return np.tanh(0.7978845608028654 * (x + 0.044715 * x**3))

    def forward(self, x):
        t = self._tanh(x)
        if self.cache == "int8":
            self._q = quantize_int8_rows(x.reshape(-1, x.shape[-1]))
            self._shape = x.shape
            self.x = None
        else:
            self.x = x
        return 0.5 * x * (1.0 + t)

    def backward(self, dy):
        x = dequantize_int8_rows(*self._q).reshape(self._shape) if self.cache == "int8" else self.x
        t = self._tanh(x)
        du = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)


class CachedRMSNorm(RMSNorm):
    """RMSNorm whose normalised activations (xhat) are kept as INT8 between forward and backward."""

    def __init__(self, d, eps, cache="int8"):
        super().__init__(d, eps)
        self.cache = cache

    def forward(self, x):
        y = super().forward(x)
        if self.cache == "int8":
            self._q = quantize_int8_rows(self.xhat.reshape(-1, self.xhat.shape[-1]))
            self._shape = self.xhat.shape
            self.xhat = None
        return y

    def backward(self, dy):
        if self.cache == "int8":
            self.xhat = dequantize_int8_rows(*self._q).reshape(self._shape)
        out = super().backward(dy)
        self.xhat = None
        return out


class CachedAttention(MultiHeadCausalAttention):
    """Attention that stores q, k, v (INT8 per head row) and the softmax matrix (UINT8 per row) compressed."""

    def forward(self, x):
        out = super().forward(x)
        self._qc = [quantize_int8_rows(a) for a in (self.q, self.k, self.v)]
        self._pc = quantize_uint8_rows(self.P)
        self.q = self.k = self.v = self.P = None
        return out

    def backward(self, dout):
        self.q, self.k, self.v = (dequantize_int8_rows(*c) for c in self._qc)
        self.P = self._pc[0].astype(np.float32) * self._pc[1]
        out = super().backward(dout)
        self.q = self.k = self.v = self.P = None
        return out


# ------------------------------------------------------------------------------ model surgery
class LowBitContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + 1000)


def convert_to_lowbit(model, cfg):
    """Swap the linear layers / norms / GELU / attention of a NumpyGPT for their low-bit versions in place.

    Embeddings, the tied output projection and the norm gains stay FP32 (the usual BitNet recipe).
    Weights are copied, so a converted model starts from exactly the FP32 model's initial point.
    """
    ctx = LowBitContext(cfg)
    model.lowbit_ctx = ctx
    mc = model.cfg
    for i in range(mc.n_layers):
        blk = model.children[f"blocks.{i}"]
        attn, mlp = blk.children["attn"], blk.children["mlp"]
        cache = cfg.cache

        share = _InputShare()  # q, k and v project the same normalised input
        for name in ("wq", "wk", "wv"):
            attn.children[name] = LowBitLinear(attn.children[name].params["W"], cfg, ctx, share)
        attn.children["wo"] = LowBitLinear(attn.children["wo"].params["W"], cfg, ctx)
        mlp.children["w1"] = LowBitLinear(mlp.children["w1"].params["W"], cfg, ctx)
        mlp.children["w2"] = LowBitLinear(mlp.children["w2"].params["W"], cfg, ctx)
        mlp.act = CachedGELU(cache)
        for name in ("attn_norm", "mlp_norm"):
            old = blk.children[name]
            new = CachedRMSNorm(mc.d_model, mc.norm_eps, cache)
            new.params["gamma"] = old.params["gamma"]
            blk.children[name] = new
        if cfg.cache == "int8" and cfg.cache_attention:
            blk.children["attn"] = _as_cached_attention(attn)
    return model


def _as_cached_attention(attn):
    attn.__class__ = CachedAttention  # same state, compressed cache; avoids re-initialising the weights
    return attn


def lowbit_layers(model):
    out = []
    for i in range(model.cfg.n_layers):
        blk = model.children[f"blocks.{i}"]
        out += [blk.children["attn"].children[n] for n in ("wq", "wk", "wv", "wo")]
        out += [blk.children["mlp"].children[n] for n in ("w1", "w2")]
    return [m for m in out if isinstance(m, LowBitLinear)]


def build_model(model_cfg, preset, seed=0):
    """NumpyGPT initialised from `seed`; `preset` is a name in PRESETS or a LowBitConfig (None = FP32)."""
    model = NumpyGPT(model_cfg, seed=seed)
    cfg = PRESETS[preset] if isinstance(preset, str) else preset
    if cfg is not None:
        convert_to_lowbit(model, replace(cfg, seed=seed))
    return model


# ------------------------------------------------------------------------------ memory accounting
def _root(a):
    return a.base if getattr(a, "base", None) is not None and isinstance(a.base, np.ndarray) else a


def _iter_arrays(obj, seen):
    """Yield every ndarray reachable from the (non-parameter) attributes of a layer, once per buffer."""
    for name, v in vars(obj).items():
        # _Wq / _wscale / master_q are the quantised *weight* copies (accounted in training_state_bytes), not activations
        if name in ("params", "grads", "children", "ctx", "cfg", "share", "rope", "loss_fn", "_Wq", "_wscale", "master_q"):
            continue
        stack = [v]
        while stack:
            item = stack.pop()
            if isinstance(item, np.ndarray):
                r = _root(item)
                if id(r) not in seen:
                    seen.add(id(r))
                    yield r
            elif isinstance(item, (tuple, list)):
                stack.extend(item)


def _modules(model):
    seen, stack = set(), [model]
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        yield m
        stack.extend(m.children.values())
        stack.extend(v for v in vars(m).values() if isinstance(v, Module))


def activation_cache_bytes(model):
    """Bytes of tensors retained after forward() for the backward pass (shared buffers counted once)."""
    seen, total = set(), 0
    for m in _modules(model):
        total += sum(a.nbytes for a in _iter_arrays(m, seen))
    probs = getattr(model.loss_fn, "probs", None)
    return total + (probs.nbytes if probs is not None else 0)


def training_state_bytes(model):
    """Bytes of long-lived training state, from the actual arrays: master weights, the packed/INT8 copy used
    by the forward pass, stored weight gradients and the AdamW moments (FP32 here)."""
    ll = lowbit_layers(model)
    ids = {id(l) for l in ll}
    master = sum(
        (l.params["W"].size * l.cfg.master_bits // 8) if l.cfg.master_bits else l.params["W"].nbytes for l in ll
    )
    forward_copy = 0
    for l in ll:
        K, N = l.params["W"].shape
        forward_copy += N * packed_row_bytes(K) if l.cfg.weights == "ternary" else (K * N if l.cfg.weights == "int8" else 0)
    grad = sum(l.grad_bfp.nbytes if l.grad_bfp is not None else l.params["W"].nbytes for l in ll)
    other_params = sum(
        p.nbytes for name, p in model.parameters().items() if not any(name.endswith(f"{t}.W") and "emb" not in name for t in ("wq", "wk", "wv", "wo", "w1", "w2"))
    )
    all_params = sum(p.nbytes for p in model.parameters().values())
    return {
        "master_weights": master + other_params,
        "forward_weights": forward_copy,
        "weight_gradients": grad + other_params,
        "optimizer_state": 2 * all_params,
        "quantised_layers": len(ids),
    }


# ------------------------------------------------------------------------------ training
@dataclass
class RunResult:
    name: str
    losses: list
    val_loss: float
    ms_per_step: float
    peak_bytes: int
    cache_bytes: int
    state: dict
    grad_cosine: float


def _cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def gradient_cosine(model_cfg, preset, batch, seed=0):
    """Cosine similarity between the low-bit model's weight gradients and the FP32 model's, same weights and batch."""
    ref = build_model(model_cfg, "fp32", seed)
    low = build_model(model_cfg, preset, seed)
    ids, tgt = batch
    ref.loss(ids, tgt)
    ref.backward()
    low.loss(ids, tgt)
    low.backward()
    gr, gl = ref.gradients(), low.gradients()
    names = [n for n in gr if n.endswith(".W")]
    return float(np.mean([_cosine(gr[n], gl[n]) for n in names]))


def train(model_cfg, preset, data, steps=200, batch_size=16, lr=3e-3, warmup=20, seed=0, measure_memory=True):
    """Train for `steps` AdamW steps; returns a RunResult. `data` = (train_tokens, val_tokens) byte arrays."""
    from data import get_batch
    from optimizers import AdamW, clip_grad_norm, cosine_lr

    train_tok, val_tok = data
    model = build_model(model_cfg, preset, seed)
    opt = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.1)
    rng = np.random.default_rng(seed + 1)
    layers = lowbit_layers(model)

    losses, times = [], []
    for step in range(steps):
        opt.lr = cosine_lr(step, lr, lr / 10, warmup, steps)
        x, y = get_batch(train_tok, batch_size, model_cfg.max_seq, rng)
        t0 = time.perf_counter()
        losses.append(model.loss(x, y))
        model.backward()
        clip_grad_norm(model.gradients(), 1.0)
        opt.step(model.gradients())
        for layer in layers:
            if layer.cfg.master_bits:
                layer.regrid_master()
        times.append(time.perf_counter() - t0)

    vrng = np.random.default_rng(123)
    val = float(np.mean([model.loss(*get_batch(val_tok, batch_size, model_cfg.max_seq, vrng)) for _ in range(8)]))

    peak = cache = 0
    state = {}
    if measure_memory:  # separate pass: tracemalloc slows everything down, so it is kept out of the timing
        x, y = get_batch(train_tok, batch_size, model_cfg.max_seq, np.random.default_rng(7))
        tracemalloc.start()
        tracemalloc.reset_peak()
        model.loss(x, y)
        cache = activation_cache_bytes(model)
        model.backward()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        state = training_state_bytes(model)

    cos = gradient_cosine(model_cfg, preset, get_batch(train_tok, 8, model_cfg.max_seq, np.random.default_rng(5)), seed) if preset != "fp32" else 1.0
    name = preset if isinstance(preset, str) else "custom"
    return RunResult(name, losses, val, 1000 * float(np.median(times)), peak, cache, state, cos)
