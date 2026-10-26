"""Rotary position embeddings and multi-head causal self-attention (NumPy, fwd + bwd)."""
import numpy as np

from nn_layers import Linear, Module, softmax, softmax_backward


class RoPE:
    """Rotates each (x[2i], x[2i+1]) pair at position p by angle p * theta^(-2i/d_head).

    A rotation is orthogonal, so backward is the same rotation by the opposite angle.
    """

    def __init__(self, head_dim, max_seq, theta=10000.0):
        inv_freq = theta ** (-np.arange(0, head_dim, 2) / head_dim)
        angles = np.outer(np.arange(max_seq), inv_freq)
        self.cos, self.sin = np.cos(angles), np.sin(angles)

    def _rotate(self, x, sign):
        T = x.shape[-2]
        c = self.cos[:T].astype(x.dtype)
        s = (sign * self.sin[:T]).astype(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
        return out

    def forward(self, x):
        return self._rotate(x, +1.0)

    def backward(self, dy):
        return self._rotate(dy, -1.0)


class MultiHeadCausalAttention(Module):
    def __init__(self, d_model, n_heads, max_seq, rng, rope_theta=10000.0,
                 out_std=None, dtype=np.float32):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        mk = lambda std=None: Linear(d_model, d_model, bias=False, rng=rng, std=std, dtype=dtype)
        self.children = {"wq": mk(0.02), "wk": mk(0.02), "wv": mk(0.02), "wo": mk(out_std or 0.02)}
        self.rope = RoPE(self.dh, max_seq, rope_theta)

    def _split(self, x):  # (B, T, d) -> (B, h, T, dh)
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)

    def _merge(self, x):  # (B, h, T, dh) -> (B, T, d)
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.h * self.dh)

    def forward(self, x):
        c = self.children
        T = x.shape[1]
        q = self.rope.forward(self._split(c["wq"].forward(x)))
        k = self.rope.forward(self._split(c["wk"].forward(x)))
        v = self._split(c["wv"].forward(x))
        self.scale = x.dtype.type(1.0 / np.sqrt(self.dh))  # same dtype as x: a float64 scalar would promote float32 activations
        scores = (q @ k.swapaxes(-1, -2)) * self.scale
        scores = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, scores)
        P = softmax(scores)  # masked entries are exactly 0
        self.q, self.k, self.v, self.P = q, k, v, P
        return c["wo"].forward(self._merge(P @ v))

    def backward(self, dout):
        c = self.children
        q, k, v, P = self.q, self.k, self.v, self.P
        dO = self._split(c["wo"].backward(dout))
        dV = P.swapaxes(-1, -2) @ dO
        dP = dO @ v.swapaxes(-1, -2)
        dS = softmax_backward(P, dP) * self.scale  # masked P == 0 => zero grad there
        dQ = self.rope.backward(dS @ k)
        dK = self.rope.backward(dS.swapaxes(-1, -2) @ q)
        return (c["wq"].backward(self._merge(dQ))
                + c["wk"].backward(self._merge(dK))
                + c["wv"].backward(self._merge(dV)))
