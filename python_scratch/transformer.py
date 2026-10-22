"""Decoder-only Transformer (pre-norm, RMSNorm, RoPE, GELU MLP, tied embeddings) in NumPy,
with a hand-written backward pass through the whole network."""
from dataclasses import dataclass, asdict

import numpy as np

from attention import MultiHeadCausalAttention
from nn_layers import CrossEntropyLoss, Embedding, GELU, Linear, Module, RMSNorm, softmax


@dataclass
class Config:
    vocab: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    max_seq: int = 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def d_ff(self):
        return 4 * self.d_model

    def to_dict(self):
        return asdict(self)


CONFIGS = {
    "tiny": Config(d_model=64, n_layers=2, n_heads=4, max_seq=64),
    "small": Config(d_model=512, n_layers=6, n_heads=8, max_seq=256),
}


class MLP(Module):
    def __init__(self, d, d_ff, rng, out_std, dtype):
        super().__init__()
        self.children = {
            "w1": Linear(d, d_ff, bias=False, rng=rng, std=0.02, dtype=dtype),
            "w2": Linear(d_ff, d, bias=False, rng=rng, std=out_std, dtype=dtype),
        }
        self.act = GELU()

    def forward(self, x):
        return self.children["w2"].forward(self.act.forward(self.children["w1"].forward(x)))

    def backward(self, dy):
        return self.children["w1"].backward(self.act.backward(self.children["w2"].backward(dy)))


class Block(Module):
    def __init__(self, cfg, rng, out_std, dtype):
        super().__init__()
        d = cfg.d_model
        self.children = {
            "attn_norm": RMSNorm(d, cfg.norm_eps, dtype),
            "attn": MultiHeadCausalAttention(d, cfg.n_heads, cfg.max_seq, rng,
                                             cfg.rope_theta, out_std, dtype),
            "mlp_norm": RMSNorm(d, cfg.norm_eps, dtype),
            "mlp": MLP(d, cfg.d_ff, rng, out_std, dtype),
        }

    def forward(self, x):
        c = self.children
        x = x + c["attn"].forward(c["attn_norm"].forward(x))
        return x + c["mlp"].forward(c["mlp_norm"].forward(x))

    def backward(self, dy):
        # residual y = x + f(x)  =>  dx = dy + f'(x)^T dy
        c = self.children
        dy = dy + c["mlp_norm"].backward(c["mlp"].backward(dy))
        return dy + c["attn_norm"].backward(c["attn"].backward(dy))


class NumpyGPT(Module):
    def __init__(self, cfg, seed=0, dtype=np.float32):
        super().__init__()
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        out_std = 0.02 / np.sqrt(2 * cfg.n_layers)  # GPT-2 scaling of residual projections
        self.children = {"emb": Embedding(cfg.vocab, cfg.d_model, rng, 0.02, dtype)}
        for i in range(cfg.n_layers):
            self.children[f"blocks.{i}"] = Block(cfg, rng, out_std, dtype)
        self.children["final_norm"] = RMSNorm(cfg.d_model, cfg.norm_eps, dtype)
        self.loss_fn = CrossEntropyLoss()

    def _blocks(self):
        return [self.children[f"blocks.{i}"] for i in range(self.cfg.n_layers)]

    def forward(self, ids):
        x = self.children["emb"].forward(ids)
        for blk in self._blocks():
            x = blk.forward(x)
        self.h = self.children["final_norm"].forward(x)
        return self.h @ self.children["emb"].params["E"].T  # tied output projection

    def loss(self, ids, targets):
        return self.loss_fn.forward(self.forward(ids), targets)

    def backward(self):
        emb = self.children["emb"]
        dlogits = self.loss_fn.backward()  # (B, T, V)
        d2 = dlogits.reshape(-1, dlogits.shape[-1])
        dx = self.children["final_norm"].backward(dlogits @ emb.params["E"])
        for blk in reversed(self._blocks()):
            dx = blk.backward(dx)
        emb.backward(dx)
        # tied weights: E receives gradient from the lookup and from the output matmul
        emb.grads["E"] = emb.grads["E"] + d2.T @ self.h.reshape(-1, self.h.shape[-1])

    def generate(self, ids, n_new, temperature=1.0, rng=None):
        """Naive sampling (recomputes the full context each step; the C++ engine uses a KV-cache)."""
        rng = rng or np.random.default_rng(0)
        ids = list(ids)
        for _ in range(n_new):
            ctx = np.array([ids[-self.cfg.max_seq:]])
            logits = self.forward(ctx)[0, -1] / max(temperature, 1e-6)
            ids.append(int(rng.choice(len(logits), p=softmax(logits.astype(np.float64)))))
        return ids

    def num_params(self):
        return sum(p.size for p in self.parameters().values())
