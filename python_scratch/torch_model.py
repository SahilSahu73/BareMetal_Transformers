"""PyTorch twin of NumpyGPT: same architecture, same parameter names and layouts.

Linear weights are stored as (in, out) exactly like the NumPy version, so a state dict
moves between the two implementations without transposes. Training the larger config
here is just faster (BLAS/GPU + autograd); nothing about the maths differs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Config


class Linear(nn.Module):
    def __init__(self, d_in, d_out, std):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_in, d_out) * std)

    def forward(self, x):
        return x @ self.W


class Embedding(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.E = nn.Parameter(torch.randn(vocab, d) * 0.02)

    def forward(self, ids):
        return self.E[ids]


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.gamma * x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)


def rope_tables(head_dim, max_seq, theta):
    inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    ang = torch.outer(torch.arange(max_seq, dtype=torch.float64), inv_freq)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: (B, h, T, dh), interleaved pairs
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg, out_std):
        super().__init__()
        d = cfg.d_model
        self.h, self.dh = cfg.n_heads, d // cfg.n_heads
        self.wq, self.wk, self.wv = (Linear(d, d, 0.02) for _ in range(3))
        self.wo = Linear(d, d, out_std)
        cos, sin = rope_tables(self.dh, cfg.max_seq, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, d = x.shape
        split = lambda a: a.view(B, T, self.h, self.dh).transpose(1, 2)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        q = apply_rope(split(self.wq(x)), cos, sin)
        k = apply_rope(split(self.wk(x)), cos, sin)
        v = split(self.wv(x))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, d))


class MLP(nn.Module):
    def __init__(self, cfg, out_std):
        super().__init__()
        self.w1 = Linear(cfg.d_model, cfg.d_ff, 0.02)
        self.w2 = Linear(cfg.d_ff, cfg.d_model, out_std)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg, out_std):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, out_std)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg, out_std)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TorchGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        out_std = 0.02 / (2 * cfg.n_layers) ** 0.5
        self.emb = Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, out_std) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, ids):
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x) @ self.emb.E.T  # tied output projection

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
