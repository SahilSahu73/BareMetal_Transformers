"""SGD+momentum, AdamW (decoupled weight decay), cosine LR with warmup, grad clipping.

Optimizers update the arrays of a {name: array} dict in place.
"""
import math

import numpy as np


class SGD:
    def __init__(self, params, lr=1e-2, momentum=0.9):
        self.params, self.lr, self.momentum = params, lr, momentum
        self.v = {k: np.zeros_like(p) for k, p in params.items()}

    def step(self, grads):
        for k, p in self.params.items():
            self.v[k] = self.momentum * self.v[k] + grads[k]
            p -= self.lr * self.v[k]


class AdamW:
    """Weight decay is applied to the weights directly, not folded into the gradient.

    Adding lambda*theta to the gradient (plain L2) would divide it by sqrt(v_hat), so
    parameters with large gradients would hardly decay. Decoupling avoids that.
    Only matrices (ndim >= 2) are decayed; norm gains are left alone.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params, self.lr, self.eps, self.wd = params, lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = {k: np.zeros_like(p) for k, p in params.items()}
        self.v = {k: np.zeros_like(p) for k, p in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            update = (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)
            if self.wd and p.ndim >= 2:
                p -= self.lr * self.wd * p
            p -= self.lr * update


def cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    """Linear warmup to max_lr, then cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def clip_grad_norm(grads, max_norm):
    """Scale all gradients in place so their global L2 norm is at most max_norm."""
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
    scale = max_norm / (total + 1e-6)
    if scale < 1.0:
        for g in grads.values():
            g *= scale
    return total
