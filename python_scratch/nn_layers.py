"""Neural-network layers in pure NumPy with hand-derived backward passes.

Every layer follows the same contract:
    y  = layer.forward(x)     # caches whatever backward needs
    dx = layer.backward(dy)   # fills layer.grads[name] for each entry of layer.params
No autograd, no graph: the model calls backward() on its layers in reverse order.
"""
import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)  # shift for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(s, ds, axis=-1):
    """Given s = softmax(z) and ds = dL/ds, return dL/dz without building the Jacobian.

    J = diag(s) - s s^T  =>  dz = s * (ds - <ds, s>)
    """
    return s * (ds - (ds * s).sum(axis=axis, keepdims=True))


class Module:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.children = {}

    def parameters(self, prefix=""):
        """Flat {dotted.name: array}. Arrays are references, so updating them in place trains the model."""
        out = {prefix + k: v for k, v in self.params.items()}
        for name, child in self.children.items():
            out.update(child.parameters(f"{prefix}{name}."))
        return out

    def gradients(self, prefix=""):
        out = {prefix + k: v for k, v in self.grads.items()}
        for name, child in self.children.items():
            out.update(child.gradients(f"{prefix}{name}."))
        return out


class Linear(Module):
    """y = x W + b, with W stored as (in, out)."""

    def __init__(self, d_in, d_out, bias=True, rng=None, std=None, dtype=np.float32):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        std = std if std is not None else 1.0 / np.sqrt(d_in)
        self.params["W"] = (rng.standard_normal((d_in, d_out)) * std).astype(dtype)
        if bias:
            self.params["b"] = np.zeros(d_out, dtype=dtype)

    def forward(self, x):
        self.x = x
        y = x @ self.params["W"]
        if "b" in self.params:
            y = y + self.params["b"]
        return y

    def backward(self, dy):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.grads["W"] = x2.T @ dy2
        if "b" in self.params:
            self.grads["b"] = dy2.sum(axis=0)
        return dy @ self.params["W"].T


class Embedding(Module):
    def __init__(self, vocab, d, rng=None, std=0.02, dtype=np.float32):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        self.params["E"] = (rng.standard_normal((vocab, d)) * std).astype(dtype)

    def forward(self, ids):
        self.ids = ids
        return self.params["E"][ids]

    def backward(self, dy):
        # repeated tokens must accumulate, so scatter-add rather than assign
        dE = np.zeros_like(self.params["E"])
        np.add.at(dE, self.ids.reshape(-1), dy.reshape(-1, dy.shape[-1]))
        self.grads["E"] = dE
        return None


class LayerNorm(Module):
    """y = gamma * (x - mean) / sqrt(var + eps) + beta, over the last axis."""

    def __init__(self, d, eps=1e-5, dtype=np.float32):
        super().__init__()
        self.eps = eps
        self.params["gamma"] = np.ones(d, dtype=dtype)
        self.params["beta"] = np.zeros(d, dtype=dtype)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.inv_std
        return self.params["gamma"] * self.xhat + self.params["beta"]

    def backward(self, dy):
        flat = lambda a: a.reshape(-1, a.shape[-1])
        self.grads["gamma"] = (flat(dy) * flat(self.xhat)).sum(axis=0)
        self.grads["beta"] = flat(dy).sum(axis=0)
        g = dy * self.params["gamma"]
        # the two subtracted terms remove the components that normalisation
        # discards: shifting the mean and rescaling the variance
        return self.inv_std * (
            g
            - g.mean(axis=-1, keepdims=True)
            - self.xhat * (g * self.xhat).mean(axis=-1, keepdims=True)
        )


class RMSNorm(Module):
    """y = gamma * x / sqrt(mean(x^2) + eps). LayerNorm without the mean term."""

    def __init__(self, d, eps=1e-5, dtype=np.float32):
        super().__init__()
        self.eps = eps
        self.params["gamma"] = np.ones(d, dtype=dtype)

    def forward(self, x):
        self.inv_rms = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps)
        self.xhat = x * self.inv_rms
        return self.params["gamma"] * self.xhat

    def backward(self, dy):
        flat = lambda a: a.reshape(-1, a.shape[-1])
        self.grads["gamma"] = (flat(dy) * flat(self.xhat)).sum(axis=0)
        g = dy * self.params["gamma"]
        return self.inv_rms * (g - self.xhat * (g * self.xhat).mean(axis=-1, keepdims=True))


_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))  # python floats are "weak" scalars: float32 stays float32
_GELU_C = 0.044715


class GELU(Module):
    """tanh approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))."""

    def forward(self, x):
        self.x = x
        self.t = np.tanh(_SQRT_2_OVER_PI * (x + _GELU_C * x**3))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dy):
        x, t = self.x, self.t
        du = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_C * x * x)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)


class CrossEntropyLoss:
    """Fused softmax + negative log-likelihood, averaged over all tokens.

    The fused gradient is simply (softmax(z) - onehot(t)) / N, which is both
    cheaper and better conditioned than chaining softmax_backward by hand.
    """

    def forward(self, logits, targets):
        z = logits.reshape(-1, logits.shape[-1])
        t = targets.reshape(-1)
        z = z - z.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(z).sum(axis=-1))
        self.probs = np.exp(z - logsumexp[:, None])
        self.t = t
        self.shape = logits.shape
        return float((logsumexp - z[np.arange(len(t)), t]).mean())

    def backward(self):
        d = self.probs.copy()
        d[np.arange(len(self.t)), self.t] -= 1.0
        return (d / len(self.t)).reshape(self.shape)
