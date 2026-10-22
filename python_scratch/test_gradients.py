"""Check every hand-written backward pass against PyTorch autograd (float64)
and against central finite differences (framework-independent)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nn_layers import (CrossEntropyLoss, Embedding, GELU, LayerNorm, Linear, RMSNorm,
                       softmax, softmax_backward)

TOL = 1e-6
rng = np.random.default_rng(1234)


def t64(a, grad=True):
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)


def maxdiff(a, b):
    return float(np.abs(a - b.detach().numpy()).max())


def finite_diff(f, x, idx, h=1e-6):
    """Central difference of scalar f w.r.t. x[idx] (x is mutated and restored)."""
    old = x[idx]
    x[idx] = old + h
    fp = f()
    x[idx] = old - h
    fm = f()
    x[idx] = old
    return (fp - fm) / (2 * h)


def check_layer(layer, torch_fn, x_np, torch_params):
    """Run forward/backward on both sides with a random upstream gradient."""
    x = t64(x_np)
    y_t = torch_fn(x)
    dy_np = rng.standard_normal(y_t.shape)
    y_t.backward(torch.tensor(dy_np))

    y = layer.forward(x_np)
    dx = layer.backward(dy_np)

    assert maxdiff(y, y_t) < TOL
    if dx is not None:
        assert maxdiff(dx, x.grad) < TOL
    for name, p in torch_params.items():
        assert maxdiff(layer.grads[name], p.grad) < TOL, name
    return dy_np


def test_linear():
    layer = Linear(7, 5, dtype=np.float64, rng=rng)
    layer.params["b"] = rng.standard_normal(5)
    W, b = t64(layer.params["W"]), t64(layer.params["b"])
    check_layer(layer, lambda x: x @ W + b, rng.standard_normal((3, 4, 7)), {"W": W, "b": b})


def test_linear_no_bias():
    layer = Linear(6, 3, bias=False, dtype=np.float64, rng=rng)
    W = t64(layer.params["W"])
    check_layer(layer, lambda x: x @ W, rng.standard_normal((2, 5, 6)), {"W": W})


def test_embedding_with_repeated_ids():
    layer = Embedding(11, 4, std=1.0, dtype=np.float64, rng=rng)
    ids = np.array([[1, 2, 2, 7], [7, 7, 0, 1]])  # repeats must accumulate
    E = t64(layer.params["E"])
    y_t = E[torch.tensor(ids)]
    dy = rng.standard_normal(y_t.shape)
    y_t.backward(torch.tensor(dy))
    assert maxdiff(layer.forward(ids), y_t) < TOL
    layer.backward(dy)
    assert maxdiff(layer.grads["E"], E.grad) < TOL


def test_layernorm():
    layer = LayerNorm(16, dtype=np.float64)
    layer.params["gamma"] = rng.standard_normal(16)
    layer.params["beta"] = rng.standard_normal(16)
    g, b = t64(layer.params["gamma"]), t64(layer.params["beta"])
    check_layer(layer, lambda x: F.layer_norm(x, (16,), g, b, eps=1e-5),
                rng.standard_normal((3, 5, 16)) * 3 + 2, {"gamma": g, "beta": b})


def test_rmsnorm():
    layer = RMSNorm(16, dtype=np.float64)
    layer.params["gamma"] = rng.standard_normal(16)
    g = t64(layer.params["gamma"])

    def torch_rms(x):
        return g * x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)

    check_layer(layer, torch_rms, rng.standard_normal((3, 5, 16)) * 3 + 2, {"gamma": g})


def test_gelu():
    check_layer(GELU(), lambda x: F.gelu(x, approximate="tanh"),
                rng.standard_normal((4, 9)) * 2, {})


def test_softmax_backward():
    z_np = rng.standard_normal((4, 6, 10))
    z = t64(z_np)
    s_t = torch.softmax(z, -1)
    ds = rng.standard_normal(s_t.shape)
    s_t.backward(torch.tensor(ds))
    s = softmax(z_np)
    assert maxdiff(s, s_t) < TOL
    assert maxdiff(softmax_backward(s, ds), z.grad) < TOL


def test_softmax_is_stable_for_large_logits():
    s = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(s).all() and abs(s.sum() - 1) < 1e-12


def test_cross_entropy():
    logits_np = rng.standard_normal((3, 7, 13))
    targets = rng.integers(0, 13, size=(3, 7))
    logits = t64(logits_np)
    loss_t = F.cross_entropy(logits.reshape(-1, 13), torch.tensor(targets).reshape(-1))
    loss_t.backward()
    ce = CrossEntropyLoss()
    loss = ce.forward(logits_np, targets)
    assert abs(loss - loss_t.item()) < TOL
    assert maxdiff(ce.backward(), logits.grad) < TOL


@pytest.mark.parametrize("make", [
    lambda: LayerNorm(8, dtype=np.float64),
    lambda: RMSNorm(8, dtype=np.float64),
    lambda: GELU(),
])
def test_finite_difference_agrees(make):
    """Independent of PyTorch: d(sum(w*y))/dx from backward() vs numerical."""
    layer = make()
    x = rng.standard_normal((2, 3, 8))
    w = rng.standard_normal((2, 3, 8))
    f = lambda: float((layer.forward(x) * w).sum())
    f()
    dx = layer.backward(w)
    for idx in [(0, 0, 0), (1, 2, 5), (0, 1, 7)]:
        assert abs(dx[idx] - finite_diff(f, x, idx)) < 1e-5
