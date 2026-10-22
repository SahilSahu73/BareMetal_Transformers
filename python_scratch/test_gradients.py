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


# ---------------------------------------------------------------- attention + RoPE
from attention import MultiHeadCausalAttention, RoPE  # noqa: E402
from transformer import Config, NumpyGPT  # noqa: E402


def torch_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def test_rope_matches_torch_and_backward_is_inverse_rotation():
    rope = RoPE(8, 16)
    x_np = rng.standard_normal((2, 3, 10, 8))
    cos, sin = torch.tensor(rope.cos[:10]), torch.tensor(rope.sin[:10])
    x = t64(x_np)
    y_t = torch_rope(x, cos, sin)
    dy = rng.standard_normal(y_t.shape)
    y_t.backward(torch.tensor(dy))
    assert maxdiff(rope.forward(x_np), y_t) < TOL
    assert maxdiff(rope.backward(dy), x.grad) < TOL
    assert np.allclose(rope.backward(rope.forward(x_np)), x_np)


def test_rope_scores_depend_only_on_relative_position():
    rope = RoPE(8, 32)
    q, k = rng.standard_normal(8), rng.standard_normal(8)
    qs = rope.forward(np.broadcast_to(q, (1, 32, 8)).copy())[0]
    ks = rope.forward(np.broadcast_to(k, (1, 32, 8)).copy())[0]
    assert np.isclose(qs[5] @ ks[2], qs[15] @ ks[12])  # same offset (3) at different positions


def test_causal_attention():
    d, h, T = 16, 4, 9
    attn = MultiHeadCausalAttention(d, h, 32, rng, dtype=np.float64)
    for lin in attn.children.values():
        lin.params["W"] = rng.standard_normal((d, d)) * 0.3
    Ws = {n: t64(attn.children[n].params["W"]) for n in ("wq", "wk", "wv", "wo")}
    cos, sin = torch.tensor(attn.rope.cos[:T]), torch.tensor(attn.rope.sin[:T])

    def torch_attn(x):
        B = x.shape[0]
        split = lambda a: a.reshape(B, T, h, d // h).transpose(1, 2)
        q, k, v = (split(x @ Ws[n]) for n in ("wq", "wk", "wv"))
        q, k = torch_rope(q, cos, sin), torch_rope(k, cos, sin)
        s = q @ k.transpose(-1, -2) / (d // h) ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        o = torch.softmax(s, -1) @ v
        return o.transpose(1, 2).reshape(B, T, d) @ Ws["wo"]

    params = {f"{n}.W": Ws[n] for n in Ws}
    layer_grads = {}

    class Flat:  # expose the nested layer grads under flat names for check_layer
        def forward(self, x): return attn.forward(x)
        def backward(self, dy):
            dx = attn.backward(dy)
            layer_grads.update(attn.gradients())
            return dx
        grads = layer_grads

    check_layer(Flat(), torch_attn, rng.standard_normal((2, T, d)), params)


def test_attention_is_causal():
    attn = MultiHeadCausalAttention(16, 4, 32, rng, dtype=np.float64)
    x = rng.standard_normal((1, 8, 16))
    y1 = attn.forward(x)
    x2 = x.copy()
    x2[:, 5:] += 10.0  # perturb the future
    y2 = attn.forward(x2)
    assert np.allclose(y1[:, :5], y2[:, :5])


# ---------------------------------------------------------------- full model
def tiny_cfg():
    return Config(vocab=11, d_model=16, n_layers=2, n_heads=2, max_seq=12)


def _randomised_model():
    model = NumpyGPT(tiny_cfg(), seed=3, dtype=np.float64)
    for p in model.parameters().values():  # larger weights => non-degenerate gradients
        p[...] = rng.standard_normal(p.shape) * 0.3 if p.ndim > 1 else 1 + 0.2 * rng.standard_normal(p.shape)
    return model


def test_full_model_gradients_by_finite_differences():
    model = _randomised_model()
    ids = rng.integers(0, 11, size=(2, 8))
    tgt = rng.integers(0, 11, size=(2, 8))
    model.loss(ids, tgt)
    model.backward()
    grads = model.gradients()
    params = model.parameters()
    assert set(grads) == set(params)
    for name, p in params.items():
        flat_idx = rng.choice(p.size, size=min(3, p.size), replace=False)
        for fi in flat_idx:
            idx = np.unravel_index(fi, p.shape)
            num = finite_diff(lambda: model.loss(ids, tgt), p, idx)
            assert abs(grads[name][idx] - num) < 1e-6, (name, idx, grads[name][idx], num)
