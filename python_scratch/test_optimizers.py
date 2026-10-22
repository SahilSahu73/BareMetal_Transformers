"""Our optimizers and LR schedule vs. torch.optim, step by step."""
import numpy as np
import torch

from optimizers import SGD, AdamW, clip_grad_norm, cosine_lr

rng = np.random.default_rng(7)


def make():
    w_np = {"w": rng.standard_normal((4, 3)), "m": rng.standard_normal((5, 2))}
    w_t = {k: torch.tensor(v.copy(), requires_grad=True) for k, v in w_np.items()}
    return w_np, w_t


def run(opt_np, opt_t, w_np, w_t, steps=25):
    for _ in range(steps):
        grads = {k: rng.standard_normal(v.shape) for k, v in w_np.items()}
        for k, p in w_t.items():
            p.grad = torch.tensor(grads[k])
        opt_np.step(grads)
        opt_t.step()
    for k in w_np:
        assert np.abs(w_np[k] - w_t[k].detach().numpy()).max() < 1e-10


def test_sgd_momentum():
    w_np, w_t = make()
    run(SGD(w_np, lr=0.05, momentum=0.9), torch.optim.SGD(w_t.values(), lr=0.05, momentum=0.9), w_np, w_t)


def test_adamw():
    w_np, w_t = make()
    run(AdamW(w_np, lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1),
        torch.optim.AdamW(w_t.values(), lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1),
        w_np, w_t)


def test_adamw_decoupled_decay_ignores_gradient_scale():
    """With zero gradient, a decayed weight must shrink by exactly (1 - lr*wd) per step."""
    p = {"w": np.ones((2, 2))}
    opt = AdamW(p, lr=0.1, weight_decay=0.5)
    opt.step({"w": np.zeros((2, 2))})
    assert np.allclose(p["w"], 1 - 0.1 * 0.5)


def test_cosine_schedule_matches_torch_lambda_lr():
    import math
    max_lr, min_lr, warm, total = 1e-3, 1e-4, 10, 100
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=max_lr)

    def lam(s):
        if s < warm:
            return (s + 1) / warm
        prog = (s - warm) / (total - warm)
        return (min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * prog))) / max_lr

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    for s in range(total):
        assert abs(opt.param_groups[0]["lr"] - cosine_lr(s, max_lr, min_lr, warm, total)) < 1e-12
        opt.step()
        sched.step()


def test_clip_grad_norm_matches_torch():
    g_np = {"a": rng.standard_normal((3, 4)), "b": rng.standard_normal(5)}
    ps = [torch.nn.Parameter(torch.zeros_like(torch.tensor(v))) for v in g_np.values()]
    for p, v in zip(ps, g_np.values()):
        p.grad = torch.tensor(v.copy())
    tn = torch.nn.utils.clip_grad_norm_(ps, 1.0).item()
    n = clip_grad_norm(g_np, 1.0)
    assert abs(n - tn) < 1e-9
    for p, v in zip(ps, g_np.values()):
        assert np.abs(p.grad.numpy() - v).max() < 1e-6
