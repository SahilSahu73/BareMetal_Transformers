"""NumpyGPT and TorchGPT must be the same function: same weights in => same logits and gradients out."""
import numpy as np
import torch
import torch.nn.functional as F

from torch_model import TorchGPT
from transformer import Config, NumpyGPT

CFG = Config(vocab=23, d_model=32, n_layers=3, n_heads=4, max_seq=16)


def build_pair(seed=0):
    np_model = NumpyGPT(CFG, seed=seed, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for p in np_model.parameters().values():
        p[...] = rng.standard_normal(p.shape) * 0.2 if p.ndim > 1 else 1 + 0.1 * rng.standard_normal(p.shape)
    t_model = TorchGPT(CFG).double()
    state = {k: torch.tensor(v) for k, v in np_model.parameters().items()}
    t_model.load_state_dict(state, strict=True)  # strict => names and shapes are identical
    return np_model, t_model


def test_parameter_names_and_shapes_identical():
    np_model, t_model = build_pair()
    t_state = t_model.state_dict()
    assert set(np_model.parameters()) == set(t_state)
    for k, v in np_model.parameters().items():
        assert tuple(t_state[k].shape) == v.shape, k


def test_logits_match():
    np_model, t_model = build_pair()
    ids = np.random.default_rng(1).integers(0, CFG.vocab, size=(3, 12))
    ref = t_model(torch.tensor(ids)).detach().numpy()
    assert np.abs(np_model.forward(ids) - ref).max() < 1e-9


def test_loss_and_all_gradients_match():
    np_model, t_model = build_pair()
    rng = np.random.default_rng(2)
    ids = rng.integers(0, CFG.vocab, size=(2, 10))
    tgt = rng.integers(0, CFG.vocab, size=(2, 10))
    loss_t = F.cross_entropy(t_model(torch.tensor(ids)).flatten(0, 1), torch.tensor(tgt).flatten())
    loss_t.backward()
    loss = np_model.loss(ids, tgt)
    np_model.backward()
    assert abs(loss - loss_t.item()) < 1e-9
    grads = np_model.gradients()
    for k, p in t_model.named_parameters():
        assert np.abs(grads[k] - p.grad.numpy()).max() < 1e-8, k


def test_num_params_agree():
    np_model, t_model = build_pair()
    assert np_model.num_params() == t_model.num_params()


def test_small_config_param_count():
    from transformer import CONFIGS
    n = TorchGPT(CONFIGS["small"]).num_params()
    assert n == 256 * 512 + 6 * (12 * 512 * 512 + 2 * 512) + 512
