"""The binary format is the contract with the C++ engine: pin it down."""
import struct

import numpy as np

from eval_quant import fake_quantize_per_tensor
from export_weights import (ALIGN, FP32, HEADER_FMT, INT8, dequantize, fake_quantize, from_file_layout,
                            is_matrix, quantize_per_channel, read_model, tensor_names,
                            tensor_shape, write_model, write_reference)
from transformer import Config, NumpyGPT

CFG = Config(vocab=31, d_model=24, n_layers=2, n_heads=3, max_seq=16)


def params():
    m = NumpyGPT(CFG, seed=5, dtype=np.float32)
    rng = np.random.default_rng(5)
    return {k: (rng.standard_normal(v.shape) * (0.3 if v.ndim > 1 else 1.0)).astype(np.float32)
            for k, v in m.parameters().items()}


def test_tensor_order_covers_every_parameter_once():
    names = tensor_names(CFG.n_layers)
    assert sorted(names) == sorted(params())
    assert len(set(names)) == len(names)


def test_quantize_error_bound_and_range():
    w = np.random.default_rng(0).standard_normal((9, 40)).astype(np.float32)
    q, s = quantize_per_channel(w)
    assert q.dtype == np.int8 and np.abs(q).max() <= 127
    assert (np.abs(q).max(axis=1) == 127).all()  # each row uses the full range
    assert (np.abs(dequantize(q, s) - w) <= s[:, None] / 2 + 1e-7).all()


def test_zero_row_is_safe():
    q, s = quantize_per_channel(np.zeros((2, 8), np.float32))
    assert (q == 0).all() and np.isfinite(s).all()


def test_fp32_roundtrip_is_exact_and_header_correct(tmp_path):
    p = params()
    path = tmp_path / "m.bin"
    write_model(path, CFG, p, FP32)
    raw = path.read_bytes()
    assert struct.unpack_from(HEADER_FMT, raw, 0)[:4] == (b"BMTF", 1, FP32, CFG.vocab)
    assert len(raw) % ALIGN == 0
    cfg, dtype, tensors, quant = read_model(path)
    assert dtype == FP32 and not quant
    assert abs(cfg.norm_eps - CFG.norm_eps) < 1e-12  # stored as float32
    assert (cfg.vocab, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_seq, cfg.rope_theta) == \
           (CFG.vocab, CFG.d_model, CFG.n_layers, CFG.n_heads, CFG.max_seq, CFG.rope_theta)
    back = from_file_layout(tensors)
    for k, v in p.items():
        assert np.array_equal(back[k], v), k
        assert tensors[k].shape == tensor_shape(CFG, k)


def test_int8_roundtrip_matches_fake_quantize(tmp_path):
    p = params()
    path = tmp_path / "m8.bin"
    write_model(path, CFG, p, INT8)
    _, dtype, tensors, quant = read_model(path)
    assert dtype == INT8
    assert set(quant) == {n for n in tensor_names(CFG.n_layers) if is_matrix(n)}
    back, fq = from_file_layout(tensors), fake_quantize(p, CFG)
    for k in p:
        assert np.array_equal(back[k], fq[k]), k
        if not is_matrix(k):
            assert np.array_equal(back[k], p[k])  # norm gains stay FP32


def test_int8_file_is_about_4x_smaller(tmp_path):
    p = params()
    write_model(tmp_path / "a.bin", CFG, p, FP32)
    write_model(tmp_path / "b.bin", CFG, p, INT8)
    ratio = (tmp_path / "a.bin").stat().st_size / (tmp_path / "b.bin").stat().st_size
    assert 3.0 < ratio < 4.0


def test_every_tensor_is_64_byte_aligned(tmp_path):
    """Recompute offsets the way the C++ loader will, and check data really lives there."""
    p = params()
    path = tmp_path / "m.bin"
    write_model(path, CFG, p, INT8)
    raw = path.read_bytes()
    pos = ALIGN
    for name in tensor_names(CFG.n_layers):
        pos += -pos % ALIGN
        shape = tensor_shape(CFG, name)
        n = int(np.prod(shape))
        assert pos % ALIGN == 0
        if is_matrix(name):
            scales = np.frombuffer(raw, np.float32, shape[0], pos)
            assert (scales > 0).all()
            pos += 4 * shape[0]
            pos += -pos % ALIGN
            pos += n
        else:
            pos += 4 * n
    assert pos + (-pos % ALIGN) == len(raw)


def test_reference_file_layout(tmp_path):
    ids = np.arange(5)
    a, b = np.random.rand(5, CFG.vocab), np.random.rand(5, CFG.vocab)
    path = tmp_path / "r.bin"
    write_reference(path, ids, a, b)
    raw = path.read_bytes()
    assert struct.unpack_from("<4sII", raw, 0) == (b"BMRF", 5, CFG.vocab)
    off = 12
    assert np.array_equal(np.frombuffer(raw, np.int32, 5, off), ids)
    off += 20
    assert np.allclose(np.frombuffer(raw, np.float32, 5 * CFG.vocab, off).reshape(5, -1), a, atol=1e-7)
    off += 4 * a.size
    assert np.allclose(np.frombuffer(raw, np.float32, 5 * CFG.vocab, off).reshape(5, -1), b, atol=1e-7)


def test_per_channel_beats_per_tensor_when_one_row_is_an_outlier():
    p = params()
    p["blocks.0.attn.wq.W"][:, 0] *= 50.0  # one output channel (column in (in,out) layout) is huge
    err = lambda q: np.abs(q["blocks.0.attn.wq.W"] - p["blocks.0.attn.wq.W"]).mean()
    assert err(fake_quantize(p, CFG)) < err(fake_quantize_per_tensor(p, CFG))
