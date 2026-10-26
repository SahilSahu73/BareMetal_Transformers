"""Low-precision edge training: ternary bit-packing, stochastic rounding, INT8 activation cache,
BFP gradients, and convergence of the whole loop against an FP32 baseline.

Tests that need the C++ library (cpp_engine/build/libbmlowbit.so) are skipped when it is not built.
"""
import itertools
import math

import numpy as np
import pytest

import low_bit_training as L
import lowbit_native
from nn_layers import Linear
from transformer import CONFIGS, Config

needs_native = pytest.mark.skipif(not lowbit_native.available(), reason="libbmlowbit not built (cmake --build cpp_engine/build --target bmlowbit)")
rng = np.random.default_rng(2024)

TEXT = (
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles "
    "and by opposing end them. To die: to sleep; no more; and by a sleep to say we end "
    "the heart-ache and the thousand natural shocks that flesh is heir to. "
) * 40


def tiny_data():
    ids = np.frombuffer(TEXT.encode(), dtype=np.uint8).astype(np.int64)
    split = int(len(ids) * 0.9)
    return ids[:split], ids[split:]


TINY = Config(vocab=256, d_model=32, n_layers=2, n_heads=2, max_seq=32)


# --------------------------------------------------------------------- 1. exact ternary bit packing
def test_four_weights_share_one_byte_with_exact_layout():
    # byte = w0 | w1<<2 | w2<<4 | w3<<6, codes 0 -> 00, +1 -> 01, -1 -> 10
    w = np.array([[1, -1, 0, 1]], dtype=np.int8)
    packed = L.pack_ternary(w)
    assert packed[0, 0] == 1 | (2 << 2) | (0 << 4) | (1 << 6) == 0x49
    assert packed.dtype == np.uint8
    assert packed.shape == (1, 8)                 # rows are padded to 32 weights = 8 bytes
    assert not packed[0, 1:].any()                # padding bytes hold zero codes


def test_every_one_of_the_81_weight_quadruples_packs_to_a_distinct_exact_byte():
    seen = {}
    for quad in itertools.product((-1, 0, 1), repeat=4):
        row = np.zeros((1, 32), dtype=np.int8)
        row[0, :4] = quad
        byte = int(L.pack_ternary(row)[0, 0])
        expected = sum(({0: 0, 1: 1, -1: 2}[w]) << (2 * j) for j, w in enumerate(quad))
        assert byte == expected
        assert byte not in seen.values()
        seen[quad] = byte
        assert np.array_equal(L.unpack_ternary(np.array([[byte] + [0] * 7], dtype=np.uint8), 4)[0], quad)
    assert len(seen) == 81


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 31, 32, 33, 100, 257])
def test_pack_unpack_roundtrip_is_lossless(K):
    w = rng.integers(-1, 2, size=(7, K)).astype(np.int8)
    packed = L.pack_ternary(w)
    assert packed.shape == (7, L.packed_row_bytes(K))
    assert np.array_equal(L.unpack_ternary(packed, K), w)


def test_two_bits_per_weight_is_16x_smaller_than_fp32():
    K, N = 2048, 512
    w = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
    assert L.pack_ternary(w).nbytes * 16 == w.size * 4  # FP32 bytes / packed bytes = 16 exactly (K multiple of 32)


def test_reserved_code_decodes_as_zero():
    row = np.array([[0b11_10_01_00] + [0] * 7], dtype=np.uint8)  # codes 0, +1, -1, reserved
    assert list(L.unpack_ternary(row, 4)[0]) == [0, 1, -1, 0]


def test_absmean_ternary_quantiser():
    W = rng.standard_normal((64, 48)).astype(np.float32)
    q, gamma = L.ternary_quantize_absmean(W)
    assert set(np.unique(q)) <= {-1, 0, 1}
    assert abs(gamma - np.abs(W).mean()) < 1e-6
    assert np.array_equal(q, np.clip(np.rint(W / gamma), -1, 1))
    # the ternary approximation is far better than a random sign pattern and keeps the dominant structure
    assert np.corrcoef((q * gamma).ravel(), W.ravel())[0, 1] > 0.8
    zq, zg = L.ternary_quantize_absmean(np.zeros((4, 4), np.float32))
    assert zg == 0 and not zq.any()


@needs_native
def test_cpp_and_python_pack_the_same_bytes():
    for K in (1, 5, 32, 33, 100, 257):
        w = rng.integers(-1, 2, size=(9, K)).astype(np.int8)
        assert np.array_equal(lowbit_native.pack_ternary(w), L.pack_ternary(w))
        assert np.array_equal(lowbit_native.unpack_ternary(L.pack_ternary(w), K), w)


@needs_native
def test_cpp_and_python_absmean_agree_exactly():
    W = rng.standard_normal((50, 70)).astype(np.float32)
    qc, gc = lowbit_native.ternary_quantize_absmean(W)
    qp, gp = L.ternary_quantize_absmean(W)
    assert np.array_equal(qc, qp) and abs(float(gc) - float(gp)) < 1e-7


@needs_native
def test_multiplier_less_cpp_kernel_computes_the_exact_integer_product():
    for M, N, K in [(1, 5, 31), (3, 17, 64), (70, 9, 257), (4, 8, 2048)]:
        w = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
        x = rng.integers(-127, 128, size=(M, K)).astype(np.int8)
        got = lowbit_native.ternary_gemm_i32(x, L.pack_ternary(w), K)
        assert np.array_equal(got, x.astype(np.int64) @ w.T.astype(np.int64))


# --------------------------------------------------------------------- 2. stochastic rounding
@pytest.mark.parametrize("x", [0.1, 0.5, 0.93, 2.3, -2.3, -0.25, 41.75, 1e-3])
def test_stochastic_rounding_expectation_equals_the_input(x):
    n = 200_000
    out = L.stochastic_round(np.full(n, x, dtype=np.float32), np.random.default_rng(7))
    assert set(np.unique(out)) <= {math.floor(x), math.ceil(x)}
    frac = x - math.floor(x)
    sigma = math.sqrt(frac * (1 - frac) / n)
    assert abs(out.mean() - x) < 5 * sigma + 1e-5                    # unbiased, to 5 standard errors
    assert abs((out == math.ceil(x)).mean() - frac) < 5 * sigma + 1e-5  # P(ceil) = x - floor(x)


def test_stochastic_rounding_leaves_integers_alone():
    ints = np.arange(-5, 6, dtype=np.float32)
    assert np.array_equal(L.stochastic_round(ints, np.random.default_rng(0)), ints)


def test_round_to_nearest_is_biased_where_stochastic_rounding_is_not():
    x = np.full(100_000, 0.3, dtype=np.float32)
    assert np.rint(x).mean() == 0.0                                   # every 0.3 vanishes
    assert abs(L.stochastic_round(x, np.random.default_rng(1)).mean() - 0.3) < 0.01


def test_updates_below_one_lsb_accumulate_only_with_stochastic_rounding():
    steps, n = 1000, 2000
    w_sr = np.zeros(n, np.float32)
    w_near = np.zeros(n, np.float32)
    r = np.random.default_rng(3)
    for _ in range(steps):                                            # a 0.05-LSB update, every step
        w_sr = L.stochastic_round(w_sr + np.float32(0.05), r)
        w_near = np.rint(w_near + np.float32(0.05))
    assert w_near.max() == 0.0                                        # round-to-nearest: training stalls
    assert abs(w_sr.mean() - 0.05 * steps) < 1.5                      # SR: ~50 LSB of progress


def test_bfp_stochastic_rounding_is_unbiased_while_nearest_keeps_its_error():
    g = (np.random.default_rng(5).standard_normal(256) * 1e-3).astype(np.float32)
    r = np.random.default_rng(6)
    mean_sr = np.mean([L.bfp_quantize(g, 32, 4, "stochastic", r).dequantize() for _ in range(3000)], axis=0)
    near = L.bfp_quantize(g, 32, 4, "nearest").dequantize()
    assert np.abs(mean_sr - g).sum() < 0.05 * np.abs(g).sum()
    assert np.abs(near - g).sum() > 3 * np.abs(mean_sr - g).sum()


@needs_native
def test_cpp_stochastic_rounding_is_unbiased_and_deterministic():
    x = np.full(300_000, 0.37, dtype=np.float32)
    a, b = lowbit_native.stochastic_round(x, 11), lowbit_native.stochastic_round(x, 11)
    assert np.array_equal(a, b) and not np.array_equal(a, lowbit_native.stochastic_round(x, 12))
    assert set(np.unique(a)) == {0.0, 1.0}
    assert abs(a.mean() - 0.37) < 5 * math.sqrt(0.37 * 0.63 / x.size)


# --------------------------------------------------------------------- block floating point gradients
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_bfp_mantissa_range_exponent_and_error_bound(bits, rounding):
    x = np.random.default_rng(9).standard_normal(1000).astype(np.float32)
    x[::97] *= 40
    t = L.bfp_quantize(x, 32, bits, rounding, np.random.default_rng(1))
    qmax = (1 << (bits - 1)) - 1
    assert np.abs(t.mant).max() <= qmax
    step = np.repeat(np.ldexp(1.0, t.exp.astype(int)), 32)[:x.size]
    assert np.all(np.abs(t.dequantize() - x) <= (0.5 if rounding == "nearest" else 1.0) * step + 1e-6)
    blocks = np.pad(x, (0, -x.size % 32)).reshape(-1, 32)
    amax = np.abs(blocks).max(1)
    e = t.exp.astype(int)
    assert np.all(amax / np.ldexp(1.0, e) <= qmax * (1 + 1e-6))       # the shared exponent fits the block ...
    assert np.all(amax / np.ldexp(1.0, e - 1) > qmax)                 # ... and is the smallest one that does


def test_bfp_exponent_is_exact_at_power_of_two_boundaries():
    # max|x| / qmax is exactly a power of two here, where a naive ceil(log2()) is easiest to get wrong
    for amax, exp in [(127.0, 0), (254.0, 1), (63.5, -1), (1.0 * 127 / 1024, -10)]:
        t = L.bfp_quantize(np.array([amax, 0.0, -amax / 2, 0.0], np.float32), 4, 8, "nearest")
        assert t.exp[0] == exp and t.mant[0] == 127
        assert t.dequantize()[0] == amax
    # hand-computed block: scale 2^-6; ties round half to even (0.5 -> 0, 1.5 -> 2)
    t = L.bfp_quantize(np.array([1.27, -0.5, 0.0078125, 0.0234375], np.float32), 4, 8, "nearest")
    assert t.exp[0] == -6 and list(t.mant) == [81, -32, 0, 2]
    z = L.bfp_quantize(np.zeros(8, np.float32), 4, 8, "nearest")
    assert not z.mant.any() and not z.exp.any()


@needs_native
@pytest.mark.parametrize("bits", [4, 8])
def test_cpp_and_python_bfp_nearest_are_bit_identical(bits):
    x = np.random.default_rng(10).standard_normal(1003).astype(np.float32) * 3
    t = L.bfp_quantize(x, 32, bits, "nearest")
    mant, exps = lowbit_native.bfp_quantize(x, 32, bits, stochastic=False)
    assert np.array_equal(mant, t.mant) and np.array_equal(exps, t.exp)
    assert np.array_equal(lowbit_native.bfp_dequantize(mant, exps, 32).reshape(x.shape), t.dequantize())


# --------------------------------------------------------------------- STE, INT8 cache, layers
def _ctx(**kw):
    return L.LowBitContext(L.LowBitConfig(**kw))


def test_ste_gradient_is_the_dense_gradient_applied_to_the_latent_weights():
    cfg = L.LowBitConfig(weights="ternary", act_bits=0, cache="fp32", grad_bits=0, quantize_dy=False)
    W = rng.standard_normal((24, 10)).astype(np.float32)
    layer = L.LowBitLinear(W, cfg, L.LowBitContext(cfg))
    x = rng.standard_normal((5, 24)).astype(np.float32)
    dy = rng.standard_normal((5, 10)).astype(np.float32)
    Wq, gamma = L.ternary_quantize_absmean(W)
    y = layer.forward(x)
    dx = layer.backward(dy)
    assert np.allclose(y, x @ (Wq * gamma), atol=1e-5)                 # forward uses the ternary weights ...
    assert np.allclose(layer.grads["W"], x.T @ dy, atol=1e-5)          # ... backward ignores the quantiser (STE)
    assert np.allclose(dx, dy @ (Wq * gamma).T, atol=1e-5)             # error signal flows through quantised weights


def test_int8_activation_cache_is_dequantised_on_the_fly_with_bounded_error():
    cfg = L.LowBitConfig(weights="ternary", grad_bits=0, quantize_dy=False)
    W = rng.standard_normal((64, 16)).astype(np.float32)
    layer = L.LowBitLinear(W, cfg, L.LowBitContext(cfg))
    x = rng.standard_normal((30, 64)).astype(np.float32)
    layer.forward(x)
    kind, xq, sx = layer._cache
    assert kind == "q8" and xq.dtype == np.int8
    assert xq.nbytes + sx.nbytes < 0.27 * x.nbytes                      # ~4x smaller (scales add a little)
    assert np.all(np.abs(L.dequantize_int8_rows(xq, sx) - x) <= sx / 2 + 1e-6)
    dy = rng.standard_normal((30, 16)).astype(np.float32)
    layer.backward(dy)
    ref = x.T @ dy                                                      # gradient computed from the exact input
    assert np.linalg.norm(layer.grads["W"] - ref) / np.linalg.norm(ref) < 0.02


def test_activations_shared_by_q_k_v_are_cached_once():
    model = L.build_model(TINY, "ternary", 0)
    ids = rng.integers(0, 256, size=(2, 16))
    model.loss(ids, ids)
    attn = model.children["blocks.0"].children["attn"]
    caches = [attn.children[n]._cache[1] for n in ("wq", "wk", "wv")]
    assert caches[0] is caches[1] is caches[2]


def test_full_model_activation_cache_shrinks_about_4x_on_the_tensors_it_covers():
    ids = rng.integers(0, 256, size=(8, 32))
    sizes = {}
    for name in ("fp32", "ternary"):
        m = L.build_model(TINY, name, 0)
        m.loss(ids, ids)
        # the logits / loss buffers (vocab-sized) are not compressed; compare the rest
        sizes[name] = L.activation_cache_bytes(m) - m.loss_fn.probs.nbytes - m.h.nbytes
        m.backward()
    assert sizes["fp32"] / sizes["ternary"] > 3.5


def test_model_computes_in_float32_not_silently_in_float64():
    for preset in ("fp32", "ternary"):
        m = L.build_model(TINY, preset, 0)
        ids = rng.integers(0, 256, size=(2, 16))
        assert m.forward(ids).dtype == np.float32
        m.loss(ids, ids)
        m.backward()
        assert all(g.dtype == np.float32 for g in m.gradients().values())


def test_converted_model_starts_from_the_same_weights_as_fp32():
    a, b = L.build_model(TINY, "fp32", 3), L.build_model(TINY, "ternary", 3)
    pa, pb = a.parameters(), b.parameters()
    assert set(pa) == set(pb)
    assert all(np.array_equal(pa[k], pb[k]) for k in pa)


def test_int8_master_weights_stay_on_the_grid_and_sr_keeps_tiny_updates():
    cfg = L.LowBitConfig(weights="ternary", master_bits=8)
    W = rng.standard_normal((32, 32)).astype(np.float32) * 0.05
    layer = L.LowBitLinear(W, cfg, L.LowBitContext(cfg))
    s = layer.master_scale
    assert np.allclose(layer.params["W"] / s, np.rint(layer.params["W"] / s), atol=1e-4)
    start = layer.params["W"].copy()
    for _ in range(400):                                    # drift of 0.02 LSB per step in every weight
        layer.params["W"] += 0.02 * s
        layer.regrid_master()
    on_grid = layer.params["W"] / s
    assert np.allclose(on_grid, np.rint(on_grid), atol=1e-4)
    moved = ((layer.params["W"] - start) / s).mean()
    assert abs(moved - 400 * 0.02) < 0.5                    # ~8 LSB gained; round-to-nearest would give exactly 0


@needs_native
def test_native_backend_matches_numpy_backend_exactly():
    W = rng.standard_normal((70, 33)).astype(np.float32)
    x = rng.standard_normal((9, 70)).astype(np.float32)
    outs = []
    for backend in ("numpy", "native"):
        cfg = L.LowBitConfig(weights="ternary", backend=backend)
        outs.append(L.LowBitLinear(W, cfg, L.LowBitContext(cfg)).forward(x))
    assert np.array_equal(outs[0], outs[1])


def test_gradients_of_the_lowbit_model_point_the_same_way_as_fp32():
    ids = rng.integers(0, 256, size=(8, 32))
    assert L.gradient_cosine(TINY, "int8", (ids, ids)) > 0.95
    assert L.gradient_cosine(TINY, "ternary", (ids, ids)) > 0.5       # 2-bit weights change the function itself


# --------------------------------------------------------------------- 3. convergence vs FP32
@pytest.fixture(scope="module")
def runs():
    data = tiny_data()
    return {p: L.train(TINY, p, data, steps=120, batch_size=8, lr=3e-3, warmup=10, measure_memory=False)
            for p in ("fp32", "int8", "ternary", "ternary_int8_master")}


def test_every_precision_trains_without_diverging(runs):
    for name, r in runs.items():
        assert np.all(np.isfinite(r.losses)), name
        assert max(r.losses[10:]) < 1.2 * r.losses[0], name            # no blow-up after warmup
        assert np.mean(r.losses[-10:]) < 0.55 * r.losses[0], name       # loss more than halves


def test_ternary_stochastic_rounding_training_converges_close_to_fp32(runs):
    fp32 = np.mean(runs["fp32"].losses[-10:])
    for name in ("ternary", "ternary_int8_master"):
        low = np.mean(runs[name].losses[-10:])
        assert low < fp32 + 0.5, (name, low, fp32)                      # within half a nat of the FP32 baseline
        assert runs[name].val_loss < fp32 + 0.5


def test_int8_training_matches_fp32(runs):
    assert abs(np.mean(runs["int8"].losses[-10:]) - np.mean(runs["fp32"].losses[-10:])) < 0.15


def test_low_precision_loss_curve_decreases_steadily(runs):
    losses = np.array(runs["ternary"].losses)
    windows = losses[: len(losses) // 20 * 20].reshape(-1, 20).mean(axis=1)
    assert np.all(np.diff(windows) < 0.02)                             # each 20-step window is no worse than the last


def test_training_is_reproducible_for_a_fixed_seed():
    data = tiny_data()
    a = L.train(TINY, "ternary", data, steps=15, batch_size=4, measure_memory=False)
    b = L.train(TINY, "ternary", data, steps=15, batch_size=4, measure_memory=False)
    assert a.losses == b.losses


# --------------------------------------------------------------------- hardware profiler model
import hardware_profiler as H  # noqa: E402


@pytest.mark.parametrize("batch,ctx", [(2, 32), (4, 32)])
@pytest.mark.parametrize("preset", ["fp32", "ternary", "int8"])
def test_analytical_activation_memory_matches_the_real_training_loop(preset, batch, ctx):
    from dataclasses import replace
    cfg = replace(TINY, max_seq=ctx)
    m = L.build_model(cfg, preset, 0)
    ids = rng.integers(0, 256, size=(batch, ctx))
    m.loss(ids, ids)
    measured = L.activation_cache_bytes(m)
    p = next(p for p in H.PRECISIONS if p.train_preset == preset)
    modelled = H.footprint(cfg, batch, ctx, p)["cached activations"]
    assert abs(modelled / measured - 1) < 0.03, (modelled, measured)


def test_ternary_pe_array_has_no_multiplier_or_dsp_at_every_size():
    for dim in (16, 32, 64):
        c = H.pe_costs(dim)
        assert c["ternary"]["multipliers"] == 0 and c["ternary"]["DSP"] == 0
        assert c["ternary"]["add/sub units"] == dim * dim
        assert c["fp32"]["multipliers"] == c["int8"]["multipliers"] == dim * dim
        assert c["fp32"]["DSP"] > c["int8"]["DSP"] > c["ternary"]["DSP"]


def test_traffic_model_orders_the_configurations_and_conserves_the_parts():
    cfg = CONFIGS["small"]
    hw = H.Hardware(pe=32)
    t = {p.name: H.traffic_per_step(cfg, 4, 256, p, hw) for p in H.PRECISIONS}
    for v in t.values():
        assert abs(v["total"] - (v["weights"] + v["activations"] + v["gradients"] + v["optimizer"])) < 1
    assert t["fp32"]["total"] > t["int8"]["total"] > t["ternary_m8"]["total"]
    assert t["ternary"]["weights"] < t["int8"]["weights"]                     # the 2-bit copy is the cheapest to stream
    assert t["fp32"]["activations"] / t["ternary"]["activations"] > 2.5
    assert t["fp32"]["optimizer"] == t["ternary"]["optimizer"]                # moments stay FP32 in every configuration


def test_gemm_traffic_rereads_operands_only_when_sram_is_too_small():
    big = H.Hardware(pe=32, sram_bytes=64 << 20)
    a, b, c = H.gemm_traffic(1024, 512, 512, 4, 4, 4, big)
    assert (a, b, c) == (1024 * 512 * 4, 512 * 512 * 4, 1024 * 512 * 4)       # everything crosses the boundary exactly once
    small = H.Hardware(pe=32, sram_bytes=256 << 10)
    assert sum(H.gemm_traffic(1024, 512, 512, 4, 4, 4, small)) > sum((a, b, c))
    assert sum(H.gemm_traffic(1024, 512, 512, 1, 0.25, 1, small)) < sum(H.gemm_traffic(1024, 512, 512, 4, 4, 4, small))
