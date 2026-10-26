"""ctypes bridge to libbmlowbit (cpp_engine): the C++ ternary / stochastic-rounding / BFP kernels.

    cmake -S cpp_engine -B cpp_engine/build && cmake --build cpp_engine/build --target bmlowbit

The library is optional: `available()` is False when it has not been built (the pure-NumPy code in
low_bit_training.py is the reference and works without it). The BM_LOWBIT_LIB environment variable
overrides the search path.
"""
import ctypes
import os
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
_i8 = ndpointer(np.int8, flags="C_CONTIGUOUS")
_u8 = ndpointer(np.uint8, flags="C_CONTIGUOUS")
_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")

_lib = None
_tried = False


def _candidates():
    env = os.environ.get("BM_LOWBIT_LIB")
    if env:
        yield env
    ext = {"win32": "bmlowbit.dll", "darwin": "libbmlowbit.dylib"}.get(sys.platform, "libbmlowbit.so")
    for d in ("cpp_engine/build", "build", "cpp_engine/build-release"):
        yield os.path.join(_ROOT, d, ext)


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    for path in _candidates():
        if not os.path.exists(path):
            continue
        lib = ctypes.CDLL(path)
        lib.bm_ternary_row_bytes.argtypes = [ctypes.c_int]
        lib.bm_ternary_row_bytes.restype = ctypes.c_int
        lib.bm_ternary_quantize_absmean.argtypes = [_f32, ctypes.c_size_t, _i8]
        lib.bm_ternary_quantize_absmean.restype = ctypes.c_float
        lib.bm_pack_ternary.argtypes = [_i8, ctypes.c_int, ctypes.c_int, _u8]
        lib.bm_unpack_ternary.argtypes = [_u8, ctypes.c_int, ctypes.c_int, _i8]
        lib.bm_ternary_gemm_i32.argtypes = [ctypes.c_int] * 3 + [_i8, _u8, _i32]
        lib.bm_ternary_gemm_f32.argtypes = [ctypes.c_int] * 3 + [_i8, _f32, _u8, ctypes.c_float, _f32]
        lib.bm_quantize_row_q8.argtypes = [_f32, ctypes.c_int, _i8]
        lib.bm_quantize_row_q8.restype = ctypes.c_float
        lib.bm_stochastic_round.argtypes = [_f32, ctypes.c_size_t, ctypes.c_uint64, _f32]
        lib.bm_quantize_to_grid_i8.argtypes = [_f32, ctypes.c_size_t, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, _i8]
        lib.bm_bfp_quantize.argtypes = [_f32, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, _i8, _i8]
        lib.bm_bfp_dequantize.argtypes = [_i8, _i8, ctypes.c_size_t, ctypes.c_int, _f32]
        lib.bm_simd_isa.restype = ctypes.c_char_p
        _lib = lib
        break
    return _lib


def available():
    return _load() is not None


def isa():
    return _load().bm_simd_isa().decode()


def row_bytes(K):
    return _load().bm_ternary_row_bytes(K)


def ternary_quantize_absmean(W):
    W = np.ascontiguousarray(W, dtype=np.float32)
    q = np.empty(W.shape, dtype=np.int8)
    gamma = _load().bm_ternary_quantize_absmean(W.reshape(-1), W.size, q.reshape(-1))
    return q, np.float32(gamma)


def pack_ternary(Wq):
    """Wq: int8 (rows, K) in {-1, 0, 1} -> uint8 (rows, row_bytes(K))."""
    Wq = np.ascontiguousarray(Wq, dtype=np.int8)
    rows, K = Wq.shape
    out = np.empty((rows, row_bytes(K)), dtype=np.uint8)
    _load().bm_pack_ternary(Wq, rows, K, out)
    return out


def unpack_ternary(packed, K):
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    out = np.empty((packed.shape[0], K), dtype=np.int8)
    _load().bm_unpack_ternary(packed, packed.shape[0], K, out)
    return out


def ternary_gemm_i32(Xq, packed, K):
    """Xq: int8 (M, K) in [-127, 127]; packed: (N, row_bytes(K)) -> int32 (M, N), add/subtract only."""
    Xq = np.ascontiguousarray(Xq, dtype=np.int8)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    M, N = Xq.shape[0], packed.shape[0]
    out = np.empty((M, N), dtype=np.int32)
    _load().bm_ternary_gemm_i32(M, N, K, Xq, packed, out)
    return out


def quantize_rows_q8(x):
    """Per-row symmetric INT8 of a float32 (M, K) matrix -> (int8 (M, K), float32 scales (M,))."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    q = np.empty(x.shape, dtype=np.int8)
    scales = np.empty(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        scales[i] = _load().bm_quantize_row_q8(x[i], x.shape[1], q[i])
    return q, scales


def stochastic_round(x, seed):
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x)
    _load().bm_stochastic_round(x.reshape(-1), x.size, seed, out.reshape(-1))
    return out


def bfp_quantize(x, block, bits, stochastic, seed=0):
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    mant = np.empty(x.size, dtype=np.int8)
    exps = np.empty((x.size + block - 1) // block, dtype=np.int8)
    _load().bm_bfp_quantize(x, x.size, block, bits, int(stochastic), seed, mant, exps)
    return mant, exps


def bfp_dequantize(mant, exps, block):
    out = np.empty(mant.size, dtype=np.float32)
    _load().bm_bfp_dequantize(np.ascontiguousarray(mant), np.ascontiguousarray(exps), mant.size, block, out)
    return out
