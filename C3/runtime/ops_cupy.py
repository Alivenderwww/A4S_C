"""CuPy implementations of the 17 supported ONNX operators (spec: "数值计算库
统一采用 CuPy").

Mirrors :mod:`runtime.ops_numpy` signature-for-signature, but every tensor is a
``cupy.ndarray`` living in GPU memory. Used by :mod:`runtime.cupy_runtime`, the
GPU graph executor that backs the C3.5 inference path (``tools/infer.py``).

Conventions
-----------
* Inputs/outputs are ``cupy.ndarray`` (fp32 for activations/weights; int64 only
  for Gather indices). Host-side metadata (Reshape shape, Split split) stays in
  numpy and is passed in as plain python lists/arrays.
* Broadcasting follows numpy semantics; CuPy reproduces them.
* Numerical formulae match ops_numpy exactly so the C3.5 1e-3 gate (calibrated
  against the PyTorch fp32 reference) holds.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as _np
import cupy as cp
from cupy.cuda import cublas as _cublas


def op_Flatten(x, axis=1, **_):
    axis = axis if axis >= 0 else x.ndim + axis
    shape = x.shape
    outer = 1
    for d in shape[:axis]:
        outer *= int(d)
    return cp.reshape(x, (outer, -1))


def op_Gemm(a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0, **_):
    if transA:
        a = cp.swapaxes(a, -1, -2)
    if transB:
        b = cp.swapaxes(b, -1, -2)
    y = alpha * (a @ b)
    if c is not None:
        y = y + beta * c
    return y


# --- tensor-core, fp32-accurate MatMul (split-fp16) ------------------------
# CuPy's fp32 ``a @ b`` runs on CUDA cores (~5 TFLOP/s on the MIG slice). The
# tensor cores do fp16 at ~90 TFLOP/s, but plain fp16 overflows the 1e-3 gate on
# a deep model. Splitting each fp32 operand into a hi + lo fp16 pair and summing
# three fp16-input / fp32-accumulate tensor-core products (cublasGemmEx) recovers
# ~fp32 accuracy (per-matmul rel-err ~7e-6, i.e. BigFormer max_diff 3e-5 << 1e-3)
# at ~4x the fp32 throughput. Applied only to large 2D-weight matmuls (the
# Transformer / BigFormer projections, which are 90%+ of that model's compute);
# small or batched matmuls fall back to fp32.
_CUDA_R_16F = 2
_CUDA_R_32F = 0
_CUBLAS_COMPUTE_32F = 68
_GEMM_TENSOR_OP = 99                 # CUBLAS_GEMM_DEFAULT_TENSOR_OP
_F32_ONE = _np.array(1.0, dtype=_np.float32)
_F32_ZERO = _np.array(0.0, dtype=_np.float32)
_TC_MIN_ELEMS = 4096 * 64            # below this the fp16 cast overhead isn't worth it

# Split fp32 -> (hi, lo) fp16 in ONE pass (one launch, one read of x) instead of
# the 3 astype passes; produces bit-identical hi/lo and is ~3.7x faster (the cast
# was ~1/4 of the split-fp16 matmul time).
_split_hilo = cp.ElementwiseKernel(
    "float32 x", "float16 hi, float16 lo",
    "hi = x; lo = x - (float)hi;", "split_hilo")


def _gemm_acc(Ah, Bh, C, beta):
    """C[M,N] = Ah[M,K] @ Bh[K,N] + beta*C -- fp16 inputs, fp32 accumulate, tensor
    core, accumulating IN PLACE into C (beta=1) so the 3-term split needs only one
    fp32 output buffer, not three. Issued column-major (C^T = Bh^T @ Ah^T)."""
    M, K = Ah.shape
    N = Bh.shape[1]
    h = cp.cuda.device.get_cublas_handle()
    beta_ptr = (_F32_ONE if beta else _F32_ZERO).ctypes.data
    _cublas.gemmEx(h, 0, 0, N, M, K,
                   _F32_ONE.ctypes.data, Bh.data.ptr, _CUDA_R_16F, N,
                   Ah.data.ptr, _CUDA_R_16F, K,
                   beta_ptr, C.data.ptr, _CUDA_R_32F, N,
                   _CUBLAS_COMPUTE_32F, _GEMM_TENSOR_OP)


def _matmul_tc(a, b):
    if (a.dtype == cp.float32 and b.dtype == cp.float32 and b.ndim == 2 and a.ndim >= 2
            and a.shape[-1] == b.shape[0] and a.size >= _TC_MIN_ELEMS):
        A2 = a.reshape(-1, a.shape[-1])
        Bh, Bl = _split_hilo(b)
        Ah, Al = _split_hilo(A2)
        C = cp.empty((A2.shape[0], b.shape[1]), dtype=cp.float32)
        _gemm_acc(Ah, Bh, C, 0.0)   # C  = Ah@Bh
        _gemm_acc(Al, Bh, C, 1.0)   # C += Al@Bh
        del Al                       # free the two big fp16 operands promptly
        _gemm_acc(Ah, Bl, C, 1.0)   # C += Ah@Bl
        return C.reshape(*a.shape[:-1], b.shape[1])
    return a @ b


def op_MatMul(a, b, **_):
    return _matmul_tc(a, b)


def op_Relu(x, **_):
    return cp.maximum(x, cp.zeros((), dtype=x.dtype))


def op_Add(a, b, **_):
    return a + b


def op_Sub(a, b, **_):
    return a - b


def op_Mul(a, b, **_):
    return a * b


def op_Div(a, b, **_):
    return a / b


def op_Erf(x, **_):
    # cupyx.scipy.special.erf matches math.erf to double precision; keep input dtype.
    from cupyx.scipy.special import erf
    return erf(x).astype(x.dtype)


def op_Sqrt(x, **_):
    return cp.sqrt(x)


def op_Softmax(x, axis=-1, **_):
    axis = axis if axis >= 0 else x.ndim + axis
    x = x - cp.max(x, axis=axis, keepdims=True)
    e = cp.exp(x)
    return e / cp.sum(e, axis=axis, keepdims=True)


def op_LayerNormalization(x, scale, bias=None, axis=-1, epsilon=1e-5, **_):
    axis = axis if axis >= 0 else x.ndim + axis
    axes = tuple(range(axis, x.ndim))
    mean = cp.mean(x, axis=axes, keepdims=True)
    var = cp.mean((x - mean) ** 2, axis=axes, keepdims=True)
    norm = (x - mean) / cp.sqrt(var + epsilon)
    y = norm * scale
    if bias is not None:
        y = y + bias
    return y


def op_Gather(data, indices, axis=0, **_):
    return cp.take(data, indices.astype(cp.int64), axis=axis)


def op_Transpose(x, perm=None, **_):
    if perm is None:
        perm = tuple(reversed(range(x.ndim)))
    return cp.transpose(x, perm)


def op_Reshape(x, shape, allowzero=0, **_):
    shape = [int(d) for d in cp.asnumpy(shape).tolist()] if hasattr(shape, "ndim") else [int(d) for d in shape]
    out = []
    for i, d in enumerate(shape):
        if d == 0 and not allowzero:
            out.append(x.shape[i])
        else:
            out.append(int(d))
    return cp.reshape(x, out)


def op_Split(x, split=None, axis=0, num_outputs=None, **_):
    axis = axis if axis >= 0 else x.ndim + axis
    if split is not None:
        split = [int(d) for d in (cp.asnumpy(split).tolist() if hasattr(split, "ndim") else split)]
        idx = list(cp.cumsum(cp.array(split[:-1], dtype=cp.int64)).tolist()) if len(split) > 1 else []
        return cp.split(x, idx, axis=axis)
    n = num_outputs or 1
    return cp.split(x, n, axis=axis)


def op_GlobalAveragePool(x, **_):
    axes = tuple(range(2, x.ndim))
    return cp.mean(x, axis=axes, keepdims=True)


def op_Constant(value=None, **_):
    return value


def op_Identity(x, **_):
    """Pass-through (ONNX Identity). BigFormer uses it for residual wiring."""
    return x


def op_Conv(x, w, b=None, strides=None, pads=None, dilations=None,
            group=1, kernel_shape=None, **_):
    """2D convolution via im2col (NCHW) on GPU.

    Two fast paths:
    * **1×1 pointwise** (kh==kw==1): a pure channel projection — reshape the
      input to (n, c, h·w) and the weight to (oc, c), one cuBLAS GEMM. Skips
      pad + im2col gather entirely (~1.65x faster on the downsample convs).
    * **General k×k**: im2col via a single fused fancy-index gather (no Python
      loop over channels) + one cuBLAS batched GEMM. Launch count is
      independent of ``ic`` — critical for ResNet where ic reaches 512.

    Only the ``group == 1`` path is exercised by the three public models; the
    grouped path falls back to a per-group loop.
    """
    n, c, h, wd = x.shape
    oc, ic_g, kh, kw = w.shape
    strides = strides or [1, 1]
    pads = pads or [0, 0, 0, 0]
    dilations = dilations or [1, 1]
    sh, sw = strides
    dh, dw = dilations
    pt, pl, pb, pr = pads

    # ---- 1x1 stride-1 fast path: pure channel projection, no pad/im2col ----
    # (1x1 convs with stride>1 fall through to the im2col path, which handles
    #  spatial downsampling correctly; the strided 1x1 in ResNet is only the
    #  3 downsample layers, so this fast path covers the common case.)
    if kh == 1 and kw == 1 and sh == 1 and sw == 1 and group == 1:
        xs = cp.reshape(x, (n, c, h * wd))          # (n, c, h*w)
        ws = cp.reshape(w, (oc, c))                  # (oc, c)
        out = cp.matmul(ws[None], xs)                # (1,oc,c)@(n,c,h*w)->(n,oc,h*w)
        out = cp.reshape(out, (n, oc, h, wd))
        if b is not None:
            out = out + cp.reshape(b, (1, -1, 1, 1))
        return out

    xp = cp.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    oh = (xp.shape[2] - (dh * (kh - 1) + 1)) // sh + 1
    ow = (xp.shape[3] - (dw * (kw - 1) + 1)) // sw + 1

    if group == 1:
        cols = _im2col(xp, kh, kw, sh, sw, dh, dw, oh, ow)   # (n, c*kh*kw, oh*ow)
        wcol = cp.reshape(w, (oc, -1))                        # (oc, c*kh*kw)
        # batched GEMM via cuBLAS: (1,oc,K) @ (n,K,P) -> (n,oc,P).
        out = cp.matmul(wcol[None], cols)
        out = cp.reshape(out, (n, oc, oh, ow))
    else:
        outs = []
        cg_in = c // group
        cg_out = oc // group
        for g in range(group):
            xg = xp[:, g * cg_in:(g + 1) * cg_in]
            wg = w[g * cg_out:(g + 1) * cg_out]
            cols = _im2col(xg, kh, kw, sh, sw, dh, dw, oh, ow)
            wcol = cp.reshape(wg, (cg_out, -1))
            og = cp.matmul(wcol[None], cols).reshape(n, cg_out, oh, ow)
            outs.append(og)
        out = cp.concatenate(outs, axis=1)

    if b is not None:
        out = out + cp.reshape(b, (1, -1, 1, 1))
    return out


def _im2col(x, kh, kw, sh, sw, dh, dw, oh, ow):
    """Build the (n, c*kh*kw, oh*ow) im2col matrix via strided slices.

    For each of the kh·kw kernel taps we take a contiguous strided slice of the
    padded input (zero-copy view) and copy it into a pre-allocated buffer. This
    is ~1.9x faster than a fancy-index gather for ResNet's layer1 (the dominant
    cost): the gather does scattered reads across (n, c, kh·kw, oh·ow) while
    each tap slice is a coalesced contiguous copy.

    The kh·kw Python loop is fixed-size (≤9 for 3×3), so launch overhead is
    bounded and far smaller than the gather's memory traffic.
    """
    n, c, h, w = x.shape
    out = cp.empty((n, c, kh * kw, oh, ow), dtype=x.dtype)
    idx = 0
    for ki in range(kh):
        for kj in range(kw):
            out[:, :, idx] = x[:, :, ki * dh:ki * dh + sh * oh:sh,
                                     kj * dw:kj * dw + sw * ow:sw]
            idx += 1
    return out.reshape(n, c * kh * kw, oh * ow)


# op_type -> callable (auto-collected, mirroring ops_numpy)
OPS: Dict[str, Any] = {
    name[3:]: fn for name, fn in list(globals().items()) if name.startswith("op_")
}
