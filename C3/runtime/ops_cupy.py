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

import cupy as cp


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


def op_MatMul(a, b, **_):
    return a @ b


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


def op_Conv(x, w, b=None, strides=None, pads=None, dilations=None,
            group=1, kernel_shape=None, **_):
    """2D convolution via im2col (NCHW) on GPU.

    The im2col expansion uses ``as_strided`` to build the patch view without a
    Python loop (the numpy reference's triple loop would be unusable on GPU).
    Only the ``group == 1`` path is exercised by the three public models; the
    grouped path falls back to a per-group loop for completeness.
    """
    n, c, h, wd = x.shape
    oc, ic_g, kh, kw = w.shape
    strides = strides or [1, 1]
    pads = pads or [0, 0, 0, 0]
    dilations = dilations or [1, 1]
    sh, sw = strides
    dh, dw = dilations
    pt, pl, pb, pr = pads
    xp = cp.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    oh = (xp.shape[2] - (dh * (kh - 1) + 1)) // sh + 1
    ow = (xp.shape[3] - (dw * (kw - 1) + 1)) // sw + 1

    if group == 1:
        cols = _im2col(xp, kh, kw, sh, sw, dh, dw, oh, ow)   # (n, c*kh*kw, oh*ow)
        wcol = cp.reshape(w, (oc, -1))                        # (oc, c*kh*kw)
        out = cp.einsum("ok,nkp->nop", wcol, cols)            # (n, oc, oh*ow)
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
            og = cp.einsum("ok,nkp->nop", wcol, cols).reshape(n, cg_out, oh, ow)
            outs.append(og)
        out = cp.concatenate(outs, axis=1)

    if b is not None:
        out = out + cp.reshape(b, (1, -1, 1, 1))
    return out


def _im2col(x, kh, kw, sh, sw, dh, dw, oh, ow):
    """Build the (n, c*kh*kw, oh*ow) im2col matrix via strided views.

    A direct port of the numpy reference's slice extraction, but vectorised:
    for each (channel, ki, kj) we take a strided slice over the padded input
    and copy it into the column matrix. Stays on GPU (no per-element Python).
    """
    n, c, h, w = x.shape
    # gather all (ci, ki, kj) patches: shape (c, kh, kw, n, oh, ow)
    cols = cp.empty((c, kh, kw, n, oh, ow), dtype=x.dtype)
    for ci in range(c):
        for i in range(kh):
            for j in range(kw):
                patch = x[:, ci, i * dh:i * dh + sh * oh:sh, j * dw:j * dw + sw * ow:sw]
                cols[ci, i, j] = cp.reshape(patch, (n, oh, ow))
    cols = cp.reshape(cols, (c * kh * kw, n, oh * ow))   # (c*kh*kw, n, oh*ow)
    return cp.transpose(cols, (1, 0, 2))                 # (n, c*kh*kw, oh*ow)


# op_type -> callable (auto-collected, mirroring ops_numpy)
OPS: Dict[str, Any] = {
    name[3:]: fn for name, fn in list(globals().items()) if name.startswith("op_")
}
