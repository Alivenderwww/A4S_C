"""Reference numpy implementations of the 17 supported ONNX operators.

Used by :mod:`runtime.mock_runtime` for the C3.3 numerical-alignment check
(original graph vs fused graph).  These are correctness-first, not speed-first;
Conv is vectorised via im2col so ResNet still runs in reasonable time.

Not used by the production C3.5 path (``tools/infer.py``), which delegates to
onnxruntime for speed and exactness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

_ERF = np.vectorize(math.erf)


def op_Flatten(x, axis=1, **_):
    axis = axis if axis >= 0 else x.ndim + axis
    shape = x.shape
    outer = int(np.prod(shape[:axis])) if axis > 0 else 1
    return x.reshape(outer, -1)


def op_Gemm(a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0, **_):
    if transA:
        a = a.swapaxes(-1, -2)
    if transB:
        b = b.swapaxes(-1, -2)
    y = alpha * (a @ b)
    if c is not None:
        y = y + beta * c
    return y


def op_MatMul(a, b, **_):
    return a @ b


def op_Relu(x, **_):
    return np.maximum(x, 0)


def op_Add(a, b, **_):
    return a + b


def op_Sub(a, b, **_):
    return a - b


def op_Mul(a, b, **_):
    return a * b


def op_Div(a, b, **_):
    return a / b


def op_Erf(x, **_):
    return _ERF(x).astype(x.dtype)


def op_Sqrt(x, **_):
    return np.sqrt(x)


def op_Softmax(x, axis=-1, **_):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def op_LayerNormalization(x, scale, bias=None, axis=-1, epsilon=1e-5, **_):
    axis = axis if axis >= 0 else x.ndim + axis
    axes = tuple(range(axis, x.ndim))
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=axes, keepdims=True)
    norm = (x - mean) / np.sqrt(var + epsilon)
    y = norm * scale
    if bias is not None:
        y = y + bias
    return y


def op_Gather(data, indices, axis=0, **_):
    return np.take(data, indices.astype(np.int64), axis=axis)


def op_Transpose(x, perm=None, **_):
    if perm is None:
        perm = tuple(reversed(range(x.ndim)))
    return np.transpose(x, perm)


def op_Reshape(x, shape, allowzero=0, **_):
    shape = np.asarray(shape).astype(np.int64).tolist()
    out = []
    for i, d in enumerate(shape):
        if d == 0 and not allowzero:
            out.append(x.shape[i])
        else:
            out.append(int(d))
    return x.reshape(out)


def op_Split(x, split=None, axis=0, num_outputs=None, **_):
    axis = axis if axis >= 0 else x.ndim + axis
    if split is not None:
        split = np.asarray(split).astype(np.int64).tolist()
        idx = np.cumsum(split)[:-1]
        return np.split(x, idx, axis=axis)
    n = num_outputs or 1
    return np.split(x, n, axis=axis)


def op_BatchNormalization(x, scale, bias, mean, var, epsilon=1e-5, **_):
    """Inference-mode BatchNorm: per-channel affine over the channel axis (1)."""
    scale = np.asarray(scale, dtype=x.dtype).reshape(1, -1, *([1] * (x.ndim - 2)))
    bias = np.asarray(bias, dtype=x.dtype).reshape(1, -1, *([1] * (x.ndim - 2)))
    mean = np.asarray(mean, dtype=x.dtype).reshape(1, -1, *([1] * (x.ndim - 2)))
    var = np.asarray(var, dtype=x.dtype).reshape(1, -1, *([1] * (x.ndim - 2)))
    return scale * (x - mean) / np.sqrt(var + epsilon) + bias


def op_GlobalAveragePool(x, **_):
    axes = tuple(range(2, x.ndim))
    return np.mean(x, axis=axes, keepdims=True)


def op_Constant(value=None, **_):
    return value


def op_Conv(x, w, b=None, strides=None, pads=None, dilations=None,
            group=1, kernel_shape=None, **_):
    """2D convolution via im2col (NCHW)."""
    n, c, h, wd = x.shape
    oc, ic_g, kh, kw = w.shape
    strides = strides or [1, 1]
    pads = pads or [0, 0, 0, 0]
    dilations = dilations or [1, 1]
    sh, sw = strides
    dh, dw = dilations
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    oh = (xp.shape[2] - (dh * (kh - 1) + 1)) // sh + 1
    ow = (xp.shape[3] - (dw * (kw - 1) + 1)) // sw + 1

    if group == 1:
        cols = _im2col(xp, kh, kw, sh, sw, dh, dw, oh, ow)  # (n, c*kh*kw, oh*ow)
        wcol = w.reshape(oc, -1)                            # (oc, c*kh*kw)
        out = np.einsum("ok,nkp->nop", wcol, cols)          # (n, oc, oh*ow)
        out = out.reshape(n, oc, oh, ow)
    else:
        outs = []
        cg_in = c // group
        cg_out = oc // group
        for g in range(group):
            xg = xp[:, g * cg_in:(g + 1) * cg_in]
            wg = w[g * cg_out:(g + 1) * cg_out]
            cols = _im2col(xg, kh, kw, sh, sw, dh, dw, oh, ow)
            wcol = wg.reshape(cg_out, -1)
            og = np.einsum("ok,nkp->nop", wcol, cols).reshape(n, cg_out, oh, ow)
            outs.append(og)
        out = np.concatenate(outs, axis=1)

    if b is not None:
        out = out + b.reshape(1, -1, 1, 1)
    return out


def _im2col(x, kh, kw, sh, sw, dh, dw, oh, ow):
    n, c, h, w = x.shape
    cols = np.empty((n, c * kh * kw, oh * ow), dtype=x.dtype)
    idx = 0
    for ci in range(c):
        for i in range(kh):
            for j in range(kw):
                patch = x[:, ci, i * dh:i * dh + sh * oh:sh, j * dw:j * dw + sw * ow:sw]
                cols[:, idx, :] = patch.reshape(n, -1)
                idx += 1
    return cols


# op_type -> callable
OPS: Dict[str, Any] = {
    name[3:]: fn for name, fn in list(globals().items()) if name.startswith("op_")
}
