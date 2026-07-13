#!/usr/bin/env python3
"""C3.5 CLI: run ONNX model inference over a batch of inputs and write outputs.

Usage (command template submitted at 报名):

    python tools/infer.py --onnx {onnx} --input {input} --output {output} --batch-size 256

Correctness-first design:
  * Prefer onnxruntime with the CUDA execution provider (GPU), then fall back to
    the CPU provider, then to ``onnx.reference.ReferenceEvaluator``.
  * Everything runs in fp32 to stay within the 1e-3 accuracy gate.
  * Samples are processed in ``--batch-size`` chunks (row order preserved), which
    also bounds peak GPU memory (a scored metric).

Reads ``<input>/manifest.json`` + ``.npy`` tensors, writes
``<output>/manifest.json`` + ``logits.npy`` (dtype float32).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_C3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3_ROOT not in sys.path:
    sys.path.insert(0, _C3_ROOT)


# ---------------------------------------------------------------------------
# ONNX dtype string -> numpy dtype (for casting inputs to what the model wants)
# ---------------------------------------------------------------------------
_ORT_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def _read_manifest(input_dir: str):
    with open(os.path.join(input_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_inputs(input_dir: str):
    manifest = _read_manifest(input_dir)
    tensors = {}
    n = None
    for t in manifest["tensors"]:
        arr = np.load(os.path.join(input_dir, t["file"]))
        tensors[t["name"]] = arr
        n = arr.shape[0] if n is None else n
    return tensors, int(n)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class _OrtBackend:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.providers = self.sess.get_providers()
        self.input_dtypes = {i.name: _ORT_TYPE_TO_NP.get(i.type, np.float32)
                             for i in self.sess.get_inputs()}
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def run(self, feed):
        feed = {k: np.ascontiguousarray(v, dtype=self.input_dtypes.get(k, v.dtype))
                for k, v in feed.items()}
        outs = self.sess.run(self.output_names, feed)
        return dict(zip(self.output_names, outs))


class _ReferenceBackend:
    """Pure-onnx fallback (no onnxruntime available)."""

    def __init__(self, onnx_path: str):
        import onnx
        from onnx.reference import ReferenceEvaluator

        model = onnx.load(onnx_path)
        self.output_names = [o.name for o in model.graph.output]
        # infer input dtypes from the model
        self.input_dtypes = {}
        for i in model.graph.input:
            et = i.type.tensor_type.elem_type
            self.input_dtypes[i.name] = {
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.INT64: np.int64,
                onnx.TensorProto.INT32: np.int32,
            }.get(et, np.float32)
        self.providers = ["ReferenceEvaluator"]
        self.ev = ReferenceEvaluator(model)

    def run(self, feed):
        feed = {k: np.ascontiguousarray(v, dtype=self.input_dtypes.get(k, v.dtype))
                for k, v in feed.items()}
        outs = self.ev.run(self.output_names, feed)
        return dict(zip(self.output_names, outs))


class _CupyBackend:
    """CuPy GPU backend — the spec-required numerical library (手写算子).

    Loads the ONNX graph via ``scheduler.import_onnx_graph`` and executes it on
    the GPU with :class:`runtime.cupy_runtime.CupyRuntime`, which runs every
    operator via :mod:`runtime.ops_cupy`. Weights upload once; each batch's
    inputs upload, run, and download.
    """

    def __init__(self, onnx_path: str):
        # import lazily so the (heavy) CuPy import only happens when this backend
        # is actually selected — keeps ORT-only runs cheap.
        import sys as _sys
        _root = _C3_ROOT
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from scheduler import import_onnx_graph
        from runtime.cupy_runtime import CupyRuntime

        self.graph = import_onnx_graph(onnx_path)
        self.rt = CupyRuntime(self.graph)
        self.output_names = [t.name for t in self.graph.outputs]
        # dtype per input, mirroring _OrtBackend (used to cast feed tensors)
        from scheduler.graph import _dtype_name  # noqa: F401  (kept for parity)
        self.input_dtypes = {t.name: _ORT_TYPE_TO_NP.get(
            "tensor(" + t.dtype.lower() + ")", np.float32) for t in self.graph.inputs}
        self.providers = ["CuPy"]

    def run(self, feed):
        feed = {k: np.ascontiguousarray(v, dtype=self.input_dtypes.get(k, v.dtype))
                for k, v in feed.items()}
        return self.rt.run(feed)


def _make_backend(onnx_path: str, backend_name: str = "cupy"):
    """Select the inference backend.

    Order: CuPy (spec-required default) -> onnxruntime -> ReferenceEvaluator.
    Pass ``backend_name="ort"`` to force onnxruntime.
    """
    if backend_name != "ort":
        try:
            backend = _CupyBackend(onnx_path)
            print(f"[infer] backend: CuPy GPU")
            return backend
        except Exception as exc:
            print(f"[infer] CuPy unavailable ({exc}); falling back", file=sys.stderr)
    try:
        backend = _OrtBackend(onnx_path)
        print(f"[infer] backend: onnxruntime ({backend.providers})")
        return backend
    except Exception as exc:
        print(f"[infer] onnxruntime unavailable ({exc}); using ReferenceEvaluator", file=sys.stderr)
        return _ReferenceBackend(onnx_path)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ONNX inference (C3.5)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True, help="input dir with manifest.json + .npy")
    ap.add_argument("--output", required=True, help="output dir")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--backend", default="cupy", choices=["cupy", "ort"],
                    help="inference backend: cupy (default, GPU) or ort")
    args = ap.parse_args(argv)

    inputs, n = _load_inputs(args.input)
    backend = _make_backend(onnx_path=args.onnx, backend_name=args.backend)

    bs = args.batch_size if args.batch_size and args.batch_size > 0 else n
    bs = max(1, bs)

    collected = {name: [] for name in backend.output_names}
    for start in range(0, n, bs):
        end = min(start + bs, n)
        feed = {name: arr[start:end] for name, arr in inputs.items()}
        out = backend.run(feed)
        for name in backend.output_names:
            collected[name].append(np.asarray(out[name], dtype=np.float32))

    results = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}

    os.makedirs(args.output, exist_ok=True)
    out_tensors = []
    for name, arr in results.items():
        fname = f"{name}.npy"
        np.save(os.path.join(args.output, fname), arr.astype(np.float32))
        out_tensors.append(
            {"name": name, "file": fname, "dtype": "float32", "shape": list(arr.shape)}
        )
    with open(os.path.join(args.output, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"tensors": out_tensors}, f, indent=2, ensure_ascii=False)

    print(f"[infer] wrote {n} samples -> {args.output} ({[t['name'] for t in out_tensors]})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[infer] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
