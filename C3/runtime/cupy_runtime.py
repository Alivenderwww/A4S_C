"""A CuPy graph executor — the GPU backend for the C3.5 inference path.

Mirrors :class:`runtime.mock_runtime.MockRuntime` but runs every operator on
the GPU via :mod:`runtime.ops_cupy`. This is the backend ``tools/infer.py``
uses to satisfy the spec requirement that "数值计算库统一采用 CuPy".

Lifecycle
---------
* Construction loads the ONNX graph once and uploads **every float32
  initializer to the GPU**, caching it so weights are transferred exactly once
  across all inference batches. int64 initializers (Reshape/Split metadata)
  stay on the host as numpy.
* ``run(feeds)`` uploads one batch of inputs, walks the graph in topological
  order executing each node with the CuPy op table, and returns outputs as
  numpy arrays (transferred back from the device).

Constant nodes are handled specially: their value is host-side metadata
(shape vectors, scale scalars) and is kept in numpy; only when such a value
feeds a float computation does the consuming op see a GPU array (the op casts
on demand via ``cp.asarray``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - CuPy is the documented hard dependency
    cp = None  # type: ignore

from scheduler.graph import Graph, Node
from .ops_cupy import OPS


class CupyRuntime:
    """GPU graph executor backed by CuPy."""

    def __init__(self, graph: Graph):
        if cp is None:
            raise RuntimeError("CuPy is not available; cannot use CupyRuntime")
        self.graph = graph
        self._topo = graph.topo_order()
        # Pre-upload float weights to the GPU once; keep int64 metadata on host.
        self._init_gpu: Dict[str, Any] = {}
        for name, val in graph.initializers.items():
            if val is None:
                continue
            arr = np.asarray(val)
            # int64 initializers are shape/split metadata — keep on host.
            if arr.dtype == np.int64 or arr.dtype == np.int32:
                self._init_gpu[name] = arr
            else:
                self._init_gpu[name] = cp.asarray(arr)

    @property
    def input_dtypes(self) -> Dict[str, Any]:
        return {t.name: t.dtype for t in self.graph.inputs}

    def run(self, feeds: Dict[str, Any]) -> Dict[str, np.ndarray]:
        env: Dict[str, Any] = {}
        # seed initializers (already uploaded / host metadata)
        for name, val in self._init_gpu.items():
            env[name] = val
        # seed external inputs (upload this batch)
        for name, val in feeds.items():
            arr = np.asarray(val)
            env[name] = cp.asarray(arr)

        for node in self._topo:
            self._exec_node(node, env)

        out: Dict[str, np.ndarray] = {}
        for t in self.graph.outputs:
            v = env.get(t.name)
            if v is None:
                continue
            out[t.name] = cp.asnumpy(v) if hasattr(v, "ndim") and not isinstance(v, np.ndarray) else np.asarray(v)
        return out

    # ------------------------------------------------------------------
    def _exec_node(self, node: Node, env: Dict[str, Any]) -> None:
        if node.fused_ops:
            # C3.5 runs the original (un-fused) graph, but handle fused nodes
            # defensively by replaying their constituents.
            for sub in node.fused_ops:
                self._exec_primitive(sub, env)
            return
        self._exec_primitive(node, env)

    def _exec_primitive(self, node: Node, env: Dict[str, Any]) -> None:
        fn = OPS.get(node.op_type)
        if fn is None:
            raise NotImplementedError(f"CupyRuntime: op {node.op_type!r} not implemented")

        if node.op_type == "Constant":
            val = node.attrs.get("value")
            if val is None:
                val = self.graph.initializers.get(node.outputs[0])
            arr = np.asarray(val)
            # Float constants feed compute ops -> promote to GPU now. Integer
            # constants are host metadata (Reshape shapes, Split sizes) and are
            # consumed as plain numpy by the ops that need them.
            env[node.outputs[0]] = cp.asarray(arr) if arr.dtype.kind == "f" else arr
            return

        # Resolve inputs. The environment already holds GPU arrays for weights
        # and prior activations; numpy only for int metadata. Float numpy
        # values (e.g. scalar constants produced before this refactor's upload)
        # are promoted lazily so compute ops never see a host float.
        args: List[Any] = []
        for i in node.inputs:
            if i == "":
                args.append(None)
                continue
            v = env.get(i)
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                v = cp.asarray(v)
            args.append(v)

        out = fn(*args, **node.attrs)
        if isinstance(out, (list, tuple)):
            for name, val in zip(node.outputs, out):
                env[name] = val
        else:
            env[node.outputs[0]] = out
