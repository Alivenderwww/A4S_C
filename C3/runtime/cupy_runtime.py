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
    """GPU graph executor backed by CuPy.

    ``streaming`` mode (for models whose weights exceed GPU memory, e.g.
    BigFormer at ~19 GB > 17 GB): float weights are kept on the host and
    uploaded just before the node that consumes them, then freed right after.
    Only the small int64 metadata stays resident. This caps the weight footprint
    at the largest single layer (~0.27 GB for BigFormer) instead of the full
    19 GB, trading H2D bandwidth for the ability to run at all.
    """

    def __init__(self, graph: Graph, streaming: bool = False):
        if cp is None:
            raise RuntimeError("CuPy is not available; cannot use CupyRuntime")
        self.graph = graph
        self._topo = graph.topo_order()
        self.streaming = streaming
        # Host-side weight store (numpy); used directly in streaming mode, and
        # as the upload source in eager mode.
        self._init_host: Dict[str, np.ndarray] = {}
        self._init_gpu: Dict[str, Any] = {}
        for name, val in graph.initializers.items():
            if val is None:
                continue
            arr = np.asarray(val)
            self._init_host[name] = arr
            if not streaming:
                if arr.dtype == np.int64 or arr.dtype == np.int32:
                    self._init_gpu[name] = arr
                else:
                    self._init_gpu[name] = cp.asarray(arr)
        # Streaming: precompute each float weight's last-use step so we can free
        # it immediately after its consuming node, keeping peak memory low.
        if streaming:
            self._weight_last_use = self._compute_last_use()

    def _compute_last_use(self) -> Dict[str, int]:
        """tensor name -> index in topo order of its last consuming node.

        Also tracks tensor *aliases* produced by Identity nodes (BigFormer
        wires weights through Identity, e.g. ``blocks.0.ln1.bias -> ln_f.bias``):
        the alias must live as long as its last consumer too.
        """
        last: Dict[str, int] = {}
        # map alias -> source initializer (built from Identity nodes)
        alias_source: Dict[str, str] = {}
        for node in self._topo:
            if node.op_type == "Identity" and len(node.inputs) == 1:
                src = node.inputs[0]
                if src in self._init_host:
                    for o in node.outputs:
                        alias_source[o] = src
        for i, node in enumerate(self._topo):
            for t in node.inputs:
                # track both original initializers and their Identity aliases
                src = alias_source.get(t, t)
                if src in self._init_host and self._init_host[src].dtype.kind == "f":
                    last[src] = i
                elif t in self._init_host and self._init_host[t].dtype.kind == "f":
                    last[t] = i
        return last

    @property
    def input_dtypes(self) -> Dict[str, Any]:
        return {t.name: t.dtype for t in self.graph.inputs}

    def run(self, feeds: Dict[str, Any]) -> Dict[str, np.ndarray]:
        env: Dict[str, Any] = {}
        if not self.streaming:
            # eager: all weights already on device / host metadata
            for name, val in self._init_gpu.items():
                env[name] = val
        # seed external inputs (upload this batch)
        for name, val in feeds.items():
            arr = np.asarray(val)
            env[name] = cp.asarray(arr)

        for i, node in enumerate(self._topo):
            if self.streaming:
                self._stream_weights_for(node, env, i)
            self._exec_node(node, env)
            if self.streaming:
                self._free_weights_after(node, env, i)

        out: Dict[str, np.ndarray] = {}
        for t in self.graph.outputs:
            v = env.get(t.name)
            if v is None:
                continue
            out[t.name] = cp.asnumpy(v) if hasattr(v, "ndim") and not isinstance(v, np.ndarray) else np.asarray(v)
        return out

    def _stream_weights_for(self, node: Node, env: Dict[str, Any], step: int) -> None:
        """Upload float weights this node needs, on demand (called BEFORE exec).

        Handles Identity-produced aliases (BigFormer wires weights through
        Identity): when a node consumes an alias, upload the source initializer
        and bind it under the alias name. int64/int32 metadata stays on host.
        Freeing happens in :meth:`_free_weights_after` (called AFTER exec) so a
        weight used and freed on the same step is still live during execution.
        """
        if not hasattr(self, "_alias_source"):
            self._alias_source = {}
            for n in self._topo:
                if n.op_type == "Identity" and len(n.inputs) == 1 and n.inputs[0] in self._init_host:
                    for o in n.outputs:
                        self._alias_source[o] = n.inputs[0]

        for t in node.inputs:
            if t == "" or t in env:
                continue
            src = self._alias_source.get(t, t)
            host = self._init_host.get(src if src in self._init_host else t)
            if host is None:
                continue
            if src in env and src != t:
                env[t] = env[src]
                continue
            if host.dtype.kind == "f":
                env[t] = cp.asarray(host)
            else:
                env[t] = host  # int metadata stays host-side

    def _free_weights_after(self, node: Node, env: Dict[str, Any], step: int) -> None:
        """Release float weights whose last use was this step (called AFTER exec).

        Dropping the ``env`` reference lets CuPy's memory pool reclaim the GPU
        block, keeping peak weight memory at ~one layer instead of the full model.
        """
        if not hasattr(self, "_alias_source"):
            return
        for t in node.inputs:
            src = self._alias_source.get(t, t)
            if self._weight_last_use.get(src) == step:
                for key in (t, src):
                    v = env.get(key)
                    if v is not None and hasattr(v, "ndim") and not isinstance(v, np.ndarray):
                        del env[key]

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
