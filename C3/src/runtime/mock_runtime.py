"""A minimal numpy graph executor used for the C3.3 alignment check.

``MockRuntime`` walks a :class:`scheduler.graph.Graph` in topological order and
evaluates each node with :mod:`runtime.ops_numpy`.  Its sole purpose is to let
the fusion pass be checked numerically: run the *original* graph and the
*optimized* graph on the same inputs and confirm ``max_abs_diff <= 1e-3``.

Fused nodes (``node.fused_ops`` non-empty) are executed by replaying their
constituent original ops in order, so a correct fusion is numerically identical
to the un-fused graph by construction.

This runtime is intentionally NOT the C3.5 inference path -- that uses
onnxruntime (see ``tools/infer.py``) for speed and exactness.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from scheduler.graph import Graph, Node
from .ops_numpy import OPS


class MockRuntime:
    def __init__(self, graph: Graph):
        self.graph = graph

    def run(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        env: Dict[str, Any] = {}
        # seed initializers (weights / folded constants) and external inputs
        for name, val in self.graph.initializers.items():
            if val is not None:
                env[name] = np.asarray(val)
        for name, val in feeds.items():
            env[name] = np.asarray(val)

        for node in self.graph.topo_order():
            self._exec_node(node, env)

        return {t.name: env[t.name] for t in self.graph.outputs if t.name in env}

    # ------------------------------------------------------------------
    def _exec_node(self, node: Node, env: Dict[str, Any]) -> None:
        if node.fused_ops:
            # Replay the original sub-ops; internal tensors live in the same env.
            for sub in node.fused_ops:
                self._exec_primitive(sub, env)
            return
        self._exec_primitive(node, env)

    def _exec_primitive(self, node: Node, env: Dict[str, Any]) -> None:
        fn = OPS.get(node.op_type)
        if fn is None:
            raise NotImplementedError(f"MockRuntime: op {node.op_type!r} not implemented")

        args: List[Any] = []
        for i in node.inputs:
            if i == "":
                args.append(None)
            else:
                args.append(env.get(i))
        # Constant has no data inputs; its value comes from attrs/initializers.
        if node.op_type == "Constant":
            val = node.attrs.get("value")
            if val is None:
                val = self.graph.initializers.get(node.outputs[0])
            env[node.outputs[0]] = np.asarray(val)
            return

        out = fn(*args, **node.attrs)
        if isinstance(out, (list, tuple)):
            for name, val in zip(node.outputs, out):
                env[name] = val
        else:
            env[node.outputs[0]] = out
