"""Shape inference helper (best-effort).

Fusion pattern matching in C3 is topology-driven and does not strictly require
shapes, but a shape map is handy for tuning ``problem_size`` and for future
passes.  This wraps ``onnx.shape_inference`` when the raw model is available and
otherwise falls back to whatever static shapes the :class:`Graph` already knows.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..graph import Graph


class ShapeInferencePass:
    name = "ShapeInference"

    def run(self, graph: Graph) -> Dict[str, Any]:
        """Return ``{tensor_name: shape}`` for graph inputs/outputs we can see.

        TODO: propagate shapes through every node (currently only graph-level
        tensors are populated; per-node intermediate shapes are left for a full
        implementation).
        """
        shapes: Dict[str, list] = {}
        for t in list(graph.inputs) + list(graph.outputs):
            shapes[t.name] = list(t.shape)
        for name, val in graph.initializers.items():
            if val is not None and hasattr(val, "shape"):
                shapes[name] = list(val.shape)
        return {"shapes": shapes}
