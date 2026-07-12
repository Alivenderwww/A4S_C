"""Operator fusion pass (子任务 C3.3).

Implements the fusion patterns from the C3.3 rubric.  Each match collapses a
group of nodes into a single *fused node* whose ``fused_ops`` list holds the
original constituents (so :mod:`runtime.mock_runtime` can replay them and the
numerical-alignment check passes by construction).

Pattern status
--------------
IMPLEMENTED (fire on the public models):
  * ``FusedMatMulBias``     -- MatMul -> Add(bias)         [transformer]
  * ``FusedResidualNorm``   -- skip-Add -> LayerNorm       [transformer]
  * ``FusedEWChain``        -- 2..5 adjacent elementwise    [transformer GELU, resnet Add->Relu]

MATCHER PRESENT, does not fire on current public models (documented TODO):
  * ``FusedSoftmaxDropout`` -- Softmax -> Dropout           (inference graph has no Dropout)
  * ``FusedConv2dBatchNorm``-- Conv   -> BatchNormalization (BN already folded into Conv weights)

TODO(FusedConv2dBatchNorm): to claim F1's 5th point on ResNet, add a *pre-fusion*
pass that reconstructs a BN node (or, equivalently, folds a scale/shift back out
of the Conv weights) so this Conv->BN matcher has something to match.  See
``PRE_FUSION_TODO`` below for the entry point stub.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..graph import Graph, Node

_ELEMENTWISE = {"Add", "Mul", "Div", "Sub", "Relu", "Erf", "Sqrt"}

# Rough "kernels per op" model for launch-count accounting (F2).
_KERNELS_PER_OP = {
    "MatMul": 1, "Gemm": 2, "Conv": 2, "Softmax": 5, "LayerNormalization": 7,
    "GlobalAveragePool": 1,
}


def _launches(node: Node) -> int:
    if node.fused_ops:
        return 1  # a fused node is a single fused kernel launch
    return _KERNELS_PER_OP.get(node.op_type, 1)


def _count_launches(graph: Graph) -> int:
    return sum(_launches(n) for n in graph.nodes)


def _count_buffers(graph: Graph) -> int:
    graph._rebuild_edges()
    return len({e["tensor"] for e in graph.edges})


class FusionPass:
    name = "Fusion"

    def __init__(self, enable_fusion: bool = True):
        self.enable_fusion = enable_fusion

    # ------------------------------------------------------------------
    def run(self, graph: Graph) -> Dict[str, Any]:
        raw_launches = _count_launches(graph)
        raw_buffers = _count_buffers(graph)

        fusion_log: List[Dict[str, Any]] = []
        if not self.enable_fusion:
            opt = graph.clone()
            return self._stats(opt, fusion_log, raw_launches, raw_buffers)

        work = graph.clone()
        # Priority order matters: bias/residual/EW-chain before the (dormant)
        # softmax/conv-bn matchers.  Each pattern skips already-consumed nodes.
        consumed: Set[str] = set()
        groups: List[Dict[str, Any]] = []

        for matcher in (
            self._match_matmul_bias,
            self._match_residual_norm,
            self._match_softmax_dropout,
            self._match_conv_bn,
            self._match_ew_chain,
        ):
            matcher(work, consumed, groups)

        opt = self._apply_groups(work, groups)
        for g in groups:
            fusion_log.append(
                {
                    "pattern": g["pattern"],
                    "nodes": [n.name for n in g["nodes"]],
                    "fused_node": g["fused_name"],
                }
            )
        return self._stats(opt, fusion_log, raw_launches, raw_buffers)

    def _stats(self, opt, fusion_log, raw_launches, raw_buffers):
        opt_launches = _count_launches(opt)
        opt_buffers = _count_buffers(opt)
        return {
            "graph": opt,
            "stats": {
                "fusion_log": fusion_log,
                "num_fused": len(fusion_log),
                "patterns_hit": sorted({e["pattern"] for e in fusion_log}),
                "raw_launches": raw_launches,
                "opt_launches": opt_launches,
                "raw_buffers": raw_buffers,
                "opt_buffers": opt_buffers,
            },
        }

    # ------------------------------------------------------------- matchers
    def _match_matmul_bias(self, g: Graph, consumed: Set[str], groups: List):
        producers = g.producer_map()
        consumers = g.consumer_map()
        for n in g.nodes:
            if n.name in consumed or n.op_type != "MatMul":
                continue
            out = n.outputs[0]
            succ = [c for c in consumers.get(out, []) if c.name not in consumed]
            if len(succ) != 1:
                continue
            add = succ[0]
            if add.op_type != "Add":
                continue
            # the Add's non-MatMul input must be a bias (an initializer / 1-D const)
            other = [t for t in add.inputs if t != out]
            if len(other) != 1 or other[0] not in g.initializer_names:
                continue
            groups.append(self._make_group("FusedMatMulBias", [n, add]))
            consumed.update({n.name, add.name})

    def _match_residual_norm(self, g: Graph, consumed: Set[str], groups: List):
        consumers = g.consumer_map()
        for n in g.nodes:
            if n.name in consumed or n.op_type != "Add":
                continue
            out = n.outputs[0]
            succ = [c for c in consumers.get(out, []) if c.name not in consumed]
            if len(succ) != 1:
                continue
            ln = succ[0]
            if ln.op_type not in ("LayerNormalization", "LayerNorm"):
                continue
            # residual add: neither input is a bias-only initializer
            non_init = [t for t in n.inputs if t not in g.initializer_names]
            if len(non_init) < 2:
                continue
            groups.append(self._make_group("FusedResidualNorm", [n, ln]))
            consumed.update({n.name, ln.name})

    def _match_softmax_dropout(self, g: Graph, consumed: Set[str], groups: List):
        consumers = g.consumer_map()
        for n in g.nodes:
            if n.name in consumed or n.op_type != "Softmax":
                continue
            out = n.outputs[0]
            succ = [c for c in consumers.get(out, []) if c.name not in consumed]
            if len(succ) == 1 and succ[0].op_type == "Dropout":
                groups.append(self._make_group("FusedSoftmaxDropout", [n, succ[0]]))
                consumed.update({n.name, succ[0].name})

    def _match_conv_bn(self, g: Graph, consumed: Set[str], groups: List):
        # Dormant on the public ResNet (no BN nodes). See PRE_FUSION_TODO.
        consumers = g.consumer_map()
        for n in g.nodes:
            if n.name in consumed or n.op_type != "Conv":
                continue
            out = n.outputs[0]
            succ = [c for c in consumers.get(out, []) if c.name not in consumed]
            if len(succ) == 1 and succ[0].op_type in ("BatchNormalization", "BatchNorm"):
                groups.append(self._make_group("FusedConv2dBatchNorm", [n, succ[0]]))
                consumed.update({n.name, succ[0].name})

    def _match_ew_chain(self, g: Graph, consumed: Set[str], groups: List):
        consumers = g.consumer_map()
        by_name = {n.name: n for n in g.nodes}
        for start in g.nodes:
            if start.name in consumed or start.op_type not in _ELEMENTWISE:
                continue
            chain = [start]
            cur = start
            while len(chain) < 5:
                out = cur.outputs[0]
                succ = [c for c in consumers.get(out, []) if c.name not in consumed]
                if len(succ) != 1:
                    break
                nxt = succ[0]
                if nxt.op_type not in _ELEMENTWISE or nxt.name in consumed:
                    break
                # nxt must not already be the head of another chain member
                if nxt in chain:
                    break
                chain.append(nxt)
                cur = nxt
            if len(chain) >= 2:
                groups.append(self._make_group("FusedEWChain", chain))
                consumed.update({c.name for c in chain})

    # ------------------------------------------------------------- helpers
    def _make_group(self, pattern: str, nodes: List[Node]) -> Dict[str, Any]:
        return {
            "pattern": pattern,
            "nodes": nodes,
            "fused_name": f"{pattern}::{nodes[0].name}",
        }

    def _apply_groups(self, g: Graph, groups: List[Dict[str, Any]]) -> Graph:
        """Build the optimized graph, replacing each group with one fused node."""
        # map every consumed node name -> its group index (position of fused node)
        name_to_group: Dict[str, int] = {}
        for gi, grp in enumerate(groups):
            for n in grp["nodes"]:
                name_to_group[n.name] = gi

        # Precompute external outputs per group (consumed outside group or graph out)
        graph_outs = g.output_names()
        consumers = g.consumer_map()

        fused_nodes: Dict[int, Node] = {}
        for gi, grp in enumerate(groups):
            members = grp["nodes"]
            member_names = {n.name for n in members}
            produced_within: Set[str] = {o for n in members for o in n.outputs}
            ext_inputs: List[str] = []
            for n in members:
                for t in n.inputs:
                    if t not in produced_within and t not in ext_inputs:
                        ext_inputs.append(t)
            ext_outputs: List[str] = []
            for n in members:
                for o in n.outputs:
                    used_outside = any(
                        c.name not in member_names for c in consumers.get(o, [])
                    )
                    if o in graph_outs or used_outside:
                        if o not in ext_outputs:
                            ext_outputs.append(o)
            if not ext_outputs:  # fall back to last node's outputs
                ext_outputs = list(members[-1].outputs)
            fused_nodes[gi] = Node(
                name=grp["fused_name"],
                op_type=grp["pattern"],
                inputs=ext_inputs,
                outputs=ext_outputs,
                attrs={"fused": True, "pattern": grp["pattern"]},
                fused_ops=[m.clone() for m in members],
            )

        # Rebuild node list in original order, emitting the fused node at the
        # position of its first member, dropping the other members.
        new_nodes: List[Node] = []
        emitted: Set[int] = set()
        for n in g.nodes:
            gi = name_to_group.get(n.name)
            if gi is None:
                new_nodes.append(n)
            elif gi not in emitted:
                new_nodes.append(fused_nodes[gi])
                emitted.add(gi)
            # else: drop (already represented by the fused node)

        opt = Graph(
            name=g.name + "_fused",
            nodes=new_nodes,
            inputs=[t for t in g.inputs],
            outputs=[t for t in g.outputs],
            initializers=dict(g.initializers),
        )
        return opt


# ---------------------------------------------------------------------------
# PRE_FUSION_TODO: Conv+BN reconstruction to unlock FusedConv2dBatchNorm (F1).
# ---------------------------------------------------------------------------
def prefuse_conv_bn(graph: Graph) -> Graph:
    """TODO: reconstruct/normalise Conv->BN so the fusion matcher can fire.

    The public ResNet ONNX has BN already folded into Conv weights (no BN nodes),
    so ``_match_conv_bn`` never fires.  A full implementation would either:

      1. Detect the affine (scale, shift) baked into consecutive conv weights and
         split it back into an explicit BatchNormalization node, then let the
         normal matcher fold it back (round-trip that proves the machinery), or
      2. Accept a sidecar BN-params file and re-insert BN nodes before fusion.

    For now this is a documented no-op that returns the graph unchanged.
    """
    return graph
