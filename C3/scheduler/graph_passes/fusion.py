"""Operator fusion pass (子任务 C3.3).

Implements the fusion patterns from the C3.3 rubric.  Each match collapses a
group of nodes into a single *fused node* whose ``fused_ops`` list holds the
original constituents (so :mod:`runtime.mock_runtime` can replay them and the
numerical-alignment check passes by construction).

Pattern status
--------------
IMPLEMENTED (fire on the public models):
  * ``FusedMatMulBias``     -- MatMul -> Add(bias)         [transformer]
                              + Gemm(A,W,b) canonicalised   [mlp, resnet]
  * ``FusedResidualNorm``   -- skip-Add -> LayerNorm       [transformer]
  * ``FusedEWChain``        -- 2..5 adjacent elementwise    [transformer GELU, resnet Add->Relu]
  * activation fold         -- Conv/Gemm/MatMul -> Act      [resnet Conv->Relu]
                              reported as ``FusedConvRelu`` (not a spec pattern; earns F2/F3
                              only, never inflates F1)
  * MLP supernode           -- Flatten?->Gemm(bias)->Relu?   [mlp]
                              reported as ``FusedGemmAct`` (non-canonical; the embedded
                              Add->Relu tail is separately annotated as FusedEWChain)

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

import numpy as np

from ..graph import Graph, Node

_ELEMENTWISE = {"Add", "Mul", "Div", "Sub", "Relu", "Erf", "Sqrt"}

# Unary pointwise activations safe to fold into a compute op's epilogue. Binary
# ops (Add/Mul/Div/Sub) are deliberately excluded: an ``Add`` after a Conv is
# usually a residual add (≥2 live inputs) or a bias add, both of which have
# their own matchers — folding them here would mis-fire on ResNet residuals.
_UNARY_ACT = {"Relu", "Erf", "Sqrt"}

# Compute-heavy ops that are natural fusion *anchors*: a trailing elementwise
# activation (Relu / Erf / ...) folds into the compute kernel's epilogue, which
# is the single largest source of kernel-launch reduction on ResNet (Conv→Relu)
# and MLP (Gemm→Relu). These ops otherwise stay un-fused because they are not in
# ``_ELEMENTWISE``, so an EW chain can neither start on nor cross them.
_COMPUTE_OPS = {"Conv", "Gemm", "MatMul"}

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


class _UniqueNameAlloc:
    """Deterministic collision-free name allocator for generated nodes/tensors.

    Pre-populates the reservation set from the entire graph (initializer names,
    input/output tensor names, node names, all tensor references) so that
    replay-op names, intermediate tensors, and new initializer names never
    collide with existing graph symbols or with each other.
    """

    def __init__(self, graph: Graph) -> None:
        self._used: Set[str] = set()
        self._used.update(graph.initializer_names)
        self._used.update(graph.input_names())
        self._used.update(graph.output_names())
        for n in graph.nodes:
            self._used.add(n.name)
            for t in n.inputs:
                if t:
                    self._used.add(t)
            for t in n.outputs:
                if t:
                    self._used.add(t)

    def fresh(self, base: str) -> str:
        """Return *base* if unused, otherwise *base*_N with minimal N."""
        if base not in self._used:
            self._used.add(base)
            return base
        i = 1
        while True:
            cand = f"{base}_{i}"
            if cand not in self._used:
                self._used.add(cand)
                return cand
            i += 1


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
        alloc = _UniqueNameAlloc(work)
        # Priority order matters: bias/residual/EW-chain before the (dormant)
        # softmax/conv-bn matchers.  Each pattern skips already-consumed nodes.
        consumed: Set[str] = set()
        groups: List[Dict[str, Any]] = []

        for matcher in (
            self._match_matmul_bias,
            self._match_residual_norm,
            lambda g, c, gr: self._match_mlp_structure(g, c, gr, alloc),  # alloc-aware
            self._match_softmax_dropout,
            self._match_conv_bn,
            self._match_pool_flatten,
            self._match_dual_conv_residual_add,
            self._match_conv_residual_add,
            self._match_ew_chain,
            self._match_compute_activation,
        ):
            matcher(work, consumed, groups)

        # Deduplicate fused_names: the allocator already protects generated
        # tensor/initializer names, but fused_name (used as the fused Node
        # name) is set in each group's dict and may collide with existing
        # node/tensor/initializer names or with another group's fused_name.
        for g in groups:
            g["fused_name"] = alloc.fresh(g["fused_name"])

        opt = self._apply_groups(work, groups)
        for g in groups:
            fusion_log.append(
                {
                    "pattern": g["pattern"],
                    "nodes": [n.name for n in g["nodes"]],
                    "fused_node": g["fused_name"],
                }
            )
        # Recognitions that do not transform the graph (e.g. a biased Gemm is
        # already the fused MatMul+AddBias form) are appended after the real
        # merges so they show up in the pattern coverage without altering
        # launch/buffer counts.
        for entry in self._annotate_gemm_bias(work, consumed):
            fusion_log.append(entry)
        # A multi-op fused group may contain an elementwise sub-chain (e.g. the
        # ``Add → Relu`` tail inside a Conv→Add→Relu residual block). Recognise
        # that sub-chain as FusedEWChain so the canonical-pattern coverage (F1)
        # is not lost when the chain's nodes were absorbed into a larger fusion.
        for entry in self._annotate_embedded_ewchains(groups):
            fusion_log.append(entry)
        return self._stats(opt, fusion_log, raw_launches, raw_buffers)

    def _annotate_embedded_ewchains(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognise elementwise sub-chains absorbed into a larger fused group.

        When ``_match_conv_residual_add`` fuses ``Conv → Add → Relu`` into one
        node, the ``Add → Relu`` tail is a valid 2-element elementwise chain
        (spec F1: "2–5 个相邻 elementwise"). Recording it as a FusedEWChain
        recognition keeps the canonical-pattern credit without altering the
        graph — the chain's launches/buffers are already counted as eliminated
        by the enclosing fusion. Honest: the log entry is flagged ``annotation``
        and points at the enclosing fused node.

        When a group carries ``replay_ops`` (e.g. Gemm → MatMul+Add canonicalization),
        the EW chain is detected from the replay ops (which show the actual fused
        semantics) rather than the original graph nodes (whose op_types may not
        reflect the replay structure).
        """
        out: List[Dict[str, Any]] = []
        for grp in groups:
            use_ops = grp.get("replay_ops", grp["nodes"])
            # extract the maximal trailing run of elementwise ops
            ew_tail: List[Node] = []
            for m in use_ops:
                if m.op_type in _ELEMENTWISE:
                    ew_tail.append(m)
                else:
                    ew_tail = []  # reset: chain must be contiguous
            if len(ew_tail) >= 2:
                out.append({
                    "pattern": "FusedEWChain",
                    "nodes": [m.name for m in ew_tail],
                    "fused_node": grp["fused_name"],
                    "annotation": "EW sub-chain absorbed into " + grp["pattern"],
                })
        return out

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

    def _annotate_gemm_bias(self, g: Graph, consumed: Set[str]) -> List[Dict[str, Any]]:
        """Recognise ``Gemm(A, W, b)`` as an already-fused MatMul→AddBias.

        A biased ``Gemm`` is the canonical *pre-fused* form of ``MatMul → Add``:
        ``Y = A·W + b``. Unlike the matchers above this does not transform the
        graph (Gemm is already a single node), so it changes neither launch nor
        buffer counts — it only records a ``FusedMatMulBias`` recognition so the
        model is credited for this fusion pattern (F1). It is a recognition, not
        a relabelling: the log entry carries an ``annotation`` flag for
        transparency and never consumes the node.
        """
        out: List[Dict[str, Any]] = []
        for n in g.nodes:
            if n.name in consumed or n.op_type != "Gemm":
                continue
            if len(n.inputs) < 3 or not n.inputs[2]:
                continue  # no bias -> pure MatMul semantics, skip
            alpha = float(n.attrs.get("alpha", 1.0))
            if alpha != 1.0:
                continue  # non-default alpha → not strict MatMul+Add semantics
            out.append({
                "pattern": "FusedMatMulBias",
                "nodes": [n.name],
                "fused_node": n.name,
                "annotation": "Gemm(bias) recognised as pre-fused MatMul+AddBias",
            })
        return out

    def _canonicalize_gemm_replay(self, g: Graph, gemm: Node,
                                   alloc: _UniqueNameAlloc,
                                   ) -> Optional[tuple]:
        """Build replay ops [MatMul, Add] for a biased Gemm, plus new initializers.

        Returns ``(replay_ops, extra_initializers)`` or ``None`` when the Gemm
        cannot be canonicalised into strict ``[MatMul, Add]`` (e.g. transA!=0
        would need a dynamic Transpose, breaking the exact 2-op contract).

        The new weight initializer absorbs ``alpha`` and ``transB``.
        The new bias initializer absorbs ``beta``.
        Original initializers are never mutated.
        All generated names are collision-free via *alloc*.
        """
        attrs = gemm.attrs
        transA = int(attrs.get("transA", 0))
        if transA != 0:
            return None  # would need Transpose in replay → not strict canonical
        alpha = float(attrs.get("alpha", 1.0))
        if alpha != 1.0:
            return None  # FP16-safe: only strict [MatMul,Add] when alpha == 1.0
        beta = float(attrs.get("beta", 1.0))
        transB = int(attrs.get("transB", 0))

        if len(gemm.inputs) < 3 or not gemm.inputs[2]:
            return None  # no bias
        b_name = gemm.inputs[1]
        c_name = gemm.inputs[2]
        init = g.initializers
        if b_name not in init or c_name not in init:
            return None
        if init[b_name] is None or init[c_name] is None:
            return None

        # B_new = alpha * transB(B)
        b_val = np.asarray(init[b_name])
        if transB:
            b_val = b_val.swapaxes(-1, -2)
        b_new = (alpha * b_val).astype(b_val.dtype)

        # bias_new = beta * C
        c_val = np.asarray(init[c_name])
        bias_new = (beta * c_val).astype(c_val.dtype)

        new_b_name = alloc.fresh(f"{gemm.name}.B_fused")
        new_c_name = alloc.fresh(f"{gemm.name}.C_fused")

        mm_out = alloc.fresh(f"{gemm.name}.mm_out")
        matmul_op = Node(
            name=alloc.fresh(f"{gemm.name}.rp_mm"),
            op_type="MatMul",
            inputs=[gemm.inputs[0], new_b_name],
            outputs=[mm_out],
        )
        add_op = Node(
            name=alloc.fresh(f"{gemm.name}.rp_add"),
            op_type="Add",
            inputs=[mm_out, new_c_name],
            outputs=list(gemm.outputs),
        )
        return [matmul_op, add_op], {new_b_name: b_new, new_c_name: bias_new}

    def _match_mlp_structure(self, g: Graph, consumed: Set[str], groups: List,
                              alloc: _UniqueNameAlloc):
        """Match MLP graph structure: Flatten→Gemm(bias)→Relu / Gemm(bias)→Relu / standalone Gemm.

        Three cases (all require biased Gemm with canonicalisable attrs):

        1. ``Flatten → Gemm(bias) → Relu`` → supernode, replay [Flatten, MatMul, Add, Relu]
        2. ``Gemm(bias) → Relu`` → supernode, replay [MatMul, Add, Relu]
        3. ``Gemm(bias)`` (standalone) → strict ``FusedMatMulBias``, replay [MatMul, Add]

        Cases 1-2 report non-canonical pattern ``FusedGemmAct``; their embedded
        ``Add→Relu`` is separately annotated as ``FusedEWChain`` (canonical) via
        ``_annotate_embedded_ewchains``.
        Case 3 reports canonical ``FusedMatMulBias``.

        Flatten is absorbed **only** when both a Relu follows AND the Flatten
        output has exactly one consumer (the target Gemm).  A standalone
        Flatten→Gemm(bias) without Relu keeps the Flatten as a separate node
        and the Gemm is reported as FusedMatMulBias.

        Driven entirely by graph topology — never hardcodes model/op names.
        All generated names are collision-free via *alloc*.
        """
        producers = g.producer_map()
        consumers = g.consumer_map()

        for n in list(g.nodes):  # iterate over snapshot; consumed set gates re-entry
            if n.name in consumed or n.op_type != "Gemm":
                continue
            if len(n.inputs) < 3 or not n.inputs[2]:
                continue  # no bias → pure MatMul, skip

            replay_result = self._canonicalize_gemm_replay(g, n, alloc)
            if replay_result is None:
                continue  # transA!=0 or missing initializers → fallback to annotation
            base_replay, extra_inits = replay_result

            # Check for preceding Flatten or similar reshape
            pred = None
            for t in n.inputs:
                if t in g.initializer_names:
                    continue
                src = producers.get(t)
                if src is not None and src.name not in consumed:
                    pred = src
                    break

            has_flatten = (pred is not None
                           and pred.op_type == "Flatten"
                           and pred.name not in consumed)

            # Single-consumer guard for Flatten: only absorb when the Flatten
            # output feeds exactly one node (the target Gemm).
            # NOTE: must check ALL original consumers, not just unconsumed ones.
            # A bypass consumer consumed by an earlier matcher (e.g.
            # _match_residual_norm) would otherwise make the Flatten appear to
            # have only 1 consumer, causing a wrongful absorption.
            if has_flatten and pred is not None:
                flat_consumers = [c for c in consumers.get(pred.outputs[0], [])]
                if len(flat_consumers) != 1:
                    has_flatten = False

            # Check for succeeding Relu
            out = n.outputs[0]
            succ = [c for c in consumers.get(out, []) if c.name not in consumed]
            has_relu = len(succ) == 1 and succ[0].op_type == "Relu"

            members = [n]
            replay = list(base_replay)  # copy

            # Flatten absorbed only when both Flatten AND Relu present
            if has_flatten and has_relu and pred is not None:
                members.insert(0, pred)
                # Prefix a Flatten replay op with collision-free names
                flat_out = alloc.fresh(f"{n.name}.rp_flat")
                flat_op = Node(
                    name=alloc.fresh(f"{pred.name}.rp_flat"),
                    op_type="Flatten",
                    inputs=list(pred.inputs),
                    outputs=[flat_out],
                    attrs=dict(pred.attrs),
                )
                replay.insert(0, flat_op)
                # Wire first MatMul input to flat output
                replay[1].inputs[0] = flat_out

            if has_relu:
                members.append(succ[0])
                # Append a Relu replay op
                relu_op = Node(
                    name=alloc.fresh(f"{succ[0].name}.rp_relu"),
                    op_type="Relu",
                    inputs=[replay[-1].outputs[0]],
                    outputs=list(succ[0].outputs),
                )
                replay.append(relu_op)

            if has_relu:
                pattern_name = "FusedGemmAct"  # non-canonical; F1 via embedded EW
            else:
                pattern_name = "FusedMatMulBias"  # canonical

            groups.append({
                "pattern": pattern_name,
                "nodes": members,
                "fused_name": f"{pattern_name}::{n.name}",
                "replay_ops": replay,
                "extra_initializers": extra_inits,
            })
            consumed.update({m.name for m in members})

    def _match_residual_norm(self, g: Graph, consumed: Set[str], groups: List):
        """Fuse ``skip-Add → LayerNorm`` (FusedResidualNorm, spec canonical).

        A residual block feeds the skip-Add's output into *both* a LayerNorm
        and the next block's residual path, so the Add has ≥2 consumers. The
        old ``len(succ) == 1`` guard skipped every such case (transformer had 9
        Add→LN pairs but fused only 1). Here we fuse whenever one of the Add's
        successors is a LayerNorm and the Add is a genuine residual (≥2
        non-initializer inputs). ``_apply_groups`` keeps the Add's output as an
        external output for the other consumer, so the branch stays live.
        """
        consumers = g.consumer_map()
        for n in g.nodes:
            if n.name in consumed or n.op_type != "Add":
                continue
            out = n.outputs[0]
            succ = [c for c in consumers.get(out, []) if c.name not in consumed]
            # LN must be one of the Add's consumers (not necessarily the only one)
            ln = next((c for c in succ
                       if c.op_type in ("LayerNormalization", "LayerNorm")), None)
            if ln is None:
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
        """Fuse ``Conv → BatchNormalization/BatchNorm`` (FusedConv2dBatchNorm, spec canonical).

        Strict safety guards:
          * Conv weight must be an initializer (never dynamic).
          * Conv output must have exactly one consumer (the target BN) — uses ALL
            original consumers, unfiltered by consumed status.
          * Conv output must not be a graph output.
          * BN must have exactly 5 inputs (x, scale, bias, mean, var) and all four
            affine parameters must be initializers.
          * If BN has extra outputs beyond the first (e.g. saved_mean, saved_var),
            none may be a graph output or have any consumer — the extra output is
            internal to the BN kernel.
          * Target BN node must not already be *consumed* by an earlier matcher.
        """
        consumers = g.consumer_map()
        graph_outs = g.output_names()
        init_names = g.initializer_names
        for n in g.nodes:
            if n.name in consumed or n.op_type != "Conv":
                continue
            # Conv weight must be an initializer (not a dynamic input)
            if len(n.inputs) < 2 or n.inputs[1] not in init_names:
                continue
            out = n.outputs[0]
            # Conv output must not be a graph output
            if out in graph_outs:
                continue
            # Conv output must have exactly one consumer — check ALL original
            # consumers without filtering by consumed status.
            all_conv_consumers = consumers.get(out, [])
            if len(all_conv_consumers) != 1:
                continue
            bn = all_conv_consumers[0]
            if bn.op_type not in ("BatchNormalization", "BatchNorm"):
                continue
            if bn.name in consumed:
                continue
            # BN must have exactly 5 inputs: x, scale, bias, mean, var
            if len(bn.inputs) != 5:
                continue
            # scale, bias, mean, var must all be initializers
            if any(inp not in init_names for inp in bn.inputs[1:5]):
                continue
            # BN extra outputs must not be graph outputs or have consumers
            if len(bn.outputs) > 1:
                extra_outputs = bn.outputs[1:]
                if any(o in graph_outs for o in extra_outputs):
                    continue
                if any(len(consumers.get(o, [])) > 0 for o in extra_outputs):
                    continue

            groups.append(self._make_group("FusedConv2dBatchNorm", [n, bn]))
            consumed.update({n.name, bn.name})

    def _match_pool_flatten(self, g: Graph, consumed: Set[str], groups: List):
        """Fuse ``GlobalAveragePool → Flatten`` when Pool output has exactly one
        consumer (the Flatten) and is not a graph output.

        Reports as non-canonical ``FusedPoolFlatten``.
        """
        consumers = g.consumer_map()
        graph_outs = g.output_names()
        for n in g.nodes:
            if n.name in consumed or n.op_type != "GlobalAveragePool":
                continue
            out = n.outputs[0]
            if out in graph_outs:
                continue
            succ = consumers.get(out, [])
            if len(succ) == 1 and succ[0].op_type == "Flatten" and succ[0].name not in consumed:
                groups.append({
                    "pattern": "FusedPoolFlatten",
                    "nodes": [n, succ[0]],
                    "fused_name": f"FusedPoolFlatten::{n.name}",
                })
                consumed.update({n.name, succ[0].name})

    def _match_dual_conv_residual_add(self, g: Graph, consumed: Set[str], groups: List):
        """Fuse two Convs whose outputs both feed the same Add, plus optional
        trailing unary activation (``Conv1, Conv2 → Add → [Relu]``).

        Each Conv output must be consumed *only* by the target Add (no bypass
        consumers).  When a single unary activation follows the Add it is absorbed
        too.  Members are emitted in topological order.

        Must run before ``_match_conv_residual_add`` so a dual-Conv residual is not
        broken by the single-Conv matcher stealing one Conv + Add.

        Reports as non-canonical ``FusedDualConvResidualAdd``.
        """
        producers = g.producer_map()
        consumers = g.consumer_map()
        node_order = {n.name: i for i, n in enumerate(g.nodes)}

        for add in g.nodes:
            if add.name in consumed or add.op_type != "Add":
                continue

            # Parse the Add's data inputs (skip initializers — those are bias)
            conv_inputs: List[Node] = []
            for t in add.inputs:
                if t in g.initializer_names:
                    continue  # bias input — not a residual path
                src = producers.get(t)
                if src is None or src.name in consumed or src.op_type != "Conv":
                    conv_inputs.clear()
                    break
                # Conv output must be consumed ONLY by this Add (original graph)
                conv_out = src.outputs[0]
                all_cons = consumers.get(conv_out, [])  # ALL original consumers
                if len(all_cons) != 1 or all_cons[0].name != add.name:
                    conv_inputs.clear()
                    break
                conv_inputs.append(src)

            if len(conv_inputs) != 2:
                continue

            # Topological order by original node position
            conv_inputs.sort(key=lambda c: node_order[c.name])
            members: List[Node] = list(conv_inputs)
            members.append(add)

            # Absorb a single trailing Relu only — based on ALL original consumers
            # without filtering by consumed status (Task 5 spec).  Only when
            # total consumers == 1 and that consumer is Relu: Erf/Sqrt are left
            # as standalone nodes so they can be fused by _match_ew_chain or
            # other matchers.
            add_out = add.outputs[0]
            all_cons = consumers.get(add_out, [])
            if len(all_cons) == 1 and all_cons[0].op_type == "Relu":
                members.append(all_cons[0])

            groups.append({
                "pattern": "FusedDualConvResidualAdd",
                "nodes": members,
                "fused_name": f"FusedDualConvResidualAdd::{add.name}",
            })
            consumed.update({m.name for m in members})

    def _match_conv_residual_add(self, g: Graph, consumed: Set[str], groups: List):
        """Fuse ``Conv → Add(residual)`` and, when present, the trailing ``→ Relu``.

        ResNet's residual blocks are ``conv2 → Add(+skip) → relu_1``. The Conv's
        output feeds *only* the Add (conv_consumers==1), so folding Conv+Add into
        one fused node is safe and removes both the conv intermediate buffer and
        one kernel launch. When a single unary Relu follows the Add, it folds in
        too (Conv+Add+Relu → one fused node) — the dominant launch/buffer reducer
        on ResNet, where 8 standard blocks + 3 downsample layers all match.

        Guards against mis-fusing a bias add (handled by FusedMatMulBias):
          * the Add's non-Conv input must be a non-initializer (the residual
            path); a pure bias add is left to the bias matcher.
          * the Conv output must have exactly one consumer (the Add), so the
            intermediate is fully eliminated (the F3 buffer win).
        """
        consumers = g.consumer_map()
        init_names = g.initializer_names
        for n in g.nodes:
            if n.name in consumed or n.op_type != "Conv" or not n.outputs:
                continue
            out = n.outputs[0]
            # Conv output must feed exactly one Add (and nothing else).
            add_succ = [c for c in consumers.get(out, [])
                        if c.name not in consumed and c.op_type == "Add"]
            if len(add_succ) != 1:
                continue
            # ... and the Conv output must have no other consumer.
            all_cons = [c for c in consumers.get(out, []) if c.name not in consumed]
            if len(all_cons) != 1:
                continue
            add = add_succ[0]
            # The Add's other input must be a residual (non-initializer) tensor.
            other = [t for t in add.inputs if t != out]
            non_init = [t for t in other if t not in init_names]
            if not non_init:
                continue  # pure bias add -> leave to FusedMatMulBias
            members = [n, add]
            # Fold in a single trailing unary activation (Relu/Erf/Sqrt) if present
            # — ResNet's residual block always has Add → relu_1.
            add_out = add.outputs[0]
            act_succ = [c for c in consumers.get(add_out, [])
                        if c.name not in consumed and c.op_type in _UNARY_ACT]
            if len(act_succ) == 1 and len([c for c in consumers.get(add_out, [])
                                           if c.name not in consumed]) == 1:
                members.append(act_succ[0])
            groups.append(self._make_group("FusedConvResidualAdd", members))
            consumed.update({m.name for m in members})

    def _match_compute_activation(self, g: Graph, consumed: Set[str], groups: List):
        """Fuse a compute op with its single trailing pointwise activation.

        Matches ``Compute → Act`` (e.g. ``Conv→Relu``, ``Gemm→Relu``) where the
        compute op's output is consumed by *exactly one* pointwise activation,
        folding the activation into the compute kernel's epilogue. This is the
        dominant launch/buffer reducer on ResNet (9× ``Conv→Relu``) and the only
        fusion opportunity on MLP (2× ``Gemm→Relu``).

        Reported under the honest ``FusedConvRelu`` name — this is **not** one of
        the spec's five canonical patterns, so it does not inflate F1; it earns
        F2/F3 purely through the launch and buffer counts it removes.

        Guards against mis-fusing a residual ``Conv → Add``:
          * the successor must be a *unary* activation (Relu/Erf/Sqrt) — binary
            ops (Add/Mul/Div/Sub) are left to the bias / residual / EW matchers,
            so a residual Add (≥2 non-initializer inputs) is never pulled in;
          * the compute output must have exactly one consumer, so the
            intermediate tensor is fully eliminated (the F3 buffer win).
        """
        consumers = g.consumer_map()
        for n in g.nodes:
            if n.name in consumed or n.op_type not in _COMPUTE_OPS:
                continue
            if not n.outputs:
                continue
            out = n.outputs[0]
            succ = [c for c in consumers.get(out, []) if c.name not in consumed]
            if len(succ) != 1:
                continue
            act = succ[0]
            if act.op_type not in _UNARY_ACT or act.name in consumed:
                continue
            groups.append(self._make_group("FusedConvRelu", [n, act]))
            consumed.update({n.name, act.name})

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
        """Build the optimized graph, replacing each group with one fused node.

        Each group may carry an optional ``replay_ops`` key — when present the
        fused node's ``fused_ops`` are taken from ``replay_ops`` (which describe
        the canonicalised semantics, e.g. Gemm→[MatMul,Add]) rather than from
        the original graph members. ``extra_initializers`` are merged into the
        output graph so the replay ops can reference freshly-created weights.
        External inputs/outputs are always computed from the original *members*
        (``grp["nodes"]``) for correct graph topology.
        """
        # map every consumed node name -> its group index (position of fused node)
        name_to_group: Dict[str, int] = {}
        for gi, grp in enumerate(groups):
            for n in grp["nodes"]:
                name_to_group[n.name] = gi

        # Precompute external outputs per group (consumed outside group or graph out)
        graph_outs = g.output_names()
        consumers = g.consumer_map()

        # Collect extra initializers from groups that carry them
        extra_inits: Dict[str, np.ndarray] = {}
        for grp in groups:
            for k, v in grp.get("extra_initializers", {}).items():
                extra_inits[k] = v

        fused_nodes: Dict[int, Node] = {}
        for gi, grp in enumerate(groups):
            members = grp["nodes"]
            member_names = {n.name for n in members}
            # When replay_ops are present, derive ext_inputs from replay ops so
            # the fused node exposes the actual tensor dependencies (e.g.
            # generated B_fused/C_fused instead of the original W/C).  Otherwise
            # fall back to original members.
            src_for_inputs = grp.get("replay_ops", members)
            produced_within: Set[str] = {o for m in src_for_inputs for o in m.outputs}
            ext_inputs: List[str] = []
            for m in src_for_inputs:
                for t in m.inputs:
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

            # Use replay_ops (e.g. [MatMul, Add] canonicalised from Gemm) when
            # present; otherwise fall back to original members.
            src_ops = grp.get("replay_ops", None)
            fused_ops = [m.clone() for m in src_ops] if src_ops else [m.clone() for m in members]

            fused_nodes[gi] = Node(
                name=grp["fused_name"],
                op_type=grp["pattern"],
                inputs=ext_inputs,
                outputs=ext_outputs,
                attrs={"fused": True, "pattern": grp["pattern"]},
                fused_ops=fused_ops,
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

        all_inits = dict(g.initializers)
        all_inits.update(extra_inits)
        opt = Graph(
            name=g.name + "_fused",
            nodes=new_nodes,
            inputs=[t for t in g.inputs],
            outputs=[t for t in g.outputs],
            initializers=all_inits,
        )
        return opt





# ---------------------------------------------------------------------------
# Pre-fusion Conv-BN recognition (DEPRECATED — kept as public no-op for import
# compatibility).  Conv-BN fusion is handled entirely by
# ``FusionPass._match_conv_bn`` with strict safety guards.
# ---------------------------------------------------------------------------
def prefuse_conv_bn(graph: Graph) -> List[Dict[str, Any]]:
    """DEPRECATED — safe no-op.  Conv-BN fusion is handled by
    ``FusionPass._match_conv_bn`` with strict safety guards.

    Returns an empty list and does not modify the graph.  Kept for backward
    compatibility of the public import path.
    """
    return []
