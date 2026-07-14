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
        self._fuse_gelu()
        self._fuse_matmul_bias()    # BigFormer/Transformer: fold MatMul bias-add into gelu / residual
        self._fuse_conv_relu()      # ResNet: fold Conv->Relu into the bias epilogue
        self._fuse_add_relu()       # ResNet: fold residual Add->Relu into one kernel
        self.streaming = streaming
        # Host-side weight store (numpy); used directly in streaming mode, and
        # as the upload source in eager mode.
        self._init_host: Dict[str, np.ndarray] = {}
        self._init_gpu: Dict[str, Any] = {}
        for name, val in graph.initializers.items():
            if val is None:
                continue
            arr = np.asarray(val)
            if streaming and arr.dtype.kind == "f":
                # Pin the streamed float weights: `cp.asarray` from pinned host
                # memory runs at ~40 GB/s vs ~9 GB/s pageable on this MIG slice,
                # cutting BigFormer's per-pass 19 GB weight upload from ~2 s to
                # ~0.5 s. Pinned once here (in the warm-up task's cached backend,
                # off the timed path); pageable fallback if pinning is unavailable.
                try:
                    import cupyx
                    pinned = cupyx.empty_pinned(arr.shape, arr.dtype)
                    pinned[...] = arr
                    arr = pinned
                except Exception:
                    pass
            self._init_host[name] = arr
            if not streaming:
                if arr.dtype == np.int64 or arr.dtype == np.int32:
                    self._init_gpu[name] = arr
                else:
                    self._init_gpu[name] = cp.asarray(arr)
        # Build the C3.4 memory plan on the graph we actually execute (post
        # GELU-fusion) and drive execution from ITS schedule, so a single
        # memory-planning strategy (scheduler.memory) feeds both the C3.4 review
        # artifact (plan.steps / summary) and this C3.5 executor. The plan's
        # per-node ``uploads_before`` / ``frees_after`` reproduce the previous
        # hand-rolled streaming + activation-lifetime freeing exactly (identical
        # last-use analysis), so peak memory, timing and precision are unchanged:
        #   * frees_after[i]  : dead intermediates (activation-lifetime reuse) and,
        #                       in streaming mode, weights past their last use.
        #   * uploads_before[i]: streamed float weights this node first consumes.
        # The O(nodes x batch) activation footprint -- not the weights -- is what
        # OOMs at large batch (bs=32 already exceeds 16 GB); freeing dead
        # activations caps peak at the live set so the full set fits in one pass.
        from scheduler.memory import build_execution_plan
        exec_graph = Graph(name=self.graph.name, nodes=self._topo,
                           inputs=self.graph.inputs, outputs=self.graph.outputs,
                           initializers=self.graph.initializers)
        self._plan = build_execution_plan(exec_graph, streaming=streaming)
        # Remap the plan's index-keyed schedule onto our own topo order by node
        # name (robust even if the plan's topo sort tie-breaks differently).
        topo_idx = {n.name: i for i, n in enumerate(self._topo)}
        self._uploads_before: Dict[int, List[str]] = {}
        self._frees_after: Dict[int, List[str]] = {}
        for i, name in enumerate(self._plan.order_names):
            j = topo_idx[name]
            self._uploads_before[j] = self._plan.uploads_before.get(i, [])
            self._frees_after[j] = self._plan.frees_after.get(i, [])
        self._output_names = {t.name for t in graph.outputs}

    def _fuse_gelu(self):
        """Fuse the ONNX exact-GELU chain Div(h, sqrt2) -> Erf -> Add(., 1) ->
        Mul(., h) -> Mul(., 0.5) into one FusedGelu node: one kernel and one
        intermediate instead of five full-size ([., ., 4d]) ones. Anchored on the
        (GELU-unique) Erf; requires each step to have a single consumer and the
        trailing Mul to reuse the SAME h the Div divided -- the GELU signature --
        so anything else is left untouched. The 1e-3 gate validates the result."""
        prod = {o: n for n in self._topo for o in n.outputs}
        cons: Dict[str, list] = {}
        for n in self._topo:
            for i in n.inputs:
                cons.setdefault(i, []).append(n)

        def only(t):
            c = cons.get(t, [])
            return c[0] if len(c) == 1 else None

        fused: Dict[int, Node] = {}
        drop = set()
        for k, erf in enumerate(self._topo):
            if erf.op_type != "Erf":
                continue
            d = prod.get(erf.inputs[0])
            if d is None or d.op_type != "Div" or not d.inputs:
                continue
            h = d.inputs[0]
            a = only(erf.outputs[0])
            if a is None or a.op_type != "Add":
                continue
            m1 = only(a.outputs[0])
            if m1 is None or m1.op_type != "Mul" or h not in m1.inputs:
                continue
            m2 = only(m1.outputs[0])
            if m2 is None or m2.op_type != "Mul":
                continue
            # Unique name (index-suffixed): the memory plan keys its per-node
            # schedule by topo position, resolved via node name, so a collision
            # (e.g. two Erf nodes with empty names) would misalign the schedule.
            fused[id(erf)] = Node(name=f"FusedGelu_{k}_" + (erf.name or ""),
                                  op_type="FusedGelu", inputs=[h],
                                  outputs=[m2.outputs[0]])
            drop.update(id(n) for n in (d, erf, a, m1, m2))
        if not fused:
            return
        self._topo = [fused[id(n)] if id(n) in fused else n
                      for n in self._topo if id(n) in fused or id(n) not in drop]

    def _cons_map(self):
        cons: Dict[str, list] = {}
        for n in self._topo:
            for i in n.inputs:
                if i:
                    cons.setdefault(i, []).append(n)
        return cons

    def _fuse_matmul_bias(self):
        """Fold a MatMul bias-add ``Add(matmul, weight_bias)`` into its SINGLE
        consumer, eliminating the standalone bias-add pass:
          * consumer FusedGelu  -> FusedGeluBias(matmul, bias) = gelu(matmul+bias)
          * consumer (residual) Add -> FusedAdd3(matmul, bias, other)
        Runs AFTER _fuse_gelu so the FFN bias-add already feeds a FusedGelu. The
        bias stays an input of the fused node, so the memory plan still streams
        it (the plan is built after all fusion)."""
        init = self.graph.initializer_names
        prod = {o: n for n in self._topo for o in n.outputs if o}
        cons = self._cons_map()
        drop = set()
        replace: Dict[int, Node] = {}
        for badd in self._topo:
            if badd.op_type != "Add" or len(badd.inputs) != 2:
                continue
            a, b = badd.inputs
            pa, pb = prod.get(a), prod.get(b)
            if pa is not None and pa.op_type == "MatMul" and b in init:
                mm, bias = a, b
            elif pb is not None and pb.op_type == "MatMul" and a in init:
                mm, bias = b, a
            else:
                continue
            outs = cons.get(badd.outputs[0], [])
            if len(outs) != 1:
                continue
            cnode = outs[0]
            if id(cnode) in drop or id(cnode) in replace:
                continue
            if cnode.op_type == "FusedGelu":
                replace[id(cnode)] = Node(name=cnode.name, op_type="FusedGeluBias",
                                          inputs=[mm, bias], outputs=list(cnode.outputs),
                                          attrs=dict(cnode.attrs))
                drop.add(id(badd))
            elif cnode.op_type == "Add" and len(cnode.inputs) == 2:
                other = (cnode.inputs[0] if cnode.inputs[1] == badd.outputs[0]
                         else cnode.inputs[1])
                replace[id(cnode)] = Node(name=cnode.name, op_type="FusedAdd3",
                                          inputs=[mm, bias, other], outputs=list(cnode.outputs),
                                          attrs=dict(cnode.attrs))
                drop.add(id(badd))
        if not drop:
            return
        self._topo = [replace.get(id(n), n) for n in self._topo if id(n) not in drop]

    def _fuse_conv_relu(self):
        """Fold ``Conv -> Relu`` (Relu the conv output's ONLY consumer) into the
        conv's bias epilogue: one ``max(gemm+bias, 0)`` kernel instead of a
        bias-add pass followed by a separate relu pass. ResNet's stem and each
        block's first conv match this."""
        prod = {o: n for n in self._topo for o in n.outputs if o}
        cons = self._cons_map()
        fused: Dict[int, Node] = {}
        drop = set()
        for relu in self._topo:
            if relu.op_type != "Relu" or not relu.inputs:
                continue
            c = prod.get(relu.inputs[0])
            if c is None or c.op_type != "Conv":
                continue
            if len(cons.get(c.outputs[0], [])) != 1:      # conv output feeds only relu
                continue
            fused[id(c)] = Node(name=c.name, op_type="Conv", inputs=list(c.inputs),
                                outputs=[relu.outputs[0]],
                                attrs={**c.attrs, "fused_relu": 1})
            drop.add(id(relu))
        if not fused:
            return
        self._topo = [fused.get(id(n), n) for n in self._topo if id(n) not in drop]

    def _fuse_add_relu(self):
        """Fold a residual ``Add -> Relu`` (Relu the add's ONLY consumer) into one
        ``max(a+b, 0)`` kernel -- the block-output relu in every ResNet block."""
        prod = {o: n for n in self._topo for o in n.outputs if o}
        cons = self._cons_map()
        fused: Dict[int, Node] = {}
        drop = set()
        for relu in self._topo:
            if relu.op_type != "Relu" or not relu.inputs:
                continue
            a = prod.get(relu.inputs[0])
            if a is None or a.op_type != "Add":
                continue
            if len(cons.get(a.outputs[0], [])) != 1:
                continue
            fused[id(a)] = Node(name=a.name, op_type="FusedAddRelu", inputs=list(a.inputs),
                                outputs=[relu.outputs[0]], attrs=dict(a.attrs))
            drop.add(id(relu))
        if not fused:
            return
        self._topo = [fused.get(id(n), n) for n in self._topo if id(n) not in drop]

    @property
    def input_dtypes(self) -> Dict[str, Any]:
        return {t.name: t.dtype for t in self.graph.inputs}

    def run(self, feeds: Dict[str, Any]) -> Dict[str, np.ndarray]:
        env: Dict[str, Any] = {}
        if not self.streaming:
            # eager: every weight already resident on device / host metadata
            for name, val in self._init_gpu.items():
                env[name] = val
        else:
            # streaming: float weights arrive per-node via the plan; only the
            # small int metadata (Reshape shapes / Split sizes) stays resident.
            for name, arr in self._init_host.items():
                if arr.dtype.kind != "f":
                    env[name] = arr
        # seed external inputs (upload this batch)
        for name, val in feeds.items():
            env[name] = cp.asarray(np.asarray(val))
        return self._run_streaming(env) if self.streaming else self._run_eager(env)

    def _run_eager(self, env: Dict[str, Any]) -> Dict[str, np.ndarray]:
        frees = self._frees_after
        for i, node in enumerate(self._topo):
            self._exec_node(node, env)
            for t in frees.get(i, ()):                    # drop dead intermediates
                env.pop(t, None)
        return self._collect_outputs(env)

    def _run_streaming(self, env: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Streaming with H2D/compute overlap: a copy stream prefetches a node's
        float weights while earlier nodes compute on the main stream, so the
        ~19 GB weight upload (host->device, on PCIe) hides behind the matmul
        compute (HBM+SM) -- measured near-full overlap.

        Two event chains keep it correct despite the shared pool:
          * the MAIN stream waits for a node's weight-upload event before running
            it (weights ready before the kernel reads them);
          * the COPY stream waits for the main stream's most recent compute before
            uploading, so it never overwrites a pool block a still-running kernel
            is reading (the free race). Transformer layers share weight shapes, so
            freed blocks are reused in place and peak weight memory stays at
            ~prefetch-depth layers -- no per-step free_all_blocks needed.
        """
        uploads, frees = self._uploads_before, self._frees_after
        n = len(self._topo)
        if not hasattr(self, "_copy_stream"):
            self._copy_stream = cp.cuda.Stream(non_blocking=True)
        cs = self._copy_stream
        main = cp.cuda.get_current_stream()
        # Prefetch look-ahead (nodes). A sweep found D=2 optimal: enough to hide
        # the next weight's H2D behind the current matmul, while holding fewer
        # weights in flight than a deep look-ahead -> both faster AND lower peak
        # (D=2: 7.38s/5.53GB vs D=8: 7.45s/5.80GB, D=16: 7.64s/6.00GB).
        D = 2
        up_evt: Dict[int, Any] = {}
        last_cmp = [None]

        def prefetch(k):
            if k >= n:
                return
            ws = [w for w in uploads.get(k, ()) if w not in env]
            if not ws:
                return
            if last_cmp[0] is not None:
                cs.wait_event(last_cmp[0])        # don't reuse a block a kernel still reads
            with cs:
                for w in ws:
                    self._upload_weight(w, env)
            up_evt[k] = cs.record()

        for k in range(D + 1):
            prefetch(k)
        for i, node in enumerate(self._topo):
            evt = up_evt.pop(i, None)
            if evt is not None:
                main.wait_event(evt)              # this node's weights are ready
            prefetch(i + D + 1)                   # keep the copy stream D nodes ahead
            self._exec_node(node, env)
            last_cmp[0] = main.record()
            for t in frees.get(i, ()):
                env.pop(t, None)
        main.synchronize()
        return self._collect_outputs(env)

    def _collect_outputs(self, env: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for t in self.graph.outputs:
            v = env.get(t.name)
            if v is None:
                continue
            out[t.name] = cp.asnumpy(v) if hasattr(v, "ndim") and not isinstance(v, np.ndarray) else np.asarray(v)
        return out

    def _upload_weight(self, name: str, env: Dict[str, Any]) -> None:
        """Upload one streamed float weight to the device before its first
        consumer (per ``plan.uploads_before``). Names already present -- int
        metadata seeded up front, or a value a Constant node produced -- are left
        as-is; a weight wired through Identity is uploaded under its source name
        here and aliased downstream when op_Identity runs."""
        if name in env:
            return
        host = self._init_host.get(name)
        if host is None:
            return
        env[name] = cp.asarray(host) if host.dtype.kind == "f" else host

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
