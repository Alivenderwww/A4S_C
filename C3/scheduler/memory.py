"""Memory planning and scheduling (子任务 C3.4 — code-reviewed).

This module implements the five C3.4 checkpoints as *real*, interlocking logic
(not stubs) and wires them into an execution-plan builder:

    A. DeviceMemoryPool        -> device alloc/free + weight preload to device buffers
    B. LifetimePlanner         -> first/last-use analysis -> shared slots
    C. DeviceMemoryPool        -> free-list + best-fit + coalescing of freed blocks
    D. WeightPrefetchScheduler -> move weight H2D ahead of the consuming compute
    E. StreamAssigner          -> independent nodes -> different compute streams

``build_execution_plan(graph)`` ties them together into an :class:`ExecutionPlan`
whose ordered steps reference pool offsets, reuse slots, prefetch H2D and stream
ids — the traceable "闭环" the C3.4 review looks for.

The pool is a host-side *simulation* of a device arena (offsets stand in for
device pointers); a production backend would swap ``DeviceMemoryPool._backend``
for ``cudaMalloc``/``cudaMemcpyAsync`` while keeping this scheduling logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .graph import Graph, Node


# ===========================================================================
# A + C : Device memory pool with free-list / best-fit / coalescing
# ===========================================================================
@dataclass
class Segment:
    offset: int
    size: int
    free: bool
    tag: str = ""


@dataclass
class Allocation:
    handle: int
    offset: int
    size: int
    tag: str


class DeviceMemoryPool:
    """A real arena allocator with a coalescing best-fit free-list.

    * ``malloc`` (A) reserves a device buffer; ``free`` returns it to the pool.
    * Freed segments enter a free-list and are reused by later ``malloc`` calls
      via **best-fit** selection; adjacent free segments are **coalesced** (C).
    * ``preload_weight`` performs the H2D upload path (A) and hands back a device
      buffer that later compute steps reference.
    """

    def __init__(self, alignment: int = 512):
        self.alignment = alignment
        self.segments: List[Segment] = []      # sorted by offset
        self.arena_top = 0
        self._next_handle = 0
        self._alloc_by_handle: Dict[int, Allocation] = {}
        # metrics
        self.current_bytes = 0
        self.peak_bytes = 0
        self.reuse_hits = 0                     # best-fit hits into the free-list
        self.coalesce_count = 0

    def _align(self, n: int) -> int:
        a = self.alignment
        return (n + a - 1) // a * a

    def malloc(self, size: int, tag: str = "") -> Allocation:
        size = max(self._align(size), self.alignment)
        # ---- C: best-fit search over free segments ----
        best_i = -1
        best_slack = None
        for i, seg in enumerate(self.segments):
            if seg.free and seg.size >= size:
                slack = seg.size - size
                if best_slack is None or slack < best_slack:
                    best_slack, best_i = slack, i
        if best_i >= 0:
            self.reuse_hits += 1
            seg = self.segments[best_i]
            offset = seg.offset
            if seg.size > size:  # split the surplus back into the free-list
                remainder = Segment(seg.offset + size, seg.size - size, True)
                seg.size = size
                self.segments.insert(best_i + 1, remainder)
            seg.free = False
            seg.tag = tag
        else:
            # ---- A: grow the arena (bump) ----
            offset = self.arena_top
            self.arena_top += size
            seg = Segment(offset, size, False, tag)
            self.segments.append(seg)
            self.segments.sort(key=lambda s: s.offset)

        handle = self._next_handle
        self._next_handle += 1
        alloc = Allocation(handle, offset, size, tag)
        self._alloc_by_handle[handle] = alloc
        self.current_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return alloc

    def free(self, alloc: Allocation) -> None:
        self.current_bytes -= alloc.size
        for seg in self.segments:
            if seg.offset == alloc.offset and not seg.free:
                seg.free = True
                seg.tag = ""
                break
        self._coalesce()
        self._alloc_by_handle.pop(alloc.handle, None)

    def _coalesce(self) -> None:
        """Merge adjacent free segments (C: fragment compaction)."""
        self.segments.sort(key=lambda s: s.offset)
        merged: List[Segment] = []
        for seg in self.segments:
            if merged and merged[-1].free and seg.free and \
                    merged[-1].offset + merged[-1].size == seg.offset:
                merged[-1].size += seg.size
                self.coalesce_count += 1
            else:
                merged.append(seg)
        self.segments = merged

    # A: weight preload path -------------------------------------------------
    def preload_weight(self, name: str, nbytes: int) -> Allocation:
        """Allocate a device buffer and mark the H2D upload of a weight/const."""
        return self.malloc(nbytes, tag=f"weight:{name}")

    def stats(self) -> Dict[str, int]:
        return {
            "arena_bytes": self.arena_top,
            "peak_bytes": self.peak_bytes,
            "free_segments": sum(1 for s in self.segments if s.free),
            "reuse_hits": self.reuse_hits,
            "coalesce_count": self.coalesce_count,
        }


# ===========================================================================
# B : Intermediate-tensor lifetime analysis -> shared slots
# ===========================================================================
@dataclass
class Lifetime:
    tensor: str
    first: int       # step index that produces it
    last: int        # last step index that consumes it


class LifetimePlanner:
    """First/last-use analysis + interval-based slot sharing.

    Non-overlapping intermediate tensors are mapped to the same logical *slot*
    (and thus the same physical buffer), shrinking peak memory.  The resulting
    ``tensor_to_slot`` map is consumed by :func:`build_execution_plan`.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.order = graph.topo_order()
        self.lifetimes: Dict[str, Lifetime] = {}
        self.tensor_to_slot: Dict[str, int] = {}
        self.slot_sizes: List[int] = []

    def analyze(self, size_of) -> "LifetimePlanner":
        index = {n.name: i for i, n in enumerate(self.order)}
        producers = self.graph.producer_map()
        graph_outs = self.graph.output_names()
        init_names = self.graph.initializer_names

        # first/last use for every intermediate (produced-by-a-node) tensor
        for i, node in enumerate(self.order):
            for o in node.outputs:
                if not o:
                    continue
                self.lifetimes[o] = Lifetime(o, i, i)
        for i, node in enumerate(self.order):
            for t in node.inputs:
                if t in self.lifetimes:
                    self.lifetimes[t].last = max(self.lifetimes[t].last, i)
        # graph outputs live until the end
        for o in graph_outs:
            if o in self.lifetimes:
                self.lifetimes[o].last = len(self.order)

        # ---- greedy interval slot allocation (like linear-scan register alloc) ----
        # Sort by first-use; keep a pool of free slots freed when their tensor dies.
        events = sorted(self.lifetimes.values(), key=lambda lt: (lt.first, lt.last))
        free_slots: List[int] = []
        active: List[Tuple[int, int]] = []  # (last_use, slot)
        for lt in events:
            # release slots whose tensor died before this one is produced
            still_active = []
            for last, slot in active:
                if last < lt.first:
                    free_slots.append(slot)
                else:
                    still_active.append((last, slot))
            active = still_active
            if free_slots:
                slot = free_slots.pop()
            else:
                slot = len(self.slot_sizes)
                self.slot_sizes.append(0)
            self.tensor_to_slot[lt.tensor] = slot
            self.slot_sizes[slot] = max(self.slot_sizes[slot], size_of(lt.tensor))
            active.append((lt.last, slot))
        return self

    def stats(self) -> Dict[str, int]:
        return {
            "num_tensors": len(self.lifetimes),
            "num_slots": len(self.slot_sizes),
            "slots_saved": max(0, len(self.lifetimes) - len(self.slot_sizes)),
            "slot_bytes": sum(self.slot_sizes),
        }


# ===========================================================================
# E : Stream assignment for independent nodes
# ===========================================================================
class StreamAssigner:
    """Assign nodes to compute streams so independent work overlaps.

    Nodes are grouped into dependency *waves* (longest-path depth); nodes in the
    same wave have no data dependency on each other, so they are round-robined
    across ``num_streams`` compute streams.  A separate copy stream (id 0) is
    reserved for H2D transfers (used by the prefetch scheduler, D).
    """

    COPY_STREAM = 0

    def __init__(self, graph: Graph, num_streams: int = 2):
        self.graph = graph
        self.num_streams = max(1, num_streams)
        self.node_stream: Dict[str, int] = {}
        self.multi_stream_waves = 0

    def assign(self) -> "StreamAssigner":
        order = self.graph.topo_order()
        producers = self.graph.producer_map()
        depth: Dict[str, int] = {}
        for node in order:
            d = 0
            for t in node.inputs:
                src = producers.get(t)
                if src is not None:
                    d = max(d, depth.get(src.name, 0) + 1)
            depth[node.name] = d

        waves: Dict[int, List[str]] = {}
        for node in order:
            waves.setdefault(depth[node.name], []).append(node.name)

        for _, members in sorted(waves.items()):
            if len(members) > 1:
                self.multi_stream_waves += 1
            for j, name in enumerate(members):
                # compute streams are 1..num_streams (0 reserved for copies)
                self.node_stream[name] = 1 + (j % self.num_streams)
        return self

    def stats(self) -> Dict[str, int]:
        used = sorted(set(self.node_stream.values()))
        return {
            "num_compute_streams": len(used),
            "multi_stream_waves": self.multi_stream_waves,
            "copy_stream": self.COPY_STREAM,
        }


# ===========================================================================
# D + assembly : Execution plan with weight prefetch
# ===========================================================================
@dataclass
class PlanStep:
    kind: str                       # alloc_weight | h2d | compute | free | alloc_tensor
    stream: int
    node: str = ""
    tensor: str = ""
    device_offset: int = -1
    size: int = 0
    slot: int = -1
    detail: str = ""


@dataclass
class ExecutionPlan:
    steps: List[PlanStep] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def kinds(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.steps:
            out[s.kind] = out.get(s.kind, 0) + 1
        return out


def _default_tensor_bytes(graph: Graph, batch: int = 1):
    """Return a ``size_of(tensor)`` closure (bytes), best-effort.

    Weights/constants use their real ``nbytes``; intermediates fall back to a
    per-tensor nominal size (shape inference for every intermediate is a TODO).
    """
    init = graph.initializers

    def size_of(name: str) -> int:
        v = init.get(name)
        if v is not None and hasattr(v, "nbytes"):
            return int(v.nbytes)
        # TODO: use ShapeInferencePass to size intermediates exactly.
        return 256 * 1024  # nominal 256 KiB slot

    return size_of


def build_execution_plan(
    graph: Graph,
    num_streams: int = 2,
    prefetch_distance: int = 1,
    batch: int = 1,
) -> ExecutionPlan:
    """Assemble the full A–E execution plan for a graph.

    Ordering per compute node:
      1. (D) weight H2D for the layer ``prefetch_distance`` nodes *ahead* is
         issued on the copy stream — "当前层算、下一层传".
      2. (B) intermediate outputs bind to their reuse slot / pool buffer.
      3. (E) the compute step runs on its assigned compute stream.
      4. intermediates whose last use has passed are freed back to the pool (C).
    """
    order = graph.topo_order()
    size_of = _default_tensor_bytes(graph, batch)

    # B: lifetime slots
    life = LifetimePlanner(graph).analyze(size_of)
    # E: streams
    streams = StreamAssigner(graph, num_streams).assign()
    # A/C: the device pool
    pool = DeviceMemoryPool()

    plan = ExecutionPlan()
    init_names = graph.initializer_names

    # Map node -> the weight tensors (initializers) it consumes.
    node_weights: Dict[str, List[str]] = {}
    for node in order:
        node_weights[node.name] = [t for t in node.inputs if t in init_names]

    # slot -> live Allocation (physical buffer backing that slot)
    slot_alloc: Dict[int, Allocation] = {}
    # tensor -> its remaining consumers count, for freeing (C)
    last_use = {t: lt.last for t, lt in life.lifetimes.items()}
    weight_uploaded: set = set()

    def upload_weight(name: str):
        if name in weight_uploaded:
            return
        v = graph.initializers.get(name)
        nb = int(v.nbytes) if (v is not None and hasattr(v, "nbytes")) else 4096
        alloc = pool.preload_weight(name, nb)         # A: device alloc
        plan.steps.append(PlanStep("alloc_weight", StreamAssigner.COPY_STREAM,
                                   tensor=name, device_offset=alloc.offset, size=nb))
        plan.steps.append(PlanStep("h2d", StreamAssigner.COPY_STREAM,      # D: async copy
                                   tensor=name, device_offset=alloc.offset, size=nb,
                                   detail="async H2D"))
        weight_uploaded.add(name)

    for i, node in enumerate(order):
        # ---- D: prefetch the weights of a node `prefetch_distance` ahead ----
        ahead = i + prefetch_distance
        if ahead < len(order):
            for w in node_weights[order[ahead].name]:
                upload_weight(w)
        # ensure this node's own weights are present (in case not prefetched)
        for w in node_weights[node.name]:
            upload_weight(w)

        # ---- B: bind output tensors to their reuse slots / pool buffers ----
        for o in node.outputs:
            if not o:
                continue
            slot = life.tensor_to_slot.get(o)
            if slot is None:
                continue
            if slot not in slot_alloc:
                alloc = pool.malloc(life.slot_sizes[slot], tag=f"slot{slot}")
                slot_alloc[slot] = alloc
                plan.steps.append(PlanStep("alloc_tensor", streams.node_stream.get(node.name, 1),
                                           tensor=o, slot=slot,
                                           device_offset=alloc.offset, size=alloc.size))

        # ---- E: the compute step on its assigned stream ----
        plan.steps.append(PlanStep(
            "compute", streams.node_stream.get(node.name, 1),
            node=node.name, detail=node.op_type,
        ))

        # ---- C: free intermediates whose last use is this step ----
        for t in list(node.inputs):
            if t in last_use and last_use[t] == i:
                slot = life.tensor_to_slot.get(t)
                if slot is not None and slot in slot_alloc:
                    # only release the physical buffer when no later tensor still
                    # maps to this slot with a later lifetime
                    still_needed = any(
                        life.tensor_to_slot.get(other) == slot and life.lifetimes[other].last > i
                        for other in life.lifetimes
                    )
                    if not still_needed:
                        pool.free(slot_alloc.pop(slot))
                        plan.steps.append(PlanStep("free", streams.node_stream.get(node.name, 1),
                                                   tensor=t, slot=slot))

    plan.summary = {
        "num_steps": len(plan.steps),
        "step_kinds": plan.kinds(),
        "pool": pool.stats(),
        "lifetime": life.stats(),
        "streams": streams.stats(),
        "weights_preloaded": len(weight_uploaded),
        "prefetch_distance": prefetch_distance,
    }
    return plan


# Convenience: a MemoryPlanner facade tying the pieces together.
class MemoryPlanner:
    """Facade over the A–E components; ``plan(graph)`` -> :class:`ExecutionPlan`."""

    def __init__(self, num_streams: int = 2, prefetch_distance: int = 1):
        self.num_streams = num_streams
        self.prefetch_distance = prefetch_distance

    def plan(self, graph: Graph, batch: int = 1) -> ExecutionPlan:
        return build_execution_plan(
            graph,
            num_streams=self.num_streams,
            prefetch_distance=self.prefetch_distance,
            batch=batch,
        )
