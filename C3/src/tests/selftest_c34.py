#!/usr/bin/env python3
"""C3.4 单元测试 (selftest_c34.py) — 对 memory scheduler 的生产路径做聚焦覆盖.

Five acceptance tests for the C3.4 scheduler (生产后应全部通过):
  1) Slot capacity: small(64B) 与 big(4096B) 映射同 slot 时 small 的 pool
     allocation 应 >= 4096 (按 slot 最大需求预分配).
  2) Lifetime free: small 在 last-use 后应存在 free PlanStep.
  3) Compute-step bindings: Gemm compute step 应携带 inputs/outputs
     metadata (weight source=weight + device_offset>=0, output source=slot
     + offset>=0).
  4) Zero-consumer tensor: 零消费者中间张量在 compute 后应被 free,
     释放的 slot 被后续 tensor 重新 alloc_tensor.
  5) Boundary condition: previous.last == next.first 时不得共享 slot.

退出码 0 表示全部通过; 有任一失败 → 退出码 1.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np

_C3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3_ROOT not in sys.path:
    sys.path.insert(0, _C3_ROOT)

from scheduler.graph import Graph, Node, TensorInfo
from scheduler.memory import (
    _shape_for_op,
    _infer_shapes,
    _default_tensor_bytes,
    build_execution_plan,
    LifetimePlanner,
    TensorBinding,
    PlanStep,
    ExecutionPlan,
    validate_execution_plan,
)

_PASS, _FAIL = 0, 0


def check(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {msg}")
    else:
        _FAIL += 1
        print(f"  FAIL  {msg}")
    return cond


# ===========================================================================
# Test 1: Slot capacity — small+big share slot, small alloc should be
#         slot-max-aware (>=4096).  Slot pre-allocates at max tenant size.
# ===========================================================================
def test_slot_capacity_shared_small_big():
    """三节点链 small(64B)->bridge(128B)->big(4096B).
    small 与 big 生命周期不重叠, 映射同 slot; bridge 占另一 slot.
    断言 small 的 shared slot allocation capacity >= 4096
    (按 slot 最大 tenant big(4096B) 预分配).
    """
    print("\n# Test 1: Slot capacity — small+big share slot, capacity >= 4096")
    sizes = {"small_t": 64, "bridge_t": 128, "big_t": 4096}

    g = Graph(
        nodes=[
            Node("small", "Relu", ["x"], ["small_t"]),         # step 0
            Node("bridge", "Relu", ["small_t"], ["bridge_t"]),  # step 1
            Node("big", "Relu", ["bridge_t"], ["big_t"]),       # step 2
        ],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("big_t")],
    )

    # Patch _default_tensor_bytes so size_of returns our custom sizes.
    with patch("scheduler.memory._default_tensor_bytes",
               return_value=lambda name: sizes.get(name, 256)):
        plan = build_execution_plan(g)

    # ---- Verify slot mapping ----
    size_of = lambda name: sizes.get(name, 256)
    life = LifetimePlanner(g).analyze(size_of)
    small_slot = life.tensor_to_slot.get("small_t")
    big_slot = life.tensor_to_slot.get("big_t")
    bridge_slot = life.tensor_to_slot.get("bridge_t")

    check(small_slot is not None, "small_t mapped to a slot")
    check(big_slot is not None, "big_t mapped to a slot")
    check(bridge_slot is not None, "bridge_t mapped to a slot")
    check(small_slot == big_slot,
          f"small_t (slot {small_slot}) and big_t (slot {big_slot}) share same slot")
    check(bridge_slot is not None and bridge_slot != small_slot,
          f"bridge_t (slot {bridge_slot}) occupies a different slot from small (slot {small_slot})")

    # ---- Assert capacity ----
    # alloc_tensor for small_t uses the slot's max tenant size (4096B from big_t)
    # so it pre-allocates a buffer large enough for any tensor in this slot.
    small_alloc = [s for s in plan.steps
                   if s.kind == "alloc_tensor" and s.tensor == "small_t"]
    check(len(small_alloc) >= 1, "alloc_tensor step exists for small_t")
    if small_alloc:
        check(small_alloc[0].size >= 4096,
              f"small alloc capacity >= 4096 (got {small_alloc[0].size})")


# ===========================================================================
# Test 2: Free step — small freed after last-use even when a future big
#         shares its slot.  death_events-based approach frees by last_use,
#         not by node.inputs scan, so still_needed is irrelevant.
# ===========================================================================
def test_small_freed_after_last_use():
    """small 在 last-use (step 1) 后应存在 free PlanStep.
    death_events 预构建了每个 step 应释放的 tensor 列表，不受同 slot 的
    big 是否仍存活影响。small freed by last_use 而非 node.inputs scan.
    """
    print("\n# Test 2: Free step — small freed after last-use despite future big")
    sizes = {"small_t": 64, "bridge_t": 128, "big_t": 4096}

    g = Graph(
        nodes=[
            Node("small", "Relu", ["x"], ["small_t"]),
            Node("bridge", "Relu", ["small_t"], ["bridge_t"]),
            Node("big", "Relu", ["bridge_t"], ["big_t"]),
        ],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("big_t")],
    )

    with patch("scheduler.memory._default_tensor_bytes",
               return_value=lambda name: sizes.get(name, 256)):
        plan = build_execution_plan(g)

    # Explicitly verify small_t and big_t share the same slot
    life = LifetimePlanner(g).analyze(lambda name: sizes.get(name, 256))
    small_slot = life.tensor_to_slot.get("small_t")
    big_slot = life.tensor_to_slot.get("big_t")
    check(small_slot is not None, "test2: small_t mapped to a slot")
    check(big_slot is not None, "test2: big_t mapped to a slot")
    check(small_slot == big_slot,
          f"test2: small_t (slot {small_slot}) and big_t (slot {big_slot}) share same slot")

    # small_t last-use is at step 1 (bridge consumes it).  death_events
    # schedules it for free at step 1; the occupant guard ensures the buffer
    # is reclaimed regardless of other tensors sharing the same slot.
    free_small = [s for s in plan.steps
                  if s.kind == "free" and s.tensor == "small_t"]
    check(len(free_small) >= 1,
          f"free PlanStep exists for small_t (found {len(free_small)})")

    # ── Order: bridge compute < free(small_t) < big alloc/compute ──
    bridge_compute_idx = next(
        (i for i, s in enumerate(plan.steps) if s.kind == "compute" and s.node == "bridge"),
        -1,
    )
    free_small_idx = next(
        (i for i, s in enumerate(plan.steps) if s.kind == "free" and s.tensor == "small_t"),
        -1,
    )
    big_first_idx = next(
        (i for i, s in enumerate(plan.steps)
         if (s.kind == "alloc_tensor" and s.tensor == "big_t")
         or (s.kind == "compute" and s.node == "big")),
        -1,
    )

    order_ok = (bridge_compute_idx >= 0 and free_small_idx >= 0 and big_first_idx >= 0
                and bridge_compute_idx < free_small_idx < big_first_idx)
    check(order_ok,
          f"Order: bridge compute (idx {bridge_compute_idx}) < free(small_t) (idx {free_small_idx})"
          f" < big alloc/compute (idx {big_first_idx})")


# ===========================================================================
# Invariant A: slot_sizes maps to max tenant size per slot.
# ===========================================================================
def test_slot_sizes_max_of_mapped_tensors():
    """Invariant: slot_sizes[slot] == max(size_of(t) for all t in slot).

    Uses the same small→bridge→big chain as Test 1/Test 2 — small(64B)
    and big(4096B) share a slot via non-overlapping lifetimes, bridge(128B)
    gets an independent slot.
    """
    print("\n# Invariant A: slot_sizes == max tenant per slot")
    sizes = {"small_t": 64, "bridge_t": 128, "big_t": 4096}

    g = Graph(
        nodes=[
            Node("small", "Relu", ["x"], ["small_t"]),         # step 0
            Node("bridge", "Relu", ["small_t"], ["bridge_t"]),  # step 1
            Node("big", "Relu", ["bridge_t"], ["big_t"]),       # step 2
        ],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("big_t")],
    )

    size_of = lambda n: sizes.get(n, 256)
    life = LifetimePlanner(g).analyze(size_of)

    # For each slot, the recorded slot_sizes must be exactly the max size
    # of any tensor mapped to it.
    for slot_idx in range(len(life.slot_sizes)):
        mapped = [t for t, s in life.tensor_to_slot.items() if s == slot_idx]
        expected = max(size_of(t) for t in mapped)
        check(life.slot_sizes[slot_idx] == expected,
              f"slot_sizes[{slot_idx}] == {expected} (got {life.slot_sizes[slot_idx]})")


# ===========================================================================
# Invariant B: overlapping lifetimes (last == first) must get distinct slots.
# ===========================================================================
def test_slot_no_share_on_overlap():
    """Invariant: two tensors whose lifetimes overlap (last >= other.first)
    must have distinct slots.  Specifically when last == first for both,
    the strict inequality last < lt.first prevents slot reuse.
    """
    print("\n# Invariant B: overlapping lifetimes → distinct slots")

    # Two outputs produced by same node, both consumed by same next node.
    # Both have first=0, last=1 — fully overlapping.
    g = Graph(
        nodes=[
            Node("n1", "Relu", ["x"], ["A", "B"]),   # step 0
            Node("n2", "Add",  ["A", "B"], ["C"]),    # step 1
        ],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("C")],
    )

    life = LifetimePlanner(g).analyze(lambda n: 256)

    slot_a = life.tensor_to_slot.get("A")
    slot_b = life.tensor_to_slot.get("B")
    check(slot_a is not None and slot_b is not None,
          "A and B both have slots assigned")
    check(slot_a != slot_b,
          f"A (slot {slot_a}) and B (slot {slot_b}) have distinct slots (overlap → no share)")


# ===========================================================================
# Test: Zero-consumer tensor — produced but never consumed, must be freed
#       so a later tensor sharing the same slot gets a fresh alloc_tensor.
# ===========================================================================
def test_zero_consumer_tensor():
    """Zero-consumer tensor dead_t (produced but never consumed as any node's
    input) must appear in death_events and generate a free PlanStep.  The
    released slot can then be reused by a later tensor, which must get a new
    alloc_tensor record (not reuse a stale buffer).

    Graph: n0 produces dead_t (unused), n1 produces stuff, n2 produces out.
    dead_t and out share the same slot via non-overlapping lifetimes.
    """
    print("\n# Test: Zero-consumer tensor — dead_t freed, out gets new alloc_tensor")
    sizes = {"dead_t": 64, "stuff": 128, "out": 256}

    g = Graph(
        nodes=[
            Node("n0", "Relu", ["x"], ["dead_t"]),     # step 0: dead_t never consumed
            Node("n1", "Relu", ["x"], ["stuff"]),       # step 1: unrelated intermediate
            Node("n2", "Relu", ["x"], ["out"]),         # step 2: graph output
        ],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("out")],
    )

    with patch("scheduler.memory._default_tensor_bytes",
               return_value=lambda name: sizes.get(name, 256)):
        plan = build_execution_plan(g)

    # Verify slot sharing: dead_t and out share the same slot
    size_of = lambda name: sizes.get(name, 256)
    life = LifetimePlanner(g).analyze(size_of)
    dead_slot = life.tensor_to_slot.get("dead_t")
    out_slot = life.tensor_to_slot.get("out")
    check(dead_slot is not None, "dead_t mapped to a slot")
    check(out_slot is not None, "out mapped to a slot")
    check(dead_slot == out_slot,
          f"dead_t (slot {dead_slot}) and out (slot {out_slot}) share same slot")

    # 1) free step exists for dead_t
    free_dead = [s for s in plan.steps
                 if s.kind == "free" and s.tensor == "dead_t"]
    check(len(free_dead) >= 1,
          f"free step exists for dead_t (found {len(free_dead)})")

    # 2) alloc_tensor step exists for out (not sharing stale buffer)
    alloc_out = [s for s in plan.steps
                 if s.kind == "alloc_tensor" and s.tensor == "out"]
    check(len(alloc_out) >= 1,
          f"alloc_tensor step exists for out (found {len(alloc_out)})")

    # 3) free(dead_t) before alloc_tensor(out) in plan-step order
    if free_dead and alloc_out:
        free_idx = next(
            i for i, s in enumerate(plan.steps)
            if s.kind == "free" and s.tensor == "dead_t"
        )
        alloc_idx = next(
            i for i, s in enumerate(plan.steps)
            if s.kind == "alloc_tensor" and s.tensor == "out"
        )
        check(free_idx < alloc_idx,
              f"free(dead_t) at plan step {free_idx}"
              f" before alloc_tensor(out) at {alloc_idx}")


# ===========================================================================
# Test: Boundary — when previous.last == next.first (same-step overlap /
#       last==first) the slots must be distinct; they are not eligible
#       for sharing.
# ===========================================================================
def test_boundary_no_share_on_same_step_overlap():
    """When previous.last == next.first (same-step overlap / last==first),
    the slots must be distinct: the linear-scan allocator uses the strict
    condition last < lt.first, so equality keeps the slot in the active set
    and the two tensors cannot share a slot.

    A: first=0, last=1 (consumed by n1 at step 1)
    B: first=1, last=2 (consumed by n2 at step 2)
    A.last (1) == B.first (1)  →  no slot sharing (same-step overlap).
    """
    print("\n# Test: Boundary — same-step overlap / last==first → no slot sharing")

    g = Graph(
        nodes=[
            Node("n0", "Relu", ["x"], ["A"]),     # step 0
            Node("n1", "Relu", ["A"], ["B"]),      # step 1: A last-used, B produced
            Node("n2", "Relu", ["B"], ["C"]),      # step 2: graph output
        ],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("C")],
    )

    life = LifetimePlanner(g).analyze(lambda n: 256)

    # A: first=0, last=1 (last-use at n1, step 1)
    # B: first=1, last=2 (last-use at n2, step 2)
    # A.last == B.first == 1 — should NOT share a slot
    slot_a = life.tensor_to_slot.get("A")
    slot_b = life.tensor_to_slot.get("B")
    check(slot_a is not None, "A has a slot")
    check(slot_b is not None, "B has a slot")
    check(slot_a != slot_b,
          f"A (slot {slot_a}) and B (slot {slot_b}) are distinct"
          f" (same-step overlap last==first, no share)")


# ===========================================================================
# Test 3: Compute-step bindings — Gemm compute step should carry
#         input/output metadata that traces weights and slot intermediates.
#         Current PlanStep has no inputs/outputs fields → FAIL gracefully.
# ===========================================================================
def test_compute_step_bindings():
    """Gemm 图: compute step 应有 inputs/outputs bindings，逐 tensor 验证。

    graph input A: source='graph_input'
    initializer W: source='weight', device_offset/size == alloc_weight(W)
    initializer B: source='weight', device_offset/size == alloc_weight(B)
    output    out: source='slot',    slot/device_offset/size == alloc_tensor(out)

    使用 dataclass 属性读取，不做松散 dict 回退。
    """
    print("\n# Test 3: Compute-step bindings — Gemm inputs/outputs metadata")
    W = np.eye(4, dtype=np.float32)
    B = np.ones(4, dtype=np.float32)
    init = {"W": W, "B": B}

    g = Graph(
        nodes=[Node("gemm", "Gemm", ["A", "W", "B"], ["out"])],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers=init,
    )

    plan = build_execution_plan(g)

    # Find the compute step for gemm
    compute_steps = [s for s in plan.steps
                     if s.kind == "compute" and s.node == "gemm"]
    check(len(compute_steps) >= 1, "compute step for Gemm exists")
    if not compute_steps:
        return

    step = compute_steps[0]

    # Verify inputs/outputs are TensorBinding lists (not loose dicts)
    check(isinstance(step.inputs, list) and all(isinstance(b, TensorBinding) for b in step.inputs),
          "compute step inputs is List[TensorBinding]")
    check(isinstance(step.outputs, list) and all(isinstance(b, TensorBinding) for b in step.outputs),
          "compute step outputs is List[TensorBinding]")

    # Locate alloc_weight and alloc_tensor steps for cross-referencing
    alloc_w = {s.tensor: s for s in plan.steps if s.kind == "alloc_weight"}
    alloc_out = [s for s in plan.steps
                 if s.kind == "alloc_tensor" and s.tensor == "out"]

    # ── inputs bindings ────────────────────────────────────────────────
    check(len(step.inputs) > 0,
          "compute step has inputs bindings (source/device_offset)")

    # Graph input A: source='graph_input'
    a_binding = next(
        (b for b in step.inputs
         if b.tensor == "A" and b.source == "graph_input"),
        None,
    )
    check(a_binding is not None,
          "input A source='graph_input'")

    # Initializer W: source='weight', device_offset/size == alloc_weight(W)
    w_binding = next(
        (b for b in step.inputs
         if b.tensor == "W" and b.source == "weight"),
        None,
    )
    check(w_binding is not None,
          "weight W source='weight'")
    w_step_ok = check("W" in alloc_w,
                      "alloc_weight(W) step present for cross-ref")
    if w_binding is not None and w_step_ok:
        aw = alloc_w["W"]
        check(w_binding.device_offset >= 0,
              f"weight W device_offset ({w_binding.device_offset}) >= 0")
        check(w_binding.size > 0,
              f"weight W size ({w_binding.size}) > 0")
        check(w_binding.device_offset == aw.device_offset,
              f"weight W device_offset ({w_binding.device_offset})"
              f" == alloc_weight ({aw.device_offset})")
        check(w_binding.size == aw.size,
              f"weight W size ({w_binding.size})"
              f" == alloc_weight ({aw.size})")

    # Initializer B: source='weight', device_offset/size == alloc_weight(B)
    b_binding = next(
        (b for b in step.inputs
         if b.tensor == "B" and b.source == "weight"),
        None,
    )
    check(b_binding is not None,
          "weight B source='weight'")
    b_step_ok = check("B" in alloc_w,
                      "alloc_weight(B) step present for cross-ref")
    if b_binding is not None and b_step_ok:
        aw = alloc_w["B"]
        check(b_binding.device_offset >= 0,
              f"weight B device_offset ({b_binding.device_offset}) >= 0")
        check(b_binding.size > 0,
              f"weight B size ({b_binding.size}) > 0")
        check(b_binding.device_offset == aw.device_offset,
              f"weight B device_offset ({b_binding.device_offset})"
              f" == alloc_weight ({aw.device_offset})")
        check(b_binding.size == aw.size,
              f"weight B size ({b_binding.size})"
              f" == alloc_weight ({aw.size})")

    # ── outputs bindings ───────────────────────────────────────────────
    check(len(step.outputs) > 0,
          "compute step has outputs bindings (source/offset)")

    # Output out: source='slot', slot/device_offset/size == alloc_tensor(out)
    o_binding = next(
        (b for b in step.outputs
         if b.tensor == "out" and b.source == "slot"),
        None,
    )
    check(o_binding is not None,
          "output out source='slot'")
    out_step_ok = check(len(alloc_out) >= 1,
                        "alloc_tensor(out) step present for cross-ref")
    if o_binding is not None and out_step_ok:
        ao = alloc_out[0]
        check(o_binding.slot >= 0,
              f"output out slot ({o_binding.slot}) >= 0")
        check(o_binding.device_offset >= 0,
              f"output out device_offset ({o_binding.device_offset}) >= 0")
        check(o_binding.size > 0,
              f"output out size ({o_binding.size}) > 0")
        check(o_binding.slot == ao.slot,
              f"output out slot ({o_binding.slot})"
              f" == alloc_tensor ({ao.slot})")
        check(o_binding.device_offset == ao.device_offset,
              f"output out device_offset ({o_binding.device_offset})"
              f" == alloc_tensor ({ao.device_offset})")
        check(o_binding.size == ao.size,
              f"output out size ({o_binding.size})"
              f" == alloc_tensor ({ao.size})")


# ===========================================================================
# Test: Validator negative — catches corrupted weight offset
# ===========================================================================
def test_validator_catches_corrupted_weight_offset():
    """Validator must reject a plan with a corrupted weight device_offset."""
    print("\n# Validator negative: corrupted weight offset → ValueError")
    W = np.eye(4, dtype=np.float32)
    B = np.ones(4, dtype=np.float32)
    g = Graph(
        nodes=[Node("gemm", "Gemm", ["A", "W", "B"], ["out"])],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers={"W": W, "B": B},
    )
    plan = build_execution_plan(g)

    # Corrupt the weight W binding in the compute step
    compute_step = next(s for s in plan.steps if s.kind == "compute")
    for b in compute_step.inputs:
        if b.tensor == "W":
            b.device_offset = -99
            break

    try:
        validate_execution_plan(plan, g)
        check(False,
              "Validator raised ValueError for corrupted weight offset")
    except ValueError:
        check(True,
              "Validator caught corrupted weight offset")


# ===========================================================================
# Test: Validator negative — catches missing output binding
# ===========================================================================
def test_validator_catches_missing_output_binding():
    """Validator must reject a plan with a removed compute output binding."""
    print("\n# Validator negative: missing output binding → ValueError")
    W = np.eye(4, dtype=np.float32)
    B = np.ones(4, dtype=np.float32)
    g = Graph(
        nodes=[Node("gemm", "Gemm", ["A", "W", "B"], ["out"])],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers={"W": W, "B": B},
    )
    plan = build_execution_plan(g)

    # Remove the output binding for 'out'
    compute_step = next(s for s in plan.steps if s.kind == "compute")
    compute_step.outputs = [
        b for b in compute_step.outputs if b.tensor != "out"
    ]

    try:
        validate_execution_plan(plan, g)
        check(False,
              "Validator raised ValueError for missing output binding")
    except ValueError:
        check(True,
              "Validator caught missing output binding")


# ===========================================================================
# C3.4 Task4 — Shape inference for Flatten / Gather / Reshape / Split
# ===========================================================================
def test_flatten_shape_inference():
    """Flatten [2,3,4] axis=1 -> [2,12]; byte=96 (fp32)."""
    print("\n# Task4: Flatten shape inference")
    g = Graph(
        nodes=[Node("flt", "Flatten", ["x"], ["y"], attrs={"axis": 1})],
        inputs=[TensorInfo("x", shape=[2, 3, 4])],
        outputs=[TensorInfo("y")],
    )
    shapes = _infer_shapes(g)
    check(shapes.get("y") == [2, 12],
          "Flatten [2,3,4] axis=1 -> [2,12]")
    size_of = _default_tensor_bytes(g)
    check(size_of("y") == 2 * 12 * 4,
          "Flatten output byte size = 96 (fp32)")


def test_gather_shape_inference():
    """Gather data[2,5,4] + indices[3,2] axis=1 -> [2,3,2,4]."""
    print("\n# Task4: Gather shape inference")
    indices = np.ones((3, 2), dtype=np.int64)

    # Normal axis
    g = Graph(
        nodes=[Node("gth", "Gather", ["data", "indices"], ["y"], attrs={"axis": 1})],
        inputs=[TensorInfo("data", shape=[2, 5, 4])],
        outputs=[TensorInfo("y")],
        initializers={"indices": indices},
    )
    shapes = _infer_shapes(g)
    check(shapes.get("y") == [2, 3, 2, 4],
          "Gather [2,5,4]+[3,2] axis=1 -> [2,3,2,4]")

    # Negative axis
    g2 = Graph(
        nodes=[Node("gth2", "Gather", ["data", "indices"], ["y2"], attrs={"axis": -1})],
        inputs=[TensorInfo("data", shape=[2, 5, 4])],
        outputs=[TensorInfo("y2")],
        initializers={"indices": indices},
    )
    shapes2 = _infer_shapes(g2)
    check(shapes2.get("y2") == [2, 5, 3, 2],
          "Gather [2,5,4]+[3,2] axis=-1 -> [2,5,3,2]")

    # Scalar indices (shape=[])
    scalar_idx = np.array(3, dtype=np.int64)
    g3 = Graph(
        nodes=[Node("gth3", "Gather", ["data", "scalar"], ["y3"], attrs={"axis": 1})],
        inputs=[TensorInfo("data", shape=[2, 5, 4])],
        outputs=[TensorInfo("y3")],
        initializers={"scalar": scalar_idx},
    )
    shapes3 = _infer_shapes(g3)
    check(shapes3.get("y3") == [2, 4],
          "Gather scalar indices -> axis dim removed [2,4]")


def test_reshape_shape_inference():
    """Reshape [2,3,4] target=[0,-1] -> [2,12] + fail-closed invalid cases."""
    print("\n# Task4: Reshape shape inference")

    # Normal 0-copy + -1 inference
    g = Graph(
        nodes=[Node("rsp", "Reshape", ["x", "target"], ["y"])],
        inputs=[TensorInfo("x", shape=[2, 3, 4])],
        outputs=[TensorInfo("y")],
        initializers={"target": np.array([0, -1], dtype=np.int64)},
    )
    shapes = _infer_shapes(g)
    check(shapes.get("y") == [2, 12],
          "Reshape [2,3,4] target=[0,-1] -> [2,12]")

    # Invalid: multiple -1 -> fail-closed (shape unchanged from init [])
    g2 = Graph(
        nodes=[Node("rsp2", "Reshape", ["x", "target2"], ["y2"])],
        inputs=[TensorInfo("x", shape=[2, 3, 4])],
        outputs=[TensorInfo("y2")],
        initializers={"target2": np.array([-1, -1, 6], dtype=np.int64)},
    )
    shapes2 = _infer_shapes(g2)
    check(shapes2.get("y2") == [],
          "Reshape multi -1: output shape unchanged (fail-closed)")

    # Invalid: element count mismatch -> fail-closed
    g3 = Graph(
        nodes=[Node("rsp3", "Reshape", ["x", "target3"], ["y3"])],
        inputs=[TensorInfo("x", shape=[2, 3, 4])],
        outputs=[TensorInfo("y3")],
        initializers={"target3": np.array([5, 10], dtype=np.int64)},
    )
    shapes3 = _infer_shapes(g3)
    check(shapes3.get("y3") == [],
          "Reshape element count mismatch: shape unchanged (fail-closed)")


def test_split_shape_inference():
    """Split uneven dim=10 n=3 -> [4,3,3]; explicit + invalid sum."""
    print("\n# Task4: Split shape inference")

    # Uneven split (divmod)
    g = Graph(
        nodes=[Node("spl", "Split", ["x"], ["y0", "y1", "y2"], attrs={"axis": 0})],
        inputs=[TensorInfo("x", shape=[10])],
        outputs=[TensorInfo("y0"), TensorInfo("y1"), TensorInfo("y2")],
    )
    shapes = _infer_shapes(g)
    check(shapes.get("y0") == [4],
          "Split [10] n=3 -> y0=[4]")
    check(shapes.get("y1") == [3],
          "Split [10] n=3 -> y1=[3]")
    check(shapes.get("y2") == [3],
          "Split [10] n=3 -> y2=[3]")

    # Explicit split via attrs
    g2 = Graph(
        nodes=[Node("spl2", "Split", ["x2"], ["z0", "z1", "z2"],
                     attrs={"axis": 1, "split": [2, 3, 5]})],
        inputs=[TensorInfo("x2", shape=[5, 10])],
        outputs=[TensorInfo("z0"), TensorInfo("z1"), TensorInfo("z2")],
    )
    shapes2 = _infer_shapes(g2)
    check(shapes2.get("z0") == [5, 2],
          "Split [5,10] split=[2,3,5] -> z0=[5,2]")
    check(shapes2.get("z1") == [5, 3],
          "Split [5,10] split=[2,3,5] -> z1=[5,3]")
    check(shapes2.get("z2") == [5, 5],
          "Split [5,10] split=[2,3,5] -> z2=[5,5]")

    # Invalid split sum (9 != 10) -> fail-closed (shapes unchanged from init [])
    g3 = Graph(
        nodes=[Node("spl3", "Split", ["x3"], ["w0", "w1", "w2"],
                     attrs={"axis": 0, "split": [3, 3, 3]})],
        inputs=[TensorInfo("x3", shape=[10])],
        outputs=[TensorInfo("w0"), TensorInfo("w1"), TensorInfo("w2")],
    )
    shapes3 = _infer_shapes(g3)
    check(shapes3.get("w0") == [],
          "Split invalid sum: output shape unchanged (fail-closed)")


# ===========================================================================
# C3.4 Task5 — 三模型 end-to-end: A–E evidence 验证 (fail-closed 门禁)
# ===========================================================================

def _resolve_models_dir():
    """Locate the public models directory. Same search order as selftest_c32.py."""
    candidates = [
        os.path.join(_C3_ROOT, "..", "public", "Track-C",
                     "C3-scheduler", "testcases", "release_to_competitors", "models"),
        os.path.join(_C3_ROOT, "..", "public", "Agentic4SystemSummerSchoolContest",
                     "Track-C", "C3-scheduler", "testcases", "release_to_competitors", "models"),
        _C3_ROOT,
        os.path.join(_C3_ROOT, "models"),
    ]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith(".onnx") for f in os.listdir(d)):
            return os.path.normpath(d)
    return None


_MODELS_DIR = _resolve_models_dir()
_THREE_MODELS = {
    "MLP":  ("mlp_v1.onnx",  "mnist_mlp"),
    "ResNet":  ("resnet_v1.onnx",  "cifar_resnet18"),
    "Transformer": ("transformer_v1.onnx", "transformer"),
}


def test_three_model_e2e():
    """三模型 end-to-end A–E 门禁.
    
    对 MLP / ResNet / Transformer 分别执行 import_onnx_graph + 
    build_execution_plan，验证 plan.summary['c3d_evidence'] 精确包含 A–E 
    五块且各维度的关键信号真实非零.
    """
    from scheduler import import_onnx_graph
    from scheduler.memory import build_execution_plan

    if _MODELS_DIR is None:
        check(False, "[三模型] 模型目录不存在 — fail-closed")
        return

    print(f"\n# 三模型 end-to-end (models dir: {_MODELS_DIR})")

    # 累积全模型指标用于跨模型聚合断言
    all_reuse_hits = 0
    all_defrag_runs = 0

    for tag, (fname, _) in _THREE_MODELS.items():
        onnx_path = os.path.join(_MODELS_DIR, fname)
        if not os.path.exists(onnx_path):
            check(False, f"[{tag}] 模型缺失: {onnx_path}")
            continue

        print(f"\n## {tag} ({fname})")

        # Load and plan
        graph = import_onnx_graph(onnx_path)
        plan = build_execution_plan(graph)
        ev = plan.summary.get("c3d_evidence")
        check(ev is not None, f"[{tag}] c3d_evidence exists in plan.summary")

        if ev is None:
            continue  # 无法继续

        # ====================================================================
        #  A: Device pool — alloc_weight / h2d / compute bindings
        # ====================================================================
        A = ev.get("A_device_pool", {})
        check("A_device_pool" in ev, f"[{tag}] A_device_pool 块存在")
        check(A.get("validation_passed") is True,
              f"[{tag}] validation_passed is True")

        aw = A.get("alloc_weight_steps", 0)
        h2 = A.get("h2d_steps", 0)
        wo = A.get("weights_on_device", 0)
        check(aw == h2 == wo > 0,
              f"[{tag}] alloc_weight({aw}) == h2d({h2}) == weights_on_device({wo}) > 0")

        cs = A.get("compute_steps", 0)
        csb = A.get("compute_steps_with_bindings", 0)
        total_nodes = len(graph.topo_order())
        check(cs > 0, f"[{tag}] compute_steps({cs}) > 0")
        check(csb == cs,
              f"[{tag}] compute_steps_with_bindings({csb}) == compute_steps({cs})")
        check(cs == total_nodes,
              f"[{tag}] compute_steps({cs}) == topo_order nodes({total_nodes})")

        wb = A.get("weight_bindings", 0)
        sb = A.get("slot_bindings", 0)
        check(wb > 0, f"[{tag}] weight_bindings({wb}) > 0")
        check(sb > 0, f"[{tag}] slot_bindings({sb}) > 0")

        # ====================================================================
        #  B: Lifetime reuse — slot mapping & alloc_tensor integrity
        # ====================================================================
        B = ev.get("B_lifetime_reuse", {})
        check("B_lifetime_reuse" in ev, f"[{tag}] B_lifetime_reuse 块存在")

        nt = B.get("num_tensors", 0)
        ns = B.get("num_slots", 0)
        check(nt > 0, f"[{tag}] num_tensors({nt}) > 0")
        check(ns > 0, f"[{tag}] num_slots({ns}) > 0")
        check(ns < nt, f"[{tag}] num_slots({ns}) < num_tensors({nt}) — 确有复用")

        # 每个 alloc_tensor 步的 size>0, offset>=0, slot>=0
        alloc_t_steps = [s for s in plan.steps if s.kind == "alloc_tensor"]
        check(len(alloc_t_steps) > 0,
              f"[{tag}] alloc_tensor steps exist ({len(alloc_t_steps)})")
        for s in alloc_t_steps:
            check(s.size > 0, f"[{tag}] alloc_tensor({s.tensor}) size({s.size}) > 0")
            check(s.device_offset >= 0,
                  f"[{tag}] alloc_tensor({s.tensor}) offset({s.device_offset}) >= 0")
            check(s.slot >= 0,
                  f"[{tag}] alloc_tensor({s.tensor}) slot({s.slot}) >= 0")

        # ====================================================================
        #  C: Defrag — free steps / strategy / reuse_hits / defrag_runs
        # ====================================================================
        C = ev.get("C_defrag", {})
        check("C_defrag" in ev, f"[{tag}] C_defrag 块存在")

        fs = C.get("free_steps", 0)
        check(fs > 0, f"[{tag}] free_steps({fs}) > 0")

        strat = C.get("strategy", "")
        check("best-fit" in strat, f"[{tag}] strategy 含 best-fit")
        check("coalesce" in strat, f"[{tag}] strategy 含 coalesce")
        check("defragment" in strat, f"[{tag}] strategy 含 defragment")

        all_reuse_hits += C.get("reuse_hits", 0)
        all_defrag_runs += C.get("defrag_runs", 0)

        # ====================================================================
        #  D: Prefetch — h2d_total / prefetch_distance / interleaved_ratio
        # ====================================================================
        D = ev.get("D_prefetch", {})
        check("D_prefetch" in ev, f"[{tag}] D_prefetch 块存在")

        h2d_total = D.get("h2d_total", 0)
        check(h2d_total > 0, f"[{tag}] h2d_total({h2d_total}) > 0")
        check(D.get("prefetch_distance") == 2,
              f"[{tag}] prefetch_distance = {D.get('prefetch_distance')} (expect 2)")

        ir = D.get("interleaved_ratio", 0.0)
        check(ir > 0, f"[{tag}] interleaved_ratio({ir}) > 0")

        # ====================================================================
        #  E: Streams — compute_streams_used 不含 0
        # ====================================================================
        E = ev.get("E_streams", {})
        check("E_streams" in ev, f"[{tag}] E_streams 块存在")

        csu = E.get("compute_streams_used", [])
        check(len(csu) > 0, f"[{tag}] compute_streams_used 非空 ({csu})")
        check(0 not in csu,
              f"[{tag}] compute_streams_used 不含 0 ({csu})")

        # ResNet/Transformer: [1,2] expected; MLP may only have [1]
        if tag == "MLP":
            check(len(csu) >= 1, f"[{tag}] compute_streams_used >=1 ({csu})")
        else:
            check(sorted(csu) == [1, 2],
                  f"[{tag}] compute_streams_used == [1,2] (got {csu})")

        # ---- 每个 compute step 须有精确非空 binding ----
        comp_steps = [s for s in plan.steps if s.kind == "compute"]
        for s in comp_steps:
            node_obj = next((n for n in graph.nodes if n.name == s.node), None)
            if node_obj is None:
                check(False, f"[{tag}] compute node '{s.node}' not found in graph")
                continue
            # 按 graph 节点定义: 期望 inputs 有 binding, outputs 有 binding
            n_inputs = [t for t in node_obj.inputs if t]
            n_outputs = [t for t in node_obj.outputs if t]
            b_in = {b.tensor for b in s.inputs}
            b_out = {b.tensor for b in s.outputs}
            missing_in = set(n_inputs) - b_in
            missing_out = set(n_outputs) - b_out
            check(len(missing_in) == 0,
                  f"[{tag}] step '{s.node}': all inputs bound (missing: {missing_in})")
            check(len(missing_out) == 0,
                  f"[{tag}] step '{s.node}': all outputs bound (missing: {missing_out})")

        # ====================================================================
        #  一行摘要
        # ====================================================================
        print(f"  >>> A(w={aw} h2d={h2} cs={cs} wb={wb} sb={sb})"
              f"  B(t={nt} sl={ns})"
              f"  C(free={fs} reuse={C.get('reuse_hits',0)} defrag={C.get('defrag_runs',0)})"
              f"  D(h2d={h2d_total} pd={D.get('prefetch_distance')} ir={ir})"
              f"  E(streams={csu})")

    # ---- 跨模型聚合断言: 三模型合计存在真实 reuse_hits 和 defrag_runs ----
    print(f"\n# 三模型汇总: reuse_hits={all_reuse_hits}  defrag_runs={all_defrag_runs}")
    check(all_reuse_hits > 0,
          f"三模型合计 reuse_hits({all_reuse_hits}) > 0 (至少一个模型含 best-fit 重用)")
    check(all_defrag_runs > 0,
          f"三模型合计 defrag_runs({all_defrag_runs}) > 0 (至少一个模型触发 wave 压缩)")


# ===========================================================================
# main
# ===========================================================================
def main() -> int:
    print("=" * 60)
    print("C3.4 单元测试 (selftest_c34.py)")
    print("生产验收 — 全部 GREEN 方可通过")
    print("=" * 60)

    test_slot_capacity_shared_small_big()
    test_small_freed_after_last_use()
    test_slot_sizes_max_of_mapped_tensors()
    test_slot_no_share_on_overlap()
    test_zero_consumer_tensor()
    test_boundary_no_share_on_same_step_overlap()
    test_compute_step_bindings()
    test_validator_catches_corrupted_weight_offset()
    test_validator_catches_missing_output_binding()

    # C3.4 Task4 — shape inference
    test_flatten_shape_inference()
    test_gather_shape_inference()
    test_reshape_shape_inference()
    test_split_shape_inference()

    # C3.4 Task5 — 三模型 end-to-end A–E evidence
    test_three_model_e2e()

    print(f"\n=== {_PASS} passed, {_FAIL} failed ===")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
