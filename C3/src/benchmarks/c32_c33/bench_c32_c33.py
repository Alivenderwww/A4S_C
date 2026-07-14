#!/usr/bin/env python3
"""Local self-scoring harness for C3.2 (D1-D5) and C3.3 (F1-F4).

The *official* grader ships its own hidden ``bench_c32_c33.py``; this file
mirrors the rubric in spec.md so you can self-assess before submitting.  It only
uses the public API the real grader uses:

    import_onnx_graph / strategy.select_precision / strategy.decompose /
    strategy.tune_kernel / hardware.* / GraphPassPipeline

Usage:

    python benchmarks/c32_c33/bench_c32_c33.py \
        --models mnist_mlp cifar_resnet18 transformer \
        --output-dir benchmarks/c32_c33/results

Notes:
  * The real C3.2/C3.3 grader evaluates on ``mnist_mlp`` + ``cifar_resnet18``.
    We also include ``transformer`` by default because Softmax/LayerNorm and the
    fusion patterns only appear there — handy for a complete self-check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_C3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _C3_ROOT not in sys.path:
    sys.path.insert(0, _C3_ROOT)
# C3/ (parent of src/) is where models and testdata live when the grader runs
# with C3/ as the working directory.
_C3_TOP = os.path.dirname(_C3_ROOT)

from scheduler import import_onnx_graph, strategy, hardware, GraphPassPipeline
from scheduler.graph_passes.fusion import FusionPass
from runtime.mock_runtime import MockRuntime

# Resolve model directory: models live in C3/ (the working directory), not
# C3/src/. Search C3_TOP first, then C3_ROOT, then public/ fallbacks.
def _resolve_models_dir():
    candidates = [
        # C3/ top-level (grader working dir): C3/*.onnx
        _C3_TOP,
        # C3_ROOT (src/) itself
        _C3_ROOT,
        # C3_TOP/models
        os.path.join(_C3_TOP, "models"),
        # public/ fallbacks
        os.path.join(_C3_TOP, "public", "Track-C",
                     "C3-scheduler", "testcases", "release_to_competitors", "models"),
        os.path.join(_C3_TOP, "public", "Agentic4SystemSummerSchoolContest",
                     "Track-C", "C3-scheduler", "testcases", "release_to_competitors", "models"),
    ]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith(".onnx") for f in os.listdir(d)):
            return os.path.normpath(d)
    # All candidates exhausted → fail clearly
    raise FileNotFoundError(
        f"C3.2 benchmark: no model directory found among {candidates}. "
        "Place .onnx files in one of these locations or point --models at a custom path."
    )


_MODELS_DIR = _resolve_models_dir()
MODEL_FILES = {
    "mnist_mlp": "mlp_v1.onnx",
    "cifar_resnet18": "resnet_v1.onnx",
    "transformer": "transformer_v1.onnx",
}
INPUT_SHAPES = {
    "mnist_mlp": ("input", np.float32, (2, 1, 28, 28)),
    "cifar_resnet18": ("input", np.float32, (2, 3, 32, 32)),
    "transformer": ("input_ids", np.int64, (2, 18)),
}

SENSITIVE = {"Softmax", "LayerNormalization", "LayerNorm", "BatchNormalization",
             "ReduceMax", "ReduceSum", "ReduceMean"}
COMPUTE = {"MatMul", "Gemm", "Conv"}


# ===================================================================== C3.2
def score_c32(graph):
    nodes = graph.nodes
    precs = {}
    decomps = {}
    for n in nodes:
        pp = strategy.select_precision(n, graph)
        precs[n.name] = pp.precision
        decomps[n.name] = strategy.decompose(n, graph, pp)

    supported = set(hardware.supported_precisions())

    # ---- D1 ----
    sens = [n for n in nodes if n.op_type in SENSITIVE]
    d1_sens = (sum(1 for n in sens if precs[n.name] == "fp32") / len(sens)) * 1.5 if sens else 1.5
    prec_kinds = set(precs.values()) & {"fp32", "fp16", "fp8", "fp4"}
    d1_div = (len(prec_kinds) / 4.0) * 1.0
    comp = [n for n in nodes if n.op_type in COMPUTE]
    d1_comp = (sum(1 for n in comp if precs[n.name] in supported) / len(comp)) * 0.5 if comp else 0.5
    D1 = round(d1_sens + d1_div + d1_comp, 3)

    # ---- D2 ----
    seq_cov = sum(1 for n in nodes if decomps[n.name]) / len(nodes)
    def prefixes(name):
        return [k.kernel for k in decomps[name]]
    key_weight_total, key_weight_hit = 0.0, 0.0
    for n in nodes:
        ks = prefixes(n.name)
        if n.op_type in ("MatMul", "Gemm"):
            key_weight_total += 0.5
            if any(k.startswith("matmul_") for k in ks):
                key_weight_hit += 0.5
        elif n.op_type == "Softmax":
            key_weight_total += 0.5
            need = ["reduce_max", "exp", "reduce_sum", "div"]
            if all(any(k.startswith(p) for k in ks) for p in need):
                key_weight_hit += 0.5
        elif n.op_type in ("LayerNormalization", "LayerNorm"):
            key_weight_total += 0.5
            need = ["reduce_mean", "sub", "mul", "sqrt"]
            if all(any(k.startswith(p) for k in ks) for p in need):
                key_weight_hit += 0.5
        elif n.op_type == "Conv":
            key_weight_total += 0.5
            if any(k.startswith("winograd_forward_") or k.startswith("im2col_") for k in ks):
                key_weight_hit += 0.5
    key_seq = (key_weight_hit / key_weight_total) if key_weight_total else 1.0
    D2 = round(min(seq_cov * 1.0 + key_seq * 2.0, 3.0), 3)

    # ---- D3 ----
    def has_inter(name, node):
        outs = set(node.outputs)
        for k in decomps[name]:
            if any(o not in outs for o in k.outputs):
                return True
        return False
    key_ops = [n for n in nodes if n.op_type in ("Softmax", "LayerNormalization", "LayerNorm", "Conv")]
    key_ratio = (sum(1 for n in key_ops if has_inter(n.name, n)) / len(key_ops)) if key_ops else 1.0
    total_ratio = sum(1 for n in nodes if has_inter(n.name, n)) / len(nodes)
    D3 = round(min(key_ratio * 2.0 + total_ratio * 1.0, 3.0), 3)

    # ---- D4 ----
    nodes_with_tuning = 0
    passed = 0
    max_block = hardware.max_threads_per_block
    smem_budget = hardware.smem_bytes
    for n in nodes:
        ks = decomps[n.name]
        if not ks:
            continue
        tp = strategy.tune_kernel(ks[0], precs[n.name], problem_size=1 << 14)
        nodes_with_tuning += 1
        passed += int(0 < tp.block_x <= max_block)
        passed += int(tp.grid_x > 0)
        passed += int(tp.smem_bytes == -1 or tp.smem_bytes <= smem_budget)
    tuning_cov = (nodes_with_tuning / len(nodes)) * 1.5
    tuning_val = (passed / (3 * nodes_with_tuning)) * 1.5 if nodes_with_tuning else 0.0
    D4 = round(tuning_cov + tuning_val, 3)

    # ---- D5 ----
    prec_variety = 1.0 if len(prec_kinds) >= 3 else (0.5 if len(prec_kinds) >= 2 else 0.0)
    all_kernels = [k.kernel for n in nodes for k in decomps[n.name]]
    gemm = set(k for k in all_kernels if k.startswith("matmul_"))
    gemm_div = 0.0
    if "matmul_f32" in gemm and "matmul_f16" in gemm:
        gemm_div = 0.5
        gemm_div += 0.25 if "matmul_f8" in gemm else 0.0
        gemm_div += 0.25 if "matmul_f4" in gemm else 0.0
    has_im2col = any(k.startswith("im2col_") for k in all_kernels)
    has_wino = any(k.startswith("winograd_forward_") for k in all_kernels)
    has_conv = any(n.op_type == "Conv" for n in nodes)
    if has_conv:
        conv_strategy = (0.5 if has_im2col else 0.0) + (0.5 if has_wino else 0.0)
    else:
        conv_strategy = 1.0  # no conv in this model -> not penalised
    D5 = round(min(prec_variety + gemm_div + conv_strategy, 3.0), 3)

    return {
        "D1": D1, "D2": D2, "D3": D3, "D4": D4, "D5": D5,
        "total": round(D1 + D2 + D3 + D4 + D5, 3),
        "precisions_seen": sorted(prec_kinds),
        "gemm_kernels": sorted(gemm),
        "conv_im2col": has_im2col, "conv_winograd": has_wino,
    }


# ===================================================================== C3.3
def score_c33(graph, model_key):
    pipe = GraphPassPipeline(enable_fusion=True)
    pipe.run(graph)
    stats = pipe.pass_results["Fusion"]["stats"]
    opt = pipe.optimized_graph

    # ---- F1 ----
    # Only the five canonical patterns in spec.md §F1 count toward the score.
    # Non-canonical fusion names (e.g. an activation fold reported as
    # 'FusedConvRelu') still earn F2/F3 through launch/buffer reduction but do
    # NOT inflate F1 — they are reported for transparency only.
    CANONICAL_PATTERNS = {
        "FusedMatMulBias", "FusedConv2dBatchNorm", "FusedEWChain",
        "FusedSoftmaxDropout", "FusedResidualNorm",
    }
    patterns_hit = set(stats["patterns_hit"])
    F1 = float(len(patterns_hit & CANONICAL_PATTERNS))  # 1 pt each, max 5

    # ---- F2 ----
    rl, ol = stats["raw_launches"], stats["opt_launches"]
    F2 = round(min((rl - ol) / rl * 5.0, 3.0), 3) if rl else 0.0

    # ---- F3 ----
    rb, ob = stats["raw_buffers"], stats["opt_buffers"]
    F3 = round(min((rb - ob) / rb * 5.0, 3.0), 3) if rb else 0.0

    # ---- F4 ----
    f4 = 0
    f4 += int(len(opt.output_names()) == len(graph.output_names()))
    f4 += int(len(opt.input_names()) == len(graph.input_names()))
    try:
        f4 += int(bool(opt.validate()))
    except Exception:
        pass
    f4 += int(len(opt.nodes) <= len(graph.nodes))
    # numeric alignment (original vs optimized) via MockRuntime
    align_ok, max_diff = _numeric_align(graph, opt, model_key)
    if not align_ok:
        f4 = 0
    F4 = float(f4)

    return {
        "F1": F1, "F2": F2, "F3": F3, "F4": F4,
        "total": round(F1 + F2 + F3 + F4, 3),
        "patterns_hit": sorted(patterns_hit),
        "launches": [rl, ol], "buffers": [rb, ob],
        "numeric_align": align_ok, "max_abs_diff": max_diff,
    }


def _numeric_align(graph, opt, model_key, feed_dict=None):
    """Numerical alignment check: original vs optimized graph.

    Returns ``(align_ok, max_diff)``:

    * ``(True, max_diff)`` — all checks pass and ``max_abs_diff <= 1e-3``.
    * ``(False, max_diff)`` — max_abs_diff > 1e-3 but computable (finite).
    * ``(False, None)``     — non-computable (exception, key/shape/dtype
      mismatch, NaN/Inf in output or diff, or empty output set).

    If *feed_dict* is provided it is used directly; otherwise a random feed is
    generated from *model_key*.
    """
    if feed_dict is not None:
        feed = feed_dict
    else:
        name, dtype, shape = INPUT_SHAPES[model_key]
        rng = np.random.default_rng(0)
        if dtype == np.int64:
            x = rng.integers(0, 13, size=shape, dtype=np.int64)
        else:
            x = rng.standard_normal(shape).astype(dtype)
        feed = {name: x}
    try:
        o1 = MockRuntime(graph).run(feed)
        o2 = MockRuntime(opt).run(feed)
    except Exception as exc:
        print(f"  [align] runtime error ({exc}); numeric check failed", file=sys.stderr)
        return False, None  # fail-closed
    # Empty output set → fail
    if not o1 or not o2:
        return False, None
    # Output key sets must match exactly
    if set(o1.keys()) != set(o2.keys()):
        return False, None
    max_diff = 0.0
    for k in o1:
        a, b = o1[k], o2[k]
        # Shape must match
        if a.shape != b.shape:
            return False, None
        # Dtype must match
        if a.dtype != b.dtype:
            return False, None
        # Both outputs must be fully finite
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return False, None
        diff = a - b
        # Diff must be fully finite
        if not np.all(np.isfinite(diff)):
            return False, None
        max_diff = max(max_diff, float(np.max(np.abs(diff))))
    # Threshold check: only ≤ 1e-3 is a pass
    return (max_diff <= 1e-3), max_diff


# ===================================================================== main
def _write_report(output_dir: str, all_scores: dict) -> None:
    """Write BENCHMARK_REPORT.md — the human-readable overview spec.md requires."""
    lines = ["# C3.2 / C3.3 Benchmark Report\n"]
    lines.append("评审可读总览。每个算子的分解 + tuning + 中间张量明细见 `bench_<model>.json`；"
                 "合并最终分见 `scores.json`。\n")

    grader_models = all_scores.get("_summary", {}).get("grader_models", [])
    for key, sc in all_scores.items():
        if key.startswith("_") or not isinstance(sc, dict):
            continue
        c32, c33 = sc["c32"], sc["c33"]
        tag = " *(official grader)*" if key in grader_models else ""
        lines.append(f"## {key}{tag}\n")
        lines.append(f"**C3.2 = {c32['total']}/15**  "
                     f"(D1={c32['D1']} D2={c32['D2']} D3={c32['D3']} D4={c32['D4']} D5={c32['D5']})\n")
        lines.append(f"- precisions seen: {c32['precisions_seen']}")
        lines.append(f"- GEMM kernels: {c32['gemm_kernels']}")
        lines.append(f"- Conv strategy: im2col={c32['conv_im2col']} winograd={c32['conv_winograd']}\n")
        lines.append(f"**C3.3 = {c33['total']}/15**  "
                     f"(F1={c33['F1']} F2={c33['F2']} F3={c33['F3']} F4={c33['F4']})\n")
        rl, ol = c33["launches"]
        rb, ob = c33["buffers"]
        lines.append(f"- patterns hit: {c33['patterns_hit']}")
        lines.append(f"- launches: {rl} -> {ol}")
        lines.append(f"- buffers: {rb} -> {ob}")
        lines.append(f"- numeric alignment: {c33['numeric_align']} (max_abs_diff={c33['max_abs_diff']})\n")

    summ = all_scores.get("_summary")
    if summ:
        total = summ.get("c32_avg", 0) + summ.get("c33_avg", 0)
        lines.append("## Summary (official grader models)\n")
        lines.append(f"- models: {summ.get('grader_models')}")
        lines.append(f"- C3.2 avg: {summ.get('c32_avg')}/15")
        lines.append(f"- C3.3 avg: {summ.get('c33_avg')}/15")
        lines.append(f"- **C3.2 + C3.3 = {round(total, 2)}/30**\n")

    with open(os.path.join(output_dir, "BENCHMARK_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mnist_mlp", "cifar_resnet18", "transformer"])
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    all_scores = {}
    for key in args.models:
        path = os.path.normpath(os.path.join(_MODELS_DIR, MODEL_FILES[key]))
        graph = import_onnx_graph(path)
        c32 = score_c32(graph)
        c33 = score_c33(graph, key)
        all_scores[key] = {"c32": c32, "c33": c33}
        with open(os.path.join(args.output_dir, f"bench_{key}.json"), "w", encoding="utf-8") as f:
            json.dump(all_scores[key], f, indent=2)
        print(f"\n=== {key} ({os.path.basename(path)}) ===")
        print(f"  C3.2  D1={c32['D1']} D2={c32['D2']} D3={c32['D3']} D4={c32['D4']} D5={c32['D5']}  "
              f"-> {c32['total']}/15")
        print(f"        precisions={c32['precisions_seen']} gemm={c32['gemm_kernels']} "
              f"im2col={c32['conv_im2col']} winograd={c32['conv_winograd']}")
        print(f"  C3.3  F1={c33['F1']} F2={c33['F2']} F3={c33['F3']} F4={c33['F4']}  "
              f"-> {c33['total']}/15")
        print(f"        patterns={c33['patterns_hit']} launches={c33['launches']} "
              f"buffers={c33['buffers']} max_abs_diff={c33['max_abs_diff']}")

    # official grader uses mlp + resnet; report that combined view
    grader_models = [m for m in ("mnist_mlp", "cifar_resnet18") if m in all_scores]
    if grader_models:
        c32_avg = np.mean([all_scores[m]["c32"]["total"] for m in grader_models])
        c33_avg = np.mean([all_scores[m]["c33"]["total"] for m in grader_models])
        print(f"\n=== official-grader models {grader_models}: "
              f"C3.2~{c32_avg:.2f}/15  C3.3~{c33_avg:.2f}/15 ===")
        all_scores["_summary"] = {
            "grader_models": grader_models,
            "c32_avg": round(float(c32_avg), 3),
            "c33_avg": round(float(c33_avg), 3),
        }
    with open(os.path.join(args.output_dir, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)
    _write_report(args.output_dir, all_scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
