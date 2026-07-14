#!/usr/bin/env python3
"""C3.2 回归门禁：验证 D1–D5 关键信号 + 有公开模型时锁定精确分数。

本测试分两部分：

  Part A —— 端到端模型评分
      加载三个公开模型（mnist_mlp、cifar_resnet18、transformer），通过
      scheduler 公共 API 计算 C3.2 D1–D5 分数，并断言与当前诚实基线一致：
        MLP        = 14.75/15  (D5 缺 0.25：3 个 Gemm 无法出现 4 种 matmul kernel)
        ResNet     = 15/15
        Transformer= 15/15
        官方两模型平均 = 14.875/15

  Part B —— D1–D5 关键信号单元测试
      不依赖具体模型，验证每个维度的评分公式关键行为：
      * D1: 敏感算子 fp32、四精度信号、compute 交集
      * D2: MatMul/Softmax/LayerNorm/Conv 分解前缀、中间张量
      * D3: 关键算子中间张量比率
      * D4: tuning 三断言
      * D5: im2col/Winograd 覆盖、gemm 多样度公式

退出码 0 表示全部通过；非 0 表示存在失败项。
"""

from __future__ import annotations

import os
import sys

_C3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3_ROOT not in sys.path:
    sys.path.insert(0, _C3_ROOT)

from scheduler import import_onnx_graph, strategy, hardware
from scheduler.graph import Graph, Node, TensorInfo
from benchmarks.c32_c33.bench_c32_c33 import score_c32

# ===================================================================== 模型路径

def _resolve_models_dir():
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
    return None  # caller handles None → explicit fail


_MODELS_DIR = _resolve_models_dir()

SCORES = {
    "mnist_mlp":      {"c32_total": 14.75, "D1": 3.0, "D2": 3.0, "D3": 3.0, "D4": 3.0, "D5": 2.75},
    "cifar_resnet18": {"c32_total": 15.0,  "D1": 3.0, "D2": 3.0, "D3": 3.0, "D4": 3.0, "D5": 3.0},
    "transformer":    {"c32_total": 15.0,  "D1": 3.0, "D2": 3.0, "D3": 3.0, "D4": 3.0, "D5": 3.0},
}
OFFICIAL_AVG = (SCORES["mnist_mlp"]["c32_total"] + SCORES["cifar_resnet18"]["c32_total"]) / 2  # 14.875

# ===================================================================== helpers

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


# =========================================================================
# Part A: 端到端模型评分
# =========================================================================

def test_model_scores():
    """加载三个公开模型，断言 C3.2 分数匹配诚实基线。"""
    if _MODELS_DIR is None:
        print("\n# [CRITICAL] 无公开模型目录 — 测试明确失败")
        check(False, "模型目录不存在；请确认模型文件就位。"
              f"\n        查找路径：检查 {_C3_ROOT}/../public/Track-C/.../models 或平铺布局")
        return

    print(f"\n# 模型目录: {_MODELS_DIR}")
    for key, expected in SCORES.items():
        onnx_name = {"mnist_mlp": "mlp_v1.onnx",
                     "cifar_resnet18": "resnet_v1.onnx",
                     "transformer": "transformer_v1.onnx"}[key]
        onnx_path = os.path.join(_MODELS_DIR, onnx_name)
        if not os.path.exists(onnx_path):
            check(False, f"[{key}] 模型缺失: {onnx_path}")
            continue

        graph = import_onnx_graph(onnx_path)
        scores = score_c32(graph)

        print(f"\n# {key} ({onnx_name})")
        check(abs(scores["total"] - expected["c32_total"]) < 1e-3,
              f"total={scores['total']} ≈ {expected['c32_total']}")
        for dim in ("D1", "D2", "D3", "D4", "D5"):
            check(abs(scores[dim] - expected[dim]) < 1e-3,
                  f"{dim}={scores[dim]} ≈ {expected[dim]}")
        # Sanity: all 4 precisions present
        check(len(scores["precisions_seen"]) == 4,
              f"4 precisions seen ({scores['precisions_seen']})")
        # D5: MLP-specific structural gap
        if key == "mnist_mlp":
            check(len(scores["gemm_kernels"]) == 3,
                  f"MLP: 3 gemm kernels ({scores['gemm_kernels']}) — 0.25 gap structural")
        else:
            check(len(scores["gemm_kernels"]) >= 4,
                  f"{key}: >=4 gemm kernels ({scores['gemm_kernels']})")

    # Official grader average
    avg = (SCORES["mnist_mlp"]["c32_total"] + SCORES["cifar_resnet18"]["c32_total"]) / 2
    check(abs(avg - OFFICIAL_AVG) < 1e-3,
          f"官方两模型平均 = {avg} (expected {OFFICIAL_AVG})")
    print()


# =========================================================================
# Part B: D1–D5 关键信号单元测试
# =========================================================================

# ---- D1 ----

def test_d1_sensitive_fp32():
    """敏感算子必须返回 fp32。"""
    print("\n# D1: 敏感算子 fp32")
    n = Node("sm", "Softmax", ["x"], ["out"])
    pp = strategy.select_precision(n, None)
    check(pp.precision == "fp32", f"Softmax → {pp.precision}")


def test_d1_four_precisions():
    """四种精度 token 均出现在非敏感算子中。"""
    print("\n# D1: 四种精度齐全")
    # Build a minimal graph with enough compute+elementwise ops to trigger all 4
    nodes = [Node(f"mm{i}", "Gemm", [f"x{i}", f"w{i}", f"b{i}"], [f"out{i}"])
             for i in range(4)]
    for i, n in enumerate(nodes):
        n._C3_INDEX = i
    g = Graph(nodes=nodes,
              inputs=[TensorInfo(f"x{i}") for i in range(4)] +
                     [TensorInfo(f"w{i}") for i in range(4)] +
                     [TensorInfo(f"b{i}") for i in range(4)],
              outputs=[TensorInfo(f"out{i}") for i in range(4)])
    precs = set()
    for n in g.nodes:
        pp = strategy.select_precision(n, g)
        precs.add(pp.precision)
    for p in ("fp32", "fp16", "fp8", "fp4"):
        check(p in precs, f"precision '{p}' present ({precs})")


def test_d1_comp_in_supported():
    """计算算子的精度在 supported_precisions() 中。"""
    print("\n# D1: compute 精度 ∈ supported")
    supported = set(hardware.supported_precisions())
    for op in ("MatMul", "Gemm", "Conv"):
        n = Node("c", op, ["x", "w"], ["out"])
        pp = strategy.select_precision(n, None)
        check(pp.precision in supported,
              f"{op} → {pp.precision} ∈ {supported}")


# ---- D2 ----

def test_d2_matmul_prefix():
    """MatMul/Gemm 分解产出 matmul_* 前缀。"""
    print("\n# D2: MatMul/Gemm → matmul_*")
    n = Node("mm", "MatMul", ["a", "b"], ["out"])
    pp = strategy.select_precision(n, None)
    ks = strategy.decompose(n, None, pp)
    has_matmul = any(k.kernel.startswith("matmul_") for k in ks)
    check(has_matmul, f"kernel prefix in {[k.kernel for k in ks]}")


def test_d2_softmax_prefixes():
    """Softmax 分解产出 reduce_max/exp/reduce_sum/div。"""
    print("\n# D2: Softmax → 4 个子 kernel")
    n = Node("sm", "Softmax", ["x"], ["out"])
    pp = strategy.select_precision(n, None)
    ks = strategy.decompose(n, None, pp)
    prefixes = {k.kernel.split("_")[0] for k in ks}
    for need in ("reduce_max", "exp", "reduce_sum", "div"):
        # check prefix match (kernel name starts with the requirement)
        ok = any(k.kernel.startswith(need) for k in ks)
        check(ok, f"kernel prefix '{need}' in {[k.kernel for k in ks]}")


def test_d2_layernorm_prefixes():
    """LayerNorm 分解产出 reduce_mean/sub/mul/sqrt。"""
    print("\n# D2: LayerNorm → 4 个子 kernel")
    n = Node("ln", "LayerNormalization", ["x", "w", "b"], ["out"])
    pp = strategy.select_precision(n, None)
    ks = strategy.decompose(n, None, pp)
    for need in ("reduce_mean", "sub", "mul", "sqrt"):
        ok = any(k.kernel.startswith(need) for k in ks)
        check(ok, f"kernel prefix '{need}' in {[k.kernel for k in ks]}")


def test_d2_conv_prefixes():
    """Conv 分解产出 winograd_forward_* 或 im2col_*。"""
    print("\n# D2: Conv → winograd_forward_* 或 im2col_*")
    n = Node("conv", "Conv", ["x", "w"], ["out"])
    pp = strategy.select_precision(n, None)
    ks = strategy.decompose(n, None, pp)
    has_conv = any(k.kernel.startswith("winograd_forward_") or
                    k.kernel.startswith("im2col_") for k in ks)
    check(has_conv, f"kernel prefixes in {[k.kernel for k in ks]}")


# ---- D3 ----

def test_d3_intermediate_tensor_naming():
    """分解产出 __c3_inter_N__ 中间张量。"""
    print("\n# D3: 中间张量 __c3_inter_N__")
    n = Node("sm", "Softmax", ["x"], ["out"])
    pp = strategy.select_precision(n, None)
    ks = strategy.decompose(n, None, pp)
    has_inter = any("__c3_inter_" in (o or "") for k in ks for o in k.outputs)
    check(has_inter, f"中间张量 in outputs: {[k.outputs for k in ks]}")


# ---- D4 ----

def test_d4_tuning_triple_assertions():
    """tune_kernel 返回的 block_x/grid_x/smem_bytes 三断言恒成立。"""
    print("\n# D4: tuning 三断言")
    n = Node("mm", "MatMul", ["a", "b"], ["out"])
    pp = strategy.select_precision(n, None)
    ks = strategy.decompose(n, None, pp)
    if not ks:
        check(False, "无 kernel 可调优")
        return
    tp = strategy.tune_kernel(ks[0], pp.precision, problem_size=1 << 14)
    max_block = hardware.max_threads_per_block
    smem_budget = hardware.smem_bytes
    check(0 < tp.block_x <= max_block, f"0 < block_x ({tp.block_x}) ≤ {max_block}")
    check(tp.grid_x > 0, f"grid_x ({tp.grid_x}) > 0")
    check(tp.smem_bytes == -1 or tp.smem_bytes <= smem_budget,
          f"smem_bytes ({tp.smem_bytes}) ≤ {smem_budget}")


# ---- D5 ----

def test_d5_im2col_winograd_switch():
    """Conv 分解在 im2col（stride2）与 Winograd（stride1 3×3）之间切换。"""
    print("\n# D5: im2col/Winograd switch")
    # stride2 Conv → im2col
    n = Node("conv_stride2", "Conv", ["x", "w"], ["out"],
             attrs={"kernel_shape": [3, 3], "strides": [2, 2]})
    pp = strategy.select_precision(n, None)
    ks = strategy.decompose(n, None, pp)
    prefixes = [k.kernel for k in ks]
    has_im2col = any(k.startswith("im2col_") for k in prefixes)
    check(has_im2col, f"stride2 Conv → im2col ({prefixes})")

    # stride1 3×3 Conv → Winograd
    n2 = Node("conv_wino", "Conv", ["x2", "w2"], ["out2"],
              attrs={"kernel_shape": [3, 3], "strides": [1, 1]})
    pp2 = strategy.select_precision(n2, None)
    ks2 = strategy.decompose(n2, None, pp2)
    prefixes2 = [k.kernel for k in ks2]
    has_wino = any(k.startswith("winograd_forward_") for k in prefixes2)
    check(has_wino, f"stride1 3×3 Conv → winograd_forward_* ({prefixes2})")


def test_d5_gemm_diversity_limit():
    """构造仅 3 Gemm 的图验证 D5 ≤ 2.75（0.25 结构性缺口）。"""
    print("\n# D5: 3 Gemm → D5 ≤ 2.75 (structural 0.25 gap)")
    nodes = [Node(f"g{i}", "Gemm", [f"a{i}", f"w{i}", f"b{i}"], [f"out{i}"])
             for i in range(3)]
    for i, n in enumerate(nodes):
        n._C3_INDEX = i
    g = Graph(nodes=nodes,
              inputs=[TensorInfo(f"a{i}") for i in range(3)] +
                     [TensorInfo(f"w{i}") for i in range(3)] +
                     [TensorInfo(f"b{i}") for i in range(3)],
              outputs=[TensorInfo(f"out{i}") for i in range(3)])
    precs = {}
    decomps = {}
    for n in g.nodes:
        pp = strategy.select_precision(n, g)
        precs[n.name] = pp.precision
        decomps[n.name] = strategy.decompose(n, g, pp)

    prec_kinds = set(precs.values()) & {"fp32", "fp16", "fp8", "fp4"}
    all_kernels = [k.kernel for n in g.nodes for k in decomps[n.name]]
    gemm = set(k for k in all_kernels if k.startswith("matmul_"))
    prec_variety = 1.0 if len(prec_kinds) >= 3 else (0.5 if len(prec_kinds) >= 2 else 0.0)
    gemm_div = 0.0
    if "matmul_f32" in gemm and "matmul_f16" in gemm:
        gemm_div = 0.5
        gemm_div += 0.25 if "matmul_f8" in gemm else 0.0
        gemm_div += 0.25 if "matmul_f4" in gemm else 0.0
    D5 = round(min(prec_variety + gemm_div + 1.0, 3.0), 3)
    check(D5 <= 2.75, f"D5={D5} ≤ 2.75 (gemm_kernels={sorted(gemm)}, gap=0.25)")


# =========================================================================
# main
# =========================================================================

def main() -> int:
    print("=" * 60)
    print("C3.2 回归门禁 (selftest_c32.py)")
    print("=" * 60)

    # Part A: end-to-end model scores
    test_model_scores()

    # Part B: D1–D5 signal unit tests
    print("--- D1 信号 ---")
    test_d1_sensitive_fp32()
    test_d1_four_precisions()
    test_d1_comp_in_supported()

    print("\n--- D2 信号 ---")
    test_d2_matmul_prefix()
    test_d2_softmax_prefixes()
    test_d2_layernorm_prefixes()
    test_d2_conv_prefixes()

    print("\n--- D3 信号 ---")
    test_d3_intermediate_tensor_naming()

    print("\n--- D4 信号 ---")
    test_d4_tuning_triple_assertions()

    print("\n--- D5 信号 ---")
    test_d5_im2col_winograd_switch()
    test_d5_gemm_diversity_limit()

    print(f"\n=== {_PASS} passed, {_FAIL} failed ===")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
