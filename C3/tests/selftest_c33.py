#!/usr/bin/env python3
"""C3.3 独立评委测试 (完全基于 spec.md / scoring.md，不复用选手自评脚本).

本测试分两部分:

  Part A —— 端到端维度打分 (F1–F4)
      仅评 spec 第 86/433 行规定的官方评测模型 mnist_mlp + cifar_resnet18.
      通过公共 API GraphPassPipeline(enable_fusion=True) 抓信号, 并按 spec 对
      每个 pattern 的「触发条件」逐条核验 fusion_log 背后的真实算子构成 ——
      不盲信 pattern 名 (spec §F1 的 FusedEWChain 须为 2–5 个相邻 elementwise;
      非 canonical 名如 FusedConvRelu 不计入 F1).

  Part B —— 单元测试: 每个 pattern matcher 在 spec 触发条件下的正确性
      用手工构造的微型图直接验证 5 个 canonical pattern + compute->act 折叠,
      不依赖公开模型是否恰好含对应算子. 这样能覆盖 spec §F1 全部 5 个 pattern
      的触发语义, 即使官方评测模型里不出现.

评分公式逐字取自 spec.md §C3.3 / scoring.md §C3.3:
  F1 = 命中的 canonical pattern 数 (每个 1 分, 满分 5)
  F2 = min((raw_launches − opt_launches) / raw_launches × 5.0, 3.0)
  F3 = min((raw_buffers − opt_buffers) / raw_buffers × 5.0, 3.0)
  F4 = 4 项结构检查 (各 1 分) + 数值对齐硬指标 (任一路径 max_abs_diff > 1e-3 则 F4 全扣)

退出码 0 表示全部通过; 非 0 表示存在失败项.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_C3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3_ROOT not in sys.path:
    sys.path.insert(0, _C3_ROOT)

from scheduler.graph import Graph, Node, TensorInfo
from scheduler.graph_passes.pipeline import GraphPassPipeline
from scheduler.graph_passes.fusion import FusionPass
from runtime.mock_runtime import MockRuntime

# 公开模型根目录: 优先 repo 布局 (public/...), 回退服务器平铺布局 (C3_ROOT/*.onnx)
def _resolve_models_dir():
    candidates = [
        os.path.normpath(os.path.join(
            _C3_ROOT, "..", "public", "Agentic4SystemSummerSchoolContest", "Track-C",
            "C3-scheduler", "testcases", "release_to_competitors", "models")),
        os.path.normpath(os.path.join(
            _C3_ROOT, "..", "public", "Track-C",
            "C3-scheduler", "testcases", "release_to_competitors", "models")),
        _C3_ROOT,
    ]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith(".onnx") for f in os.listdir(d)):
            return d
    return candidates[0]


_MODELS_DIR = _resolve_models_dir()
_MODELS = {
    "mnist_mlp": ("mlp_v1.onnx", "input", np.float32, (2, 1, 28, 28)),
    "cifar_resnet18": ("resnet_v1.onnx", "input", np.float32, (2, 3, 32, 32)),
}

# spec §C3.2 算子清单里的 elementwise 集合 (用于核验 FusedEWChain 触发条件)
_ELEMENTWISE = {"Add", "Mul", "Div", "Sub", "Relu", "Erf", "Sqrt"}
# spec §F1 的五个 canonical pattern 名 (scoring.md 第 85-92 行)
CANONICAL_PATTERNS = {
    "FusedMatMulBias",
    "FusedConv2dBatchNorm",
    "FusedEWChain",
    "FusedSoftmaxDropout",
    "FusedResidualNorm",
}

_PASS, _FAIL = 0, 0
_RESULTS = []  # 收集每项打分明细, 末尾汇总


def check(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"    PASS  {msg}")
    else:
        _FAIL += 1
        print(f"    FAIL  {msg}")
    return cond


# ===========================================================================
# spec 触发条件核验: 判定一条 fusion 记录背后真实算子构成是否满足该 pattern
# ===========================================================================
def _member_types(fused_node):
    return [m.op_type for m in fused_node.fused_ops]


def _verify_pattern(pattern, fused_node):
    """按 spec §F1 触发条件核验一条 fusion 记录是否真合规.

    返回 (合规bool, 说明). 注意: 非 canonical 名 (如 FusedConvRelu) 直接判为
    不计入 F1 (但仍可能在 F2/F3 计分).
    """
    if pattern not in CANONICAL_PATTERNS:
        return False, f"非 canonical pattern({pattern}), 不计入 F1"
    types = _member_types(fused_node)

    if pattern == "FusedEWChain":
        # spec: "2–5 个相邻 elementwise (Add → Mul → ReLU 等)"
        ok = 2 <= len(types) <= 5 and all(t in _ELEMENTWISE for t in types)
        return ok, f"EW链 {'->'.join(types)}" + ("" if ok else " 不满足2-5个elementwise")

    if pattern == "FusedMatMulBias":
        # spec: "MatMul → AddBias"
        ok = len(types) == 2 and types[0] == "MatMul" and types[1] == "Add"
        return ok, "->".join(types) + ("" if ok else " 不符 MatMul->AddBias")

    if pattern == "FusedConv2dBatchNorm":
        # spec: "Conv2d → BatchNorm"
        ok = (len(types) == 2 and types[0] == "Conv"
              and types[1] in ("BatchNormalization", "BatchNorm"))
        return ok, "->".join(types) + ("" if ok else " 不符 Conv->BN")

    if pattern == "FusedSoftmaxDropout":
        # spec: "Softmax → Dropout"
        ok = len(types) == 2 and types[0] == "Softmax" and types[1] == "Dropout"
        return ok, "->".join(types) + ("" if ok else " 不符 Softmax->Dropout")

    if pattern == "FusedResidualNorm":
        # spec: "skip-Add → LayerNorm"
        ok = (len(types) == 2 and types[0] == "Add"
              and types[1] in ("LayerNormalization", "LayerNorm"))
        return ok, "->".join(types) + ("" if ok else " 不符 skipAdd->LayerNorm")

    return False, "未知 pattern"


# ===========================================================================
# Part A: 端到端维度打分 (官方评测模型)
# ===========================================================================
def _run_fusion(graph):
    """通过公共 API 运行融合, 返回 (stats, optimized_graph)."""
    pipe = GraphPassPipeline(enable_fusion=True)
    pipe.run(graph)
    return pipe.pass_results["Fusion"]["stats"], pipe.optimized_graph


def score_f1(opt_graph, stats):
    """F1 (5 分): 5 个 canonical pattern 各 1 分, 逐条按 spec 触发条件核验.

    核验规则:
    * 真实融合 (fused_node 在 opt 图中且有 fused_ops): 按该节点的 fused_ops
      构成逐条核验 spec 触发条件.
    * 标注 (annotation): fused_node 指向一个更大的融合节点, 此时核验该节点的
      fused_ops 里是否包含所述 pattern 的子序列 (如 Conv→Add→Relu 里的
      Add→Relu 是合法 FusedEWChain). 这是合法的 pattern 识别.
    * 无法定位 fused_node 的标注 (如 BN 预折叠): 记录但不计分, 留给 code review.
    """
    fnode = {n.name: n for n in opt_graph.nodes}
    strict_hits = set()
    detail = []
    for rec in stats["fusion_log"]:
        fn = fnode.get(rec["fused_node"])
        if fn is None:
            # annotation 指向不存在的节点 (如 BN 预折叠的 Conv) -> 不计分, 不崩溃
            detail.append((rec["pattern"], False, f"标注: fused_node {rec['fused_node'][:30]} 不在优化图中 (预折叠识别)"))
            continue
        if not fn.fused_ops:
            # 标注节点本身无 fused_ops (如 Gemm(bias)) -> 不计分
            detail.append((rec["pattern"], False, f"标注: {rec.get('annotation','')} (节点无 fused_ops)"))
            continue
        ok, why = _verify_pattern(rec["pattern"], fn)
        # 对标注的 FusedEWChain: 核验 fused_ops 里是否有连续 elementwise 子链
        if not ok and rec["pattern"] == "FusedEWChain":
            types = [m.op_type for m in fn.fused_ops]
            # 找最长的连续 elementwise 子序列
            best = []
            cur = []
            for t in types:
                if t in _ELEMENTWISE:
                    cur.append(t)
                else:
                    if len(cur) > len(best): best = cur
                    cur = []
            if len(cur) > len(best): best = cur
            if 2 <= len(best) <= 5:
                ok = True
                why = f"标注: 嵌入 EW 子链 {'->'.join(best)} (在 {fn.op_type[:20]} 内)"
        detail.append((rec["pattern"], ok, why))
        if ok:
            strict_hits.add(rec["pattern"])
    for pat, ok, why in detail[:6]:
        print(f"        [{'✓' if ok else '·'}] {pat}: {why}")
    if len(detail) > 6:
        print(f"        ... 共 {len(detail)} 条融合记录")
    return float(len(strict_hits)), sorted(strict_hits), detail


def score_f2(stats):
    """F2 (3 分): min((raw−opt)/raw × 5, 3), 60% 缩减即满分."""
    rl, ol = stats["raw_launches"], stats["opt_launches"]
    if rl <= 0:
        return 0.0, (rl, ol)
    reduction = (rl - ol) / rl
    return round(min(reduction * 5.0, 3.0), 3), (rl, ol)


def score_f3(stats):
    """F3 (3 分): min((raw−opt)/raw × 5, 3), 60% 缩减即满分."""
    rb, ob = stats["raw_buffers"], stats["opt_buffers"]
    if rb <= 0:
        return 0.0, (rb, ob)
    reduction = (rb - ob) / rb
    return round(min(reduction * 5.0, 3.0), 3), (rb, ob)


def score_f4(graph, opt, model_key, feed_dict=None):
    """F4 (4 分): 4 项结构检查 + 数值对齐硬指标(超阈全扣).

    Args:
        graph: 原始图.
        opt: 优化图.
        model_key: ``_MODELS`` 键名, 用于生成随机输入 (shape/dtype/input name).
        feed_dict: 可选覆盖, 提供自定义输入 (用于 fail-closed 测试). 此时忽略
            从 model_key 生成的随机输入.
    """
    fn, name, dtype, shape = _MODELS[model_key]
    if feed_dict is not None:
        feed = feed_dict
    else:
        rng = np.random.default_rng(0)
        if dtype == np.int64:
            x = rng.integers(0, 13, size=shape, dtype=np.int64)
        else:
            x = rng.standard_normal(shape).astype(dtype)
        feed = {name: x}

    pts = 0
    pts += int(len(opt.output_names()) == len(graph.output_names()))  # graph.outputs 保留
    pts += int(len(opt.input_names()) == len(graph.input_names()))    # graph.inputs 保留
    try:
        pts += int(bool(opt.validate()))                              # validate() 通过
    except Exception:
        pass
    pts += int(len(opt.nodes) <= len(graph.nodes))                    # 节点数不增

    # 数值对齐: MockRuntime 跑原图 + 优化图, 任一 > 1e-3 则 F4 全扣 (spec 第 212 行)
    # 异常 / 输出 key 不完全一致也算 fail (fail-closed).
    try:
        o1 = MockRuntime(graph).run(feed)
        o2 = MockRuntime(opt).run(feed)
        max_diff = 0.0
        align_ok = True
        if set(o1.keys()) != set(o2.keys()):
            align_ok = False
        else:
            for k in o1:
                a, b = o1[k], o2[k]
                if a.shape != b.shape:
                    align_ok = False
                    break
                if a.dtype != b.dtype:
                    align_ok = False
                    break
                if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                    align_ok = False
                    break
                diff = a - b
                if not np.all(np.isfinite(diff)):
                    align_ok = False
                    break
                max_diff = max(max_diff, float(np.max(np.abs(diff))))
        if align_ok and max_diff > 1e-3:
            align_ok = False
    except Exception as exc:
        print(f"        [warn] 数值对齐运行异常 (fail-closed): {exc}")
        align_ok, max_diff = False, None
    if not align_ok:
        pts = 0
    return float(pts), align_ok, max_diff


def judge_model(key):
    fn, name, dtype, shape = _MODELS[key]
    path = os.path.join(_MODELS_DIR, fn)
    if not os.path.exists(path):
        print(f"  [SKIP] {key}: 模型缺失 {path}")
        return None
    from scheduler.graph import import_onnx_graph
    g = import_onnx_graph(path)
    raw_n = len(g.nodes)

    print(f"\n# {key} ({fn}) — raw 节点数 {raw_n}")
    stats, opt = _run_fusion(g)

    F1, hits, detail = score_f1(opt, stats)
    F2, (rl, ol) = score_f2(stats)
    F3, (rb, ob) = score_f3(stats)
    F4, align_ok, md = score_f4(g, opt, key)
    total = round(F1 + F2 + F3 + F4, 3)

    print(f"    F1={F1}/5  命中canonical={hits}")
    print(f"    F2={F2}/3  launch {rl}->{ol} (缩减 {(rl-ol)/max(rl,1)*100:.1f}%)")
    print(f"    F3={F3}/3  buffer {rb}->{ob} (缩减 {(rb-ob)/max(rb,1)*100:.1f}%)")
    print(f"    F4={F4}/4  数值对齐={'PASS' if align_ok else 'FAIL'} max_diff={md}")
    print(f"    >>> {key} C3.3 = {total}/15")
    _RESULTS.append((key, F1, F2, F3, F4, total))
    return total


# ===========================================================================
# Part B: 单元测试 —— 每个 pattern matcher 的 spec 触发条件正确性
# ===========================================================================
def _mk(name, op, ins, outs, attrs=None):
    return Node(name=name, op_type=op, inputs=list(ins), outputs=list(outs),
                attrs=attrs or {})


def _run_one_pass(graph):
    """直接用 FusionPass, 返回 (fusion_log, optimized_graph)."""
    res = FusionPass(enable_fusion=True).run(graph)
    return res["stats"]["fusion_log"], res["graph"]


def _log_patterns(log):
    return sorted({r["pattern"] for r in log})


def test_fusedewchain_real_chain():
    """spec: FusedEWChain = 2–5 个相邻 elementwise. 构造 Add->Mul->Relu 三元链."""
    print("\n# 单元测试: FusedEWChain (3 元真 EW 链 Add->Mul->Relu)")
    g = Graph(nodes=[
        _mk("a", "Add", ["x", "y"], ["t1"]),
        _mk("m", "Mul", ["t1", "s"], ["t2"]),
        _mk("r", "Relu", ["t2"], ["out"]),
    ], inputs=[TensorInfo("x"), TensorInfo("y"), TensorInfo("s")],
       outputs=[TensorInfo("out")])
    log, opt = _run_one_pass(g)
    check("FusedEWChain" in _log_patterns(log), "命中 FusedEWChain")
    check(len(opt.nodes) == 1, f"三节点折叠为 1 个融合节点 (实际 {len(opt.nodes)})")
    check(opt.validate(), "折叠后图 validate() 通过")


def test_fusedewchain_upper_bound():
    """spec: EW 链上限 5 个. 构造 5 元链应折叠; 6 元链应停在 5."""
    print("\n# 单元测试: FusedEWChain 上限 5 个")
    nodes = [_mk("n0", "Add", ["x0", "x1"], ["t0"])]
    for i in range(1, 6):
        nodes.append(_mk(f"n{i}", "Relu", [f"t{i-1}"], [f"t{i}"]))
    g = Graph(nodes=nodes,
              inputs=[TensorInfo("x0"), TensorInfo("x1")],
              outputs=[TensorInfo("t5")])
    log, opt = _run_one_pass(g)
    ew = [r for r in log if r["pattern"] == "FusedEWChain"]
    check(len(ew) >= 1, "6 元链至少产出 1 个 FusedEWChain")
    # 每条链成员数应 <= 5
    fnode = {n.name: n for n in opt.nodes}
    all_le5 = all(len(fnode[r["fused_node"]].fused_ops) <= 5 for r in ew)
    check(all_le5, "每条 FusedEWChain 成员数 ≤ 5 (spec 上限)")


def test_fusedewchain_single_not_match():
    """spec: 单个 elementwise 不构成链 (需 ≥2). 单个 Relu 不应报 FusedEWChain."""
    print("\n# 单元测试: 单个 elementwise 不触发 FusedEWChain")
    g = Graph(nodes=[_mk("r", "Relu", ["x"], ["out"])],
              inputs=[TensorInfo("x")], outputs=[TensorInfo("out")])
    log, opt = _run_one_pass(g)
    check("FusedEWChain" not in _log_patterns(log), "单个 Relu 不报 FusedEWChain")


def test_compute_activation_not_ewchain():
    """compute->act (Conv->Relu) 不是 spec 的 FusedEWChain, 不应冒充."""
    print("\n# 单元测试: Conv->Relu 不冒充 FusedEWChain")
    g = Graph(
        nodes=[_mk("c", "Conv", ["x", "w"], ["c_out"]),
               _mk("r", "Relu", ["c_out"], ["out"])],
        inputs=[TensorInfo("x"), TensorInfo("w", shape=[1, 1, 1, 1])],
        outputs=[TensorInfo("out")])
    log, opt = _run_one_pass(g)
    check("FusedEWChain" not in _log_patterns(log),
          "Conv->Relu 不挂 FusedEWChain 名 (compute 非 elementwise)")
    # 仍应被某种融合折叠 (F2/F3 收益), 节点数应减少
    check(len(opt.nodes) == 1, f"Conv->Relu 仍被折叠 (节点数 {len(opt.nodes)})")


def test_matmul_bias():
    """spec: FusedMatMulBias = MatMul -> AddBias (Add 的另一输入须为 initializer)."""
    print("\n# 单元测试: FusedMatMulBias (MatMul->Add, bias 为 initializer)")
    init = {"bias": np.zeros(4, dtype=np.float32)}
    g = Graph(
        nodes=[_mk("mm", "MatMul", ["x", "w"], ["mm_out"]),
               _mk("add", "Add", ["mm_out", "bias"], ["out"])],
        inputs=[TensorInfo("x"), TensorInfo("w")],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedMatMulBias" in _log_patterns(log), "命中 FusedMatMulBias")


def test_matmul_bias_rejects_residual_add():
    """Add 的两个输入都非 initializer (残差) 不应被当 bias 融合."""
    print("\n# 单元测试: 残差 Add (两输入均非 initializer) 不触发 FusedMatMulBias")
    g = Graph(
        nodes=[_mk("mm", "MatMul", ["x", "w"], ["mm_out"]),
               _mk("add", "Add", ["mm_out", "residual"], ["out"])],
        inputs=[TensorInfo("x"), TensorInfo("w"), TensorInfo("residual")],
        outputs=[TensorInfo("out")])
    log, opt = _run_one_pass(g)
    check("FusedMatMulBias" not in _log_patterns(log), "残差 Add 不误判为 bias 融合")


def test_residual_norm():
    """spec: FusedResidualNorm = skip-Add -> LayerNorm."""
    print("\n# 单元测试: FusedResidualNorm (skip-Add -> LayerNorm)")
    init = {"ln_weight": np.ones(4, np.float32), "ln_bias": np.zeros(4, np.float32)}
    g = Graph(
        nodes=[_mk("add", "Add", ["a", "b"], ["t"]),
               _mk("ln", "LayerNormalization", ["t", "ln_weight", "ln_bias"], ["out"])],
        inputs=[TensorInfo("a"), TensorInfo("b")],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedResidualNorm" in _log_patterns(log), "命中 FusedResidualNorm")


def test_softmax_dropout():
    """spec: FusedSoftmaxDropout = Softmax -> Dropout."""
    print("\n# 单元测试: FusedSoftmaxDropout (Softmax->Dropout)")
    g = Graph(
        nodes=[_mk("sm", "Softmax", ["x"], ["s"]),
               _mk("dr", "Dropout", ["s"], ["out"])],
        inputs=[TensorInfo("x")], outputs=[TensorInfo("out")])
    log, opt = _run_one_pass(g)
    check("FusedSoftmaxDropout" in _log_patterns(log), "命中 FusedSoftmaxDropout")


def test_conv_bn():
    """spec: FusedConv2dBatchNorm = Conv -> BatchNorm."""
    print("\n# 单元测试: FusedConv2dBatchNorm (Conv->BatchNorm)")
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 1, 1, 1)).astype(np.float32)
    init = {
        "w": w,
        "bn_w": np.ones(4, dtype=np.float32),
        "bn_b": np.zeros(4, dtype=np.float32),
        "bn_m": np.zeros(4, dtype=np.float32),
        "bn_v": np.ones(4, dtype=np.float32),
    }
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w"], ["c"]),
               _mk("bn", "BatchNormalization", ["c", "bn_w", "bn_b", "bn_m", "bn_v"], ["out"])],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" in _log_patterns(log), "命中 FusedConv2dBatchNorm")


def test_conv_bn_exact_fused_ops():
    """Conv->BN: exact fused_ops=[Conv, BN], outer fused node active, numerical align."""
    print("\n# 单元测试: FusedConv2dBatchNorm exact fused_ops + 数值对齐")
    rng = np.random.default_rng(42)
    w = rng.standard_normal((2, 1, 3, 3)).astype(np.float32)
    init = {
        "w": w,
        "bn_w": np.ones(2, dtype=np.float32) * 2.0,
        "bn_b": np.ones(2, dtype=np.float32) * 0.5,
        "bn_m": np.zeros(2, dtype=np.float32),
        "bn_v": np.ones(2, dtype=np.float32),
    }
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w"], ["c"]),
               _mk("bn", "BatchNormalization", ["c", "bn_w", "bn_b", "bn_m", "bn_v"], ["out"])],
        inputs=[TensorInfo("x", shape=[1, 1, 5, 5])],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" in _log_patterns(log), "命中 FusedConv2dBatchNorm")
    fn_map = {n.name: n for n in opt.nodes}
    cb_rec = [r for r in log if r["pattern"] == "FusedConv2dBatchNorm"]
    check(len(cb_rec) == 1, "恰好 1 条 FusedConv2dBatchNorm 记录")
    if cb_rec:
        fused = fn_map.get(cb_rec[0]["fused_node"])
        check(fused is not None, "fused_node 在 opt 图中")
        if fused and fused.fused_ops:
            types = [m.op_type for m in fused.fused_ops]
            check(types == ["Conv", "BatchNormalization"],
                  f"fused_ops == [Conv, BatchNormalization] (实际 {types})")
        # outer fused node on active path
        producers = opt.producer_map()
        reachable = set()
        stack = list(opt.output_names())
        while stack:
            t = stack.pop()
            src = producers.get(t)
            if src is not None and src.name not in reachable:
                reachable.add(src.name)
                for inp in src.inputs:
                    stack.append(inp)
        fused_name = cb_rec[0]["fused_node"]
        check(fused_name in reachable, f"fused_node '{fused_name}' 在 active path 上")
    # 数值对齐
    x = rng.standard_normal((1, 1, 5, 5)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_conv_bn_nonzero_bias_no_bn():
    """Conv with non-zero bias but no BN -> no FusedConv2dBatchNorm, no annotation."""
    print("\n# 单元测试: Conv 非零 bias 无 BN — 不命中")
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 1, 1, 1)).astype(np.float32)
    b = np.ones(4, dtype=np.float32)
    init = {"w": w, "b": b}
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w", "b"], ["out"])],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" not in _log_patterns(log), "无 BN → 不报 FusedConv2dBatchNorm")
    annots = [r for r in log if r.get("annotation")]
    check(len(annots) == 0, f"无 annotation (实际 {len(annots)})")


def test_conv_bn_multi_consumer():
    """Conv output has 2+ consumers -> no fusion."""
    print("\n# 单元测试: Conv 输出多消费者 — 拒绝融合")
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 1, 1, 1)).astype(np.float32)
    init = {"w": w, "bn_w": np.ones(4, dtype=np.float32), "bn_b": np.zeros(4, dtype=np.float32),
            "bn_m": np.zeros(4, dtype=np.float32), "bn_v": np.ones(4, dtype=np.float32)}
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w"], ["c"]),
               _mk("bn", "BatchNormalization", ["c", "bn_w", "bn_b", "bn_m", "bn_v"], ["bn_out"]),
               _mk("bypass", "Relu", ["c"], ["bypass_out"])],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("bn_out"), TensorInfo("bypass_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" not in _log_patterns(log), "多消费者 → 不报 FusedConv2dBatchNorm")


def test_conv_bn_graph_output():
    """Conv output is graph output -> no fusion."""
    print("\n# 单元测试: Conv 输出是 graph output — 拒绝融合")
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 1, 1, 1)).astype(np.float32)
    init = {"w": w, "bn_w": np.ones(4, dtype=np.float32), "bn_b": np.zeros(4, dtype=np.float32),
            "bn_m": np.zeros(4, dtype=np.float32), "bn_v": np.ones(4, dtype=np.float32)}
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w"], ["c"]),
               _mk("bn", "BatchNormalization", ["c", "bn_w", "bn_b", "bn_m", "bn_v"], ["out"])],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("c"), TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" not in _log_patterns(log), "Conv 输出是 graph output → 不报")


def test_conv_bn_dynamic_weight():
    """Conv weight is not an initializer -> no fusion."""
    print("\n# 单元测试: Conv weight 非 initializer — 拒绝融合")
    init = {"bn_w": np.ones(4, dtype=np.float32), "bn_b": np.zeros(4, dtype=np.float32),
            "bn_m": np.zeros(4, dtype=np.float32), "bn_v": np.ones(4, dtype=np.float32)}
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w"], ["c"]),
               _mk("bn", "BatchNormalization", ["c", "bn_w", "bn_b", "bn_m", "bn_v"], ["out"])],
        inputs=[TensorInfo("x"), TensorInfo("w")],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" not in _log_patterns(log), "动态 weight → 不报 FusedConv2dBatchNorm")


def test_conv_bn_bad_params():
    """BN params missing / not initializers -> no fusion."""
    print("\n# 单元测试: BN 参数缺失/非 initializer — 拒绝融合")
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 1, 1, 1)).astype(np.float32)
    init = {"w": w}
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w"], ["c"]),
               _mk("bn", "BatchNormalization", ["c", "bn_w", "bn_b", "bn_m", "bn_v"], ["out"])],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" not in _log_patterns(log), "BN 参数非 initializer → 不报")


def test_conv_bn_extra_output_consumed():
    """BN extra output has a consumer -> no fusion."""
    print("\n# 单元测试: BN extra output 有消费者 — 拒绝融合")
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 1, 1, 1)).astype(np.float32)
    init = {"w": w, "bn_w": np.ones(4, dtype=np.float32), "bn_b": np.zeros(4, dtype=np.float32),
            "bn_m": np.zeros(4, dtype=np.float32), "bn_v": np.ones(4, dtype=np.float32)}
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w"], ["c"]),
               _mk("bn", "BatchNormalization", ["c", "bn_w", "bn_b", "bn_m", "bn_v"],
                   ["bn_out", "bn_mean"]),
               _mk("extra", "Relu", ["bn_mean"], ["extra_out"])],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("bn_out"), TensorInfo("extra_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" not in _log_patterns(log), "BN extra output 被消费 → 不报")


def test_conv_bn_extra_output_graph_out():
    """BN extra output is a graph output -> no fusion."""
    print("\n# 单元测试: BN extra output 是 graph output — 拒绝融合")
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 1, 1, 1)).astype(np.float32)
    init = {"w": w, "bn_w": np.ones(4, dtype=np.float32), "bn_b": np.zeros(4, dtype=np.float32),
            "bn_m": np.zeros(4, dtype=np.float32), "bn_v": np.ones(4, dtype=np.float32)}
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w"], ["c"]),
               _mk("bn", "BatchNormalization", ["c", "bn_w", "bn_b", "bn_m", "bn_v"],
                   ["bn_out", "bn_mean"])],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("bn_out"), TensorInfo("bn_mean")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedConv2dBatchNorm" not in _log_patterns(log), "BN extra output 是 graph output → 不报")


def test_prefuse_conv_bn_prefolded_annotation():
    """A biased Conv is annotated as pre-folded BN without rewriting graph data."""
    print("\n# 单元测试: prefuse_conv_bn 预折叠 annotation 不改图")
    from scheduler.graph_passes.fusion import prefuse_conv_bn
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 1, 1, 1)).astype(np.float32)
    b = np.ones(4, dtype=np.float32)
    init = {"w": w.copy(), "b": b.copy()}
    init_before = {k: v.copy() for k, v in init.items()}
    g = Graph(
        nodes=[_mk("conv", "Conv", ["x", "w", "b"], ["out"])],
        inputs=[TensorInfo("x")],
        outputs=[TensorInfo("out")],
        initializers=init)
    result = prefuse_conv_bn(g)
    check(len(result) == 1, f"prefuse_conv_bn 返回 1 条 annotation (实际 {len(result)})")
    if result:
        check(result[0].get("pattern") == "FusedConv2dBatchNorm",
              f"annotation pattern 正确 (实际 {result[0].get('pattern')})")
        check(result[0].get("nodes") == ["conv"],
              f"annotation 仅引用预折叠 Conv (实际 {result[0].get('nodes')})")
    # Graph structure unchanged
    check(len(g.nodes) == 1, "graph 节点数不变")
    check(g.nodes[0].op_type == "Conv", "Conv 节点保留")
    # Initializers unchanged
    for k, v in init_before.items():
        check(k in g.initializers, f"initializer {k} 保留")
        check(np.allclose(g.initializers[k], v), f"initializer {k} 值不变")


def test_correctness_numeric_align():
    """F4 核心: 优化图与原图数值对齐 (max_abs_diff <= 1e-3). 用带权重的真实小图."""
    print("\n# 单元测试: F4 数值对齐 (Add->Relu 链, 带 initializer)")
    rng = np.random.default_rng(42)
    s = rng.standard_normal((3, 4)).astype(np.float32)
    init = {"scale": rng.standard_normal(4).astype(np.float32)}
    g = Graph(
        nodes=[_mk("a", "Add", ["x", "scale"], ["t"]),
               _mk("r", "Relu", ["t"], ["out"])],
        inputs=[TensorInfo("x", shape=[3, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    o1 = MockRuntime(g).run({"x": s})
    o2 = MockRuntime(opt).run({"x": s})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"折叠前后数值一致 max_abs_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): Gemm→MatMulBias canonicalization 微图测试
# ===========================================================================

def _gemm_replay_verify(opt, log):
    """验证 Gemm 的 FusedMatMulBias 规范: fused_node 在 opt 图, fused_ops=[MatMul,Add]."""
    fn = {n.name: n for n in opt.nodes}
    for rec in log:
        if rec["pattern"] == "FusedMatMulBias" and "annotation" not in rec:
            fn2 = fn.get(rec["fused_node"])
            if fn2 is None:
                return False, "fused_node 不在 opt 图"
            types = [m.op_type for m in fn2.fused_ops]
            if types != ["MatMul", "Add"]:
                return False, f"fused_ops={types} 应为 [MatMul, Add]"
            return True, "OK"
    return False, "未找到实时 FusedMatMulBias 记录"


def test_gemm_to_matmul_bias_default():
    """Gemm(A,W,C) 默认参数 → FusedMatMulBias replay=[MatMul,Add] + 数值对齐."""
    print("\n# 单元测试: Gemm→MatMulBias 默认参数")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"])],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    ok, why = _gemm_replay_verify(opt, log)
    check(ok, why)
    # 数值对齐 (same feed for both)
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")


def test_gemm_to_matmul_bias_transb():
    """Gemm transB=1, alpha=1 → B 转置吸收进新 initializer."""
    print("\n# 单元测试: Gemm→MatMulBias transB=1")
    rng = np.random.default_rng(42)
    W = rng.standard_normal((3, 4)).astype(np.float32)  # transB → (4,3)
    C = np.ones(3, dtype=np.float32)  # output = A@W^T → (2,4)@(4,3) = (2,3) ✓
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"],
                    attrs={"transB": 1, "alpha": 1.0, "beta": 1.0})],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    ok, why = _gemm_replay_verify(opt, log)
    check(ok, why)
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"transB 数值对齐 max_diff={md:.2e}")


def test_gemm_to_matmul_bias_beta_only():
    """Gemm alpha=1.0, beta=3.0 → beta 吸收进新 bias, 仍 canonical."""
    print("\n# 单元测试: Gemm→MatMulBias beta 吸收 (alpha=1.0)")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32) * 2.0
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"],
                    attrs={"alpha": 1.0, "beta": 3.0})],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    ok, why = _gemm_replay_verify(opt, log)
    check(ok, why)
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"beta 吸收数值对齐 max_diff={md:.2e}")


def test_gemm_alpha_non1_fallback():
    """Gemm alpha=2.0 → 不 canonicalize，保留 Gemm, 数值保持."""
    print("\n# 单元测试: Gemm alpha=2.0 fallback (不 canonicalize)")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"],
                    attrs={"alpha": 2.0, "beta": 1.0})],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    # alpha != 1 → no real fused node, no annotation
    real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                and "annotation" not in r]
    check(len(real_fmb) == 0, "alpha=2 → 不生成实时 FusedMatMulBias")
    annotation_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                      and "annotation" in r]
    check(len(annotation_fmb) == 0, "alpha=2 → 无 annotation (不冒充)")
    # Gemm 保留原样
    check(any(n.op_type == "Gemm" for n in opt.nodes), "Gemm 保留原样")
    # 无 fused 节点
    fused_types = [n.op_type for n in opt.nodes if n.fused_ops]
    check(len(fused_types) == 0, f"无 fused 节点 (实际 {fused_types})")
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"alpha=2 数值对齐 max_diff={md:.2e}")


def test_gemm_alpha_fp16_fallback():
    """FP16 alpha=0.1 → 不 canonicalize，保留 Gemm, 数值保持."""
    print("\n# 单元测试: Gemm alpha=0.1 (FP16 safe — fallback)")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"],
                    attrs={"alpha": 0.1, "beta": 1.0})],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    # alpha != 1 → no canonicalization
    real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                and "annotation" not in r]
    check(len(real_fmb) == 0, "alpha=0.1 → 不生成实时 FusedMatMulBias")
    # Gemm 保留
    check(any(n.op_type == "Gemm" for n in opt.nodes), "Gemm 保留原样")
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"alpha=0.1 数值对齐 max_diff={md:.2e}")


def test_gemm_to_matmul_bias_broadcast():
    """Gemm 1-D bias C 广播到 (M,N)."""
    print("\n# 单元测试: Gemm→MatMulBias bias broadcast")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"])],
        inputs=[TensorInfo("A", shape=[3, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    ok, why = _gemm_replay_verify(opt, log)
    check(ok, why)
    feed = {"A": rng.standard_normal((3, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"bias broadcast 数值对齐 max_diff={md:.2e}")


def test_gemm_shared_initializer_not_mutated():
    """两个 Gemm 共享 W → canonicalization 不原地修改."""
    print("\n# 单元测试: Gemm 共享 initializer 不被原地修改")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    w_before = init["W"].copy()
    g = Graph(
        nodes=[_mk("g1", "Gemm", ["A", "W", "C"], ["o1"]),
               _mk("g2", "Gemm", ["B", "W", "C"], ["o2"])],
        inputs=[TensorInfo("A", shape=[2, 4]), TensorInfo("B", shape=[3, 4])],
        outputs=[TensorInfo("o1"), TensorInfo("o2")], initializers=init)
    log, opt = _run_one_pass(g)
    check(np.allclose(init["W"], w_before), "共享 W 未被原地修改")
    n_fused = sum(1 for r in log if r["pattern"] == "FusedMatMulBias" and "annotation" not in r)
    check(n_fused >= 2, f"两个 Gemm 各自生成 FusedMatMulBias (实际 {n_fused})")


def test_gemm_no_bias_not_canonical():
    """Gemm 无 bias (2 inputs) 不报 FusedMatMulBias."""
    print("\n# 单元测试: Gemm 无 bias 不 claim canonical")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    init = {"W": W.copy()}
    g = Graph(
        nodes=[_mk("g", "Gemm", ["A", "W"], ["out"])],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                and "annotation" not in r]
    check(len(real_fmb) == 0, "无 bias Gemm 不生成实时 FusedMatMulBias 节点")
    # annotation 可以有 (认出是 MatMul semantics, 但不计分)
    annotation_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                      and "annotation" in r]
    # 无所谓; 关键是 real fused node 不存在


def test_gemm_transa_fallback():
    """Gemm transA=1 → 不冒充 strict FusedMatMulBias (fallback to annotation)."""
    print("\n# 单元测试: Gemm transA=1 fallback to annotation")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"],
                    attrs={"transA": 1})],
        inputs=[TensorInfo("A", shape=[4, 2])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                and "annotation" not in r]
    check(len(real_fmb) == 0, "transA=1 不生成实时 FusedMatMulBias (不冒充)")


def test_mlp_structure_fusion():
    """MLP 三段结构: Flatten→Gemm→Relu / Gemm→Relu / standalone Gemm."""
    print("\n# 单元测试: MLP 三段结构融合")
    rng = np.random.default_rng(0)
    # transB=1 → stored weight shape = (out_dim, in_dim) so W^T = (in_dim, out_dim)
    init = {
        "W1": rng.standard_normal((128, 784)).astype(np.float32),
        "B1": rng.standard_normal(128).astype(np.float32),
        "W2": rng.standard_normal((64, 128)).astype(np.float32),
        "B2": rng.standard_normal(64).astype(np.float32),
        "W3": rng.standard_normal((10, 64)).astype(np.float32),
        "B3": rng.standard_normal(10).astype(np.float32),
    }
    g = Graph(
        nodes=[
            _mk("flat", "Flatten", ["input"], ["flat_out"]),
            _mk("g1", "Gemm", ["flat_out", "W1", "B1"], ["g1_out"],
                attrs={"transB": 1, "alpha": 1.0, "beta": 1.0}),
            _mk("r1", "Relu", ["g1_out"], ["r1_out"]),
            _mk("g2", "Gemm", ["r1_out", "W2", "B2"], ["g2_out"],
                attrs={"transB": 1, "alpha": 1.0, "beta": 1.0}),
            _mk("r2", "Relu", ["g2_out"], ["r2_out"]),
            _mk("g3", "Gemm", ["r2_out", "W3", "B3"], ["out"],
                attrs={"transB": 1, "alpha": 1.0, "beta": 1.0}),
        ],
        inputs=[TensorInfo("input", shape=[2, 1, 28, 28])],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedMatMulBias" in pats, "命中 FusedMatMulBias (canonical)")
    check("FusedEWChain" in pats, "命中 FusedEWChain (embedded Add→Relu)")
    check(len(opt.nodes) == 3, f"MLP 折叠为 3 个 fused 节点 (实际 {len(opt.nodes)})")
    # 无死节点: 从 graph outputs 反向可达覆盖所有 opt nodes
    by_name = {n.name: n for n in opt.nodes}
    producers = opt.producer_map()
    reachable = set()
    stack = list(opt.output_names())
    while stack:
        t = stack.pop()
        src = producers.get(t)
        if src is not None and src.name not in reachable:
            reachable.add(src.name)
            for inp in src.inputs:
                stack.append(inp)
    all_names = {n.name for n in opt.nodes}
    dead = all_names - reachable
    check(len(dead) == 0, f"无死节点 (dead: {dead})")
    # F4 数值一致
    x = rng.standard_normal((2, 1, 28, 28)).astype(np.float32)
    o1 = MockRuntime(g).run({"input": x})
    o2 = MockRuntime(opt).run({"input": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"MLP F4 数值对齐 max_diff={md:.2e}")


# ===========================================================================
# Part B (续): 问题 1 — Flatten→Gemm(bias) 无 Relu 不吸收 Flatten, 不报 FusedGemmAct
# ===========================================================================

def test_flatten_no_relu_not_gemmact():
    """Flatten→Gemm(bias) 无 Relu → Flatten 保留, Gemm→FusedMatMulBias, 无 FusedGemmAct."""
    print("\n# 单元测试: Flatten→Gemm(bias) 无 Relu 不吸收 Flatten")
    rng = np.random.default_rng(42)
    W = np.eye(16, dtype=np.float32)
    C = np.ones(16, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[
            _mk("flat", "Flatten", ["input"], ["flat_out"]),
            _mk("gemm", "Gemm", ["flat_out", "W", "C"], ["out"]),
        ],
        inputs=[TensorInfo("input", shape=[2, 4, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedGemmAct" not in pats, "无 Relu → 不报 FusedGemmAct")
    check("FusedMatMulBias" in pats, "有 FusedMatMulBias (Gemm 独立融合)")
    # Flatten 应保留为独立节点
    flat_nodes = [n for n in opt.nodes if n.op_type == "Flatten"]
    check(len(flat_nodes) == 1, f"Flatten 保留为独立节点 (实际 {len(flat_nodes)})")
    # 数值对齐
    x = rng.standard_normal((2, 4, 4)).astype(np.float32)
    o1 = MockRuntime(g).run({"input": x})
    o2 = MockRuntime(opt).run({"input": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_flatten_no_relu_gemm_matmul_bias():
    """Flatten→Gemm(bias) 无 Relu → Gemm 中 fused_ops 应为 [MatMul, Add]."""
    print("\n# 单元测试: Flatten→Gemm(bias) 无 Relu → Gemm replay=MatMulBias")
    rng = np.random.default_rng(42)
    W = np.eye(16, dtype=np.float32)
    C = np.ones(16, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[
            _mk("flat", "Flatten", ["input"], ["flat_out"]),
            _mk("gemm", "Gemm", ["flat_out", "W", "C"], ["out"]),
        ],
        inputs=[TensorInfo("input", shape=[2, 4, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    fn = {n.name: n for n in opt.nodes}
    gemm_fused = [r for r in log if r["pattern"] == "FusedMatMulBias"
                  and "annotation" not in r]
    check(len(gemm_fused) >= 1, "实时 FusedMatMulBias 存在")
    if gemm_fused:
        fused_node = fn.get(gemm_fused[0]["fused_node"])
        check(fused_node is not None, "fused_node 在 opt 图中")
        if fused_node and fused_node.fused_ops:
            types = [m.op_type for m in fused_node.fused_ops]
            check(types == ["MatMul", "Add"],
                  f"fused_ops={types} 应为 [MatMul, Add]")


# ===========================================================================
# Part B (续): 问题 2 — 唯一命名
# ===========================================================================

def test_unique_name_allocation():
    """两个 Gemm 共享同名前缀时生成名字不冲突, 且不与已有 initializer 冲突."""
    print("\n# 单元测试: 唯一命名不冲突")
    rng = np.random.default_rng(42)
    # 两个 Gemm: g1 用 W1(4,4)+C1(4,), g2 用 W2(4,4)+C1(4,)
    W1 = np.eye(4, dtype=np.float32)
    W2 = np.eye(4, dtype=np.float32) * 2.0
    # 添加一个 initializer 名叫 "g1.B_fused" 来模拟冲突
    init = {
        "W1": W1.copy(),
        "W2": W2.copy(),
        "C": np.ones(4, dtype=np.float32),
        "g1.B_fused": np.full(4, 99.0, dtype=np.float32),  # 故意同名冲突
        "g1.C_fused": np.full(4, 88.0, dtype=np.float32),
    }
    g = Graph(
        nodes=[
            _mk("g1", "Gemm", ["A", "W1", "C"], ["o1"]),
            _mk("g2", "Gemm", ["B", "W2", "C"], ["o2"]),
        ],
        inputs=[TensorInfo("A", shape=[2, 4]), TensorInfo("B", shape=[3, 4])],
        outputs=[TensorInfo("o1"), TensorInfo("o2")], initializers=init)
    log, opt = _run_one_pass(g)
    # 两 Gemm 都应被融合
    real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                and "annotation" not in r]
    check(len(real_fmb) == 2, f"两个 Gemm 融合 (实际 {len(real_fmb)})")
    all_init_names = set(opt.initializers.keys())
    # "g1.B_fused" 已存在, allocator 应生成 "g1.B_fused_1" 等
    check("g1.B_fused" in all_init_names, "原有 g1.B_fused 仍在")
    # 确认新添加的 B_fused 有 unique 后缀
    b_fused_variants = [k for k in all_init_names if k.startswith("g1.B_fused")]
    check(len(b_fused_variants) >= 2,
          f"g1.B_fused 至少有 2 个变体 (实际 {len(b_fused_variants)})")
    # 数值对齐
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32),
            "B": rng.standard_normal((3, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_fused_name_collision():
    """fused_name (如 FusedMatMulBias::gemm) 与已有 node/tensor/initializer 名冲突时自动去重."""
    print("\n# 单元测试: fused_name 与已有符号冲突时自动去重")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32)
    # 已有节点名 FusedMatMulBias::gemm 和 initializer gemm.B_fused, gemm.C_fused
    init = {
        "W": W.copy(),
        "C": C.copy(),
        "gemm.B_fused": np.full(4, 99.0, dtype=np.float32),
        "gemm.C_fused": np.full(4, 88.0, dtype=np.float32),
    }
    g = Graph(
        nodes=[
            # 已有节点占用候选 fused_name "FusedMatMulBias::gemm"
            _mk("FusedMatMulBias::gemm", "Relu", ["dummy_in"], ["dummy_out"]),
            _mk("gemm", "Gemm", ["A", "W", "C"], ["out"]),
        ],
        inputs=[TensorInfo("A", shape=[2, 4]), TensorInfo("dummy_in", shape=[4])],
        outputs=[TensorInfo("out"), TensorInfo("dummy_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                and "annotation" not in r]
    check(len(real_fmb) == 1, f"Gemm 被融合 (实际 {len(real_fmb)})")
    if real_fmb:
        fused_name = real_fmb[0]["fused_node"]
        # fused_name 应与已有节点名不同
        check(fused_name != "FusedMatMulBias::gemm",
              f"fused_name 不与已有节点重名 (实际 {fused_name})")
        # fused_name 应在 opt 图中
        check(fused_name in {n.name for n in opt.nodes},
              f"fused_node '{fused_name}' 在 opt 图中")
        # 原始节点 "FusedMatMulBias::gemm" 仍在图中
        orig_node = next((n for n in opt.nodes if n.name == "FusedMatMulBias::gemm"), None)
        check(orig_node is not None, "原始 FusedMatMulBias::gemm 节点保留")
    # 数值对齐
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32),
            "dummy_in": np.zeros(4, dtype=np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_init_name_clash_active_bypass():
    """已有 initializer gemm.B_fused/gemm.C_fused 与 allocator 候选同名, 活跃旁路消费 gemm.C_fused."""
    print("\n# 单元测试: initializer 同名冲突且被活跃旁路消费")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    # 预塞入 "gemm.B_fused" 和 "gemm.C_fused" — allocator 会尝试用这两个名字
    # 生成新的初始值; 旁路 Add 活跃消费 gemm.C_fused, 因此该原有值必须保留.
    gemm_b_fused_init = np.array([99.0, 99.0, 99.0, 99.0], dtype=np.float32)
    gemm_c_fused_init = np.array([88.0, 88.0, 88.0, 88.0], dtype=np.float32)
    init = {
        "W": W.copy(),
        "C": np.ones(4, dtype=np.float32),
        "gemm.B_fused": gemm_b_fused_init,
        "gemm.C_fused": gemm_c_fused_init,
    }
    g = Graph(
        nodes=[
            _mk("gemm", "Gemm", ["A", "W", "C"], ["gemm_out"]),
            # 活跃旁路: 消费 "gemm.C_fused" initializer (值 88.0)
            _mk("bypass", "Add", ["gemm_out", "gemm.C_fused"], ["out"]),
        ],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                and "annotation" not in r]
    check(len(real_fmb) == 1, "Gemm 被融合")
    # allocator 应生成去重后的名字
    all_init_names = set(opt.initializers.keys())
    check("gemm.B_fused" in all_init_names, "原始 gemm.B_fused 保留")
    check("gemm.C_fused" in all_init_names, "原始 gemm.C_fused 保留")
    # 应有去重变体
    b_variants = [k for k in all_init_names if k.startswith("gemm.B_fused")]
    c_variants = [k for k in all_init_names if k.startswith("gemm.C_fused")]
    check(len(b_variants) >= 2, f"gemm.B_fused 至少有 2 个变体 (实际 {len(b_variants)})")
    check(len(c_variants) >= 2, f"gemm.C_fused 至少有 2 个变体 (实际 {len(c_variants)})")
    # 原始 gemm.C_fused 值不变 (旁路依赖它)
    check(np.allclose(opt.initializers["gemm.C_fused"], gemm_c_fused_init),
          "原始 gemm.C_fused 值不变")
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): 问题 3 — multi-consumer Flatten 不可吸收
# ===========================================================================

def test_flatten_multi_consumer_not_absorbed():
    """Flatten 输出有 Gemm 以外的消费者 → 不吸收 Flatten."""
    print("\n# 单元测试: Flatten multi-consumer 不吸收")
    rng = np.random.default_rng(42)
    W = np.eye(16, dtype=np.float32)
    C = np.ones(16, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    # Flatten 输出 "flat_out" 同时被 Gemm 和 bypass(Relu) 消费
    g = Graph(
        nodes=[
            _mk("flat", "Flatten", ["input"], ["flat_out"]),
            _mk("bypass", "Relu", ["flat_out"], ["bypass_out"]),
            _mk("gemm", "Gemm", ["flat_out", "W", "C"], ["out"],
                attrs={"transB": 1, "alpha": 1.0, "beta": 1.0}),
        ],
        inputs=[TensorInfo("input", shape=[2, 4, 4])],
        outputs=[TensorInfo("out"), TensorInfo("bypass_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    # Flatten 应保留
    flat_nodes = [n for n in opt.nodes if n.op_type == "Flatten"]
    check(len(flat_nodes) == 1, f"Flatten 保留 (实际 {len(flat_nodes)})")
    # 图形状正确
    flat_out_tensor = opt.nodes[0].outputs[0]
    # 两个输出都可达
    producers = opt.producer_map()
    check("out" in producers or any("out" in n.outputs for n in opt.nodes),
          "原输出 out 仍可达")
    check("bypass_out" in producers or any("bypass_out" in n.outputs for n in opt.nodes),
          "旁路输出 bypass_out 仍可达")
    # 数值对齐
    x = rng.standard_normal((2, 4, 4)).astype(np.float32)
    o1 = MockRuntime(g).run({"input": x})
    o2 = MockRuntime(opt).run({"input": x})
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_flatten_multi_consumer_with_relu():
    """Flatten→Gemm→Relu 但 Flatten 有旁路消费者 → Flatten 不吸收, Gemm+Relu 仍融合."""
    print("\n# 单元测试: Flatten multi-consumer + Relu → Flatten 不吸收, Gemm+Relu 融合")
    rng = np.random.default_rng(42)
    W = np.eye(16, dtype=np.float32)
    C = np.ones(16, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[
            _mk("flat", "Flatten", ["input"], ["flat_out"]),
            _mk("bypass", "Relu", ["flat_out"], ["bypass_out"]),
            _mk("gemm", "Gemm", ["flat_out", "W", "C"], ["g_out"],
                attrs={"transB": 1, "alpha": 1.0, "beta": 1.0}),
            _mk("relu", "Relu", ["g_out"], ["out"]),
        ],
        inputs=[TensorInfo("input", shape=[2, 4, 4])],
        outputs=[TensorInfo("out"), TensorInfo("bypass_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    flat_nodes = [n for n in opt.nodes if n.op_type == "Flatten"]
    check(len(flat_nodes) == 1, "Flatten 保留 (multi-consumer)")
    # Gemm+Relu 融合 (有 FusedGemmAct)
    check("FusedGemmAct" in _log_patterns(log), "Gemm+Relu 报 FusedGemmAct")
    # FusedMatMulBias 不一定出现, 因为 Gemm 已被 FusedGemmAct 消费
    # 数值对齐
    x = rng.standard_normal((2, 4, 4)).astype(np.float32)
    o1 = MockRuntime(g).run({"input": x})
    o2 = MockRuntime(opt).run({"input": x})
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): Item 1 — Flatten replay 复制 pred.attrs
# ===========================================================================

def test_flatten_replay_copies_attrs():
    """Flatten replay op 复制 pred.attrs (axis=2)."""
    print("\n# 单元测试: Flatten replay 复制 attrs (axis=2)")
    rng = np.random.default_rng(42)
    W = np.eye(16, dtype=np.float32)
    C = np.ones(16, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[
            # Flatten axis=2 on [2,4,4,4] -> [8, 16]; Gemm transB=1 with eye(16) preserves dim
            _mk("flat", "Flatten", ["input"], ["flat_out"], attrs={"axis": 2}),
            _mk("gemm", "Gemm", ["flat_out", "W", "C"], ["g_out"],
                attrs={"transB": 1, "alpha": 1.0, "beta": 1.0}),
            _mk("relu", "Relu", ["g_out"], ["out"]),
        ],
        inputs=[TensorInfo("input", shape=[2, 4, 4, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    fn = {n.name: n for n in opt.nodes}
    gemmact = [r for r in log if r["pattern"] == "FusedGemmAct"]
    check(len(gemmact) >= 1, "Flatten→Gemm→Relu → FusedGemmAct")
    if gemmact:
        fused = fn.get(gemmact[0]["fused_node"])
        check(fused is not None, "fused_node 存在")
        if fused and fused.fused_ops:
            flat_op = next((m for m in fused.fused_ops if m.op_type == "Flatten"), None)
            check(flat_op is not None, "fused_ops 含 Flatten replay")
            if flat_op:
                check(flat_op.attrs.get("axis") == 2,
                      f"Flatten replay axis=2 (实际 {flat_op.attrs.get('axis')})")
    x = rng.standard_normal((2, 4, 4, 4)).astype(np.float32)
    o1 = MockRuntime(g).run({"input": x})
    o2 = MockRuntime(opt).run({"input": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_flatten_replay_negative_axis():
    """Flatten replay op 复制 pred.attrs (axis=-2, 即 axis=1 for 3D tensor)."""
    print("\n# 单元测试: Flatten replay 复制 attrs (axis=-2)")
    rng = np.random.default_rng(42)
    W = np.eye(16, dtype=np.float32)
    C = np.ones(16, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[
            # Flatten axis=-2 on [2,4,4] -> effective axis=1 -> [2, 16]; Gemm transB=1 works
            _mk("flat", "Flatten", ["input"], ["flat_out"], attrs={"axis": -2}),
            _mk("gemm", "Gemm", ["flat_out", "W", "C"], ["g_out"],
                attrs={"transB": 1, "alpha": 1.0, "beta": 1.0}),
            _mk("relu", "Relu", ["g_out"], ["out"]),
        ],
        inputs=[TensorInfo("input", shape=[2, 4, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    fn = {n.name: n for n in opt.nodes}
    gemmact = [r for r in log if r["pattern"] == "FusedGemmAct"]
    check(len(gemmact) >= 1, "Flatten→Gemm→Relu → FusedGemmAct")
    if gemmact:
        fused = fn.get(gemmact[0]["fused_node"])
        if fused and fused.fused_ops:
            flat_op = next((m for m in fused.fused_ops if m.op_type == "Flatten"), None)
            if flat_op:
                check(flat_op.attrs.get("axis") == -2,
                      f"Flatten replay axis=-2 (实际 {flat_op.attrs.get('axis')})")
    x = rng.standard_normal((2, 4, 4)).astype(np.float32)
    o1 = MockRuntime(g).run({"input": x})
    o2 = MockRuntime(opt).run({"input": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): Item 1a — Flatten 旁路消费者已被更早 matcher consumed 时不得吸收
# ===========================================================================

def test_flatten_multi_consumer_bypass_consumed_earlier():
    """Flatten 输出给 gemm 和 bypass(Add→LN,被 _match_residual_norm 消费).
    
    旧逻辑用 `if c.name not in consumed` 过滤, 导致旁路 Add 被消费后只剩
    Gemm 一个消费者, 错误吸收 Flatten. 必须基于原图全部消费者判断.
    """
    print("\n# 单元测试: Flatten 旁路被更早 matcher 消费 — 不吸收 Flatten")
    rng = np.random.default_rng(42)
    init = {
        "mm_w": np.eye(16, dtype=np.float32),
        "ln_w": np.ones(16, np.float32),
        "ln_b": np.zeros(16, np.float32),
        "W": np.eye(16, dtype=np.float32),
        "C": np.ones(16, dtype=np.float32),
    }
    g = Graph(
        nodes=[
            _mk("matmul", "MatMul", ["mm_in", "mm_w"], ["mm_out"]),
            _mk("flat", "Flatten", ["input"], ["flat_out"]),
            _mk("add", "Add", ["mm_out", "flat_out"], ["add_out"]),
            _mk("ln", "LayerNormalization", ["add_out", "ln_w", "ln_b"], ["ln_out"]),
            _mk("gemm", "Gemm", ["flat_out", "W", "C"], ["out"]),
        ],
        inputs=[TensorInfo("input", shape=[2, 4, 4]),
                TensorInfo("mm_in", shape=[2, 16])],
        outputs=[TensorInfo("out"), TensorInfo("ln_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    # Flatten 必须保留为独立节点
    flat_nodes = [n for n in opt.nodes if n.op_type == "Flatten"]
    check(len(flat_nodes) == 1,
          f"Flatten 应保留 (旁路被消费前是多消费者), 实际 {len(flat_nodes)}")
    # Gemm 仍然被融合 (但不应吸收 Flatten)
    pats = _log_patterns(log)
    check("FusedMatMulBias" in pats, "Gemm 融合为 FusedMatMulBias")
    # 不应有 FusedGemmAct (那会表示吸收了 Flatten)
    gemmact_for_flat = [r for r in log if r["pattern"] == "FusedGemmAct"
                        and any("flat" in n for n in r.get("nodes", []))]
    check(len(gemmact_for_flat) == 0,
          f"FusedGemmAct 不应涉及 Flatten 吸收 (实际 {len(gemmact_for_flat)})")
    # LN 输出和 out 都应可达
    producers = opt.producer_map()
    check("ln_out" in producers or any("ln_out" in n.outputs for n in opt.nodes),
          "LN 输出 ln_out 保留")
    check("out" in producers or any("out" in n.outputs for n in opt.nodes),
          "Gemm 输出 out 保留")
    # 数值对齐
    x = rng.standard_normal((2, 4, 4)).astype(np.float32)
    feed = {"input": x, "mm_in": rng.standard_normal((2, 16)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): Item 3 — replay_ops 时 ext_inputs 从 replay ops 推导
# ===========================================================================

def test_replay_ops_ext_inputs_derivation():
    """replay_ops 存在时 ext_inputs 从 replay ops 输入推导, 含 B_fused/C_fused."""
    print("\n# 单元测试: replay_ops ext_inputs 推导 (含 B_fused/C_fused)")
    rng = np.random.default_rng(42)
    W = np.eye(16, dtype=np.float32)
    C = np.ones(16, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[
            _mk("flat", "Flatten", ["input"], ["flat_out"]),
            _mk("gemm", "Gemm", ["flat_out", "W", "C"], ["g_out"],
                attrs={"transB": 1, "alpha": 1.0, "beta": 1.0}),
            _mk("relu", "Relu", ["g_out"], ["out"]),
        ],
        inputs=[TensorInfo("input", shape=[2, 4, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    # Find the fused FusedGemmAct node
    gemmact = [r for r in log if r["pattern"] == "FusedGemmAct"]
    check(len(gemmact) >= 1, "FusedGemmAct 存在")
    if gemmact:
        fused = {n.name: n for n in opt.nodes}.get(gemmact[0]["fused_node"])
        check(fused is not None, "fused_node 存在")
        if fused:
            # ext_inputs should contain B_fused variant and C_fused variant
            has_gen_B = any("gemm.B_fused" in i for i in fused.inputs)
            has_gen_C = any("gemm.C_fused" in i for i in fused.inputs)
            check(has_gen_B, f"ext_inputs 含生成 B_fused (inputs={fused.inputs})")
            check(has_gen_C, f"ext_inputs 含生成 C_fused (inputs={fused.inputs})")
            # ext_inputs should NOT contain old W or C
            has_old_W = "W" in fused.inputs
            has_old_C = "C" in fused.inputs
            check(not has_old_W, f"ext_inputs 不再含旧 W (inputs={fused.inputs})")
            check(not has_old_C, f"ext_inputs 不再含旧 C (inputs={fused.inputs})")
    x = rng.standard_normal((2, 4, 4)).astype(np.float32)
    o1 = MockRuntime(g).run({"input": x})
    o2 = MockRuntime(opt).run({"input": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_replay_ops_ext_inputs_standalone_gemm():
    """Standalone Gemm canonicalization 同样推导正确的 ext_inputs."""
    print("\n# 单元测试: standalone Gemm replay ext_inputs 推导")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32)
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[_mk("gemm", "Gemm", ["A", "W", "C"], ["out"])],
        inputs=[TensorInfo("A", shape=[2, 4])],
        outputs=[TensorInfo("out")], initializers=init)
    log, opt = _run_one_pass(g)
    fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
           and "annotation" not in r]
    check(len(fmb) >= 1, "FusedMatMulBias 存在")
    if fmb:
        fused = {n.name: n for n in opt.nodes}.get(fmb[0]["fused_node"])
        check(fused is not None, "fused_node 存在")
        if fused:
            has_gen_B = any("gemm.B_fused" in i for i in fused.inputs)
            has_gen_C = any("gemm.C_fused" in i for i in fused.inputs)
            check(has_gen_B, f"ext_inputs 含 B_fused (inputs={fused.inputs})")
            check(has_gen_C, f"ext_inputs 含 C_fused (inputs={fused.inputs})")
            has_old_W = "W" in fused.inputs
            has_old_C = "C" in fused.inputs
            check(not has_old_W, f"ext_inputs 不再含旧 W (inputs={fused.inputs})")
            check(not has_old_C, f"ext_inputs 不再含旧 C (inputs={fused.inputs})")
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): 问题 4 — 测试矩阵补齐
# ===========================================================================

def test_gemm_transa_transb_matrix():
    """transA/transB 四组合: (0,0) (0,1) 可 canonicalize; (1,0) (1,1) fallback."""
    print("\n# 单元测试: transA/transB 四组合")
    rng = np.random.default_rng(42)

    for ta, tb, desc in [(0, 0, "transA=0,transB=0 canonical"),
                          (0, 1, "transA=0,transB=1 canonical"),
                          (1, 0, "transA=1,transB=0 fallback"),
                          (1, 1, "transA=1,transB=1 fallback")]:
        # Shape: for transA=1, A must be (out_features, batch_features)
        A_shape = (4, 2) if ta else (2, 4)
        W = np.eye(4, dtype=np.float32)
        C = np.ones(4, dtype=np.float32)
        init = {"W": W.copy(), "C": C.copy()}
        g = Graph(
            nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"],
                        attrs={"transA": ta, "transB": tb})],
            inputs=[TensorInfo("A", shape=A_shape)],
            outputs=[TensorInfo("out")], initializers=init)
        log, opt = _run_one_pass(g)
        real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                    and "annotation" not in r]
        if ta:  # transA=1 → fallback, no real fused node
            check(len(real_fmb) == 0,
                  f"{desc}: 无实时 FusedMatMulBias 节点 (实际 {len(real_fmb)})")
            # Gemm 保持原样
            check(any(n.op_type == "Gemm" for n in opt.nodes),
                  f"{desc}: Gemm 保留")
            # 无 strict 节点 (没有 FusedGemmAct / FusedMatMulBias fused node)
            fused_node_types = [n.op_type for n in opt.nodes if n.fused_ops]
            check(len(fused_node_types) == 0,
                  f"{desc}: 无 fused 节点 (实际 {fused_node_types})")
        else:  # transA=0 → canonical
            check(len(real_fmb) >= 1,
                  f"{desc}: 有实时 FusedMatMulBias (实际 {len(real_fmb)})")
        # 数值不变
        feed = {"A": rng.standard_normal(A_shape).astype(np.float32)}
        o1 = MockRuntime(g).run(feed)
        o2 = MockRuntime(opt).run(feed)
        md = float(np.max(np.abs(o1["out"] - o2["out"])))
        check(md <= 1e-3, f"{desc}: 数值对齐 max_diff={md:.2e}")


def test_gemm_bias_shapes():
    """bias scalar/vector/matrix 均正确 canonicalize 且数值对齐."""
    print("\n# 单元测试: Gemm bias 形状 (scalar/vector/matrix)")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)

    for bias_val, desc in [
        (np.float32(5.0), "scalar bias"),
        (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), "vector bias"),
        (np.full((1, 4), 0.5, dtype=np.float32), "matrix bias (1×N)"),
    ]:
        init = {"W": W.copy(), "C": np.asarray(bias_val)}
        g = Graph(
            nodes=[_mk("g", "Gemm", ["A", "W", "C"], ["out"])],
            inputs=[TensorInfo("A", shape=[2, 4])],
            outputs=[TensorInfo("out")], initializers=init)
        log, opt = _run_one_pass(g)
        real_fmb = [r for r in log if r["pattern"] == "FusedMatMulBias"
                    and "annotation" not in r]
        check(len(real_fmb) >= 1, f"{desc}: 有实时 FusedMatMulBias")
        feed = {"A": rng.standard_normal((2, 4)).astype(np.float32)}
        o1 = MockRuntime(g).run(feed)
        o2 = MockRuntime(opt).run(feed)
        md = float(np.max(np.abs(o1["out"] - o2["out"])))
        check(md <= 1e-3, f"{desc}: 数值对齐 max_diff={md:.2e}")


def test_shared_w_c_dual_path():
    """共享 W 和 C 同时被 canonicalized 及未融合活跃路径使用, 输出对齐且原键值不变."""
    print("\n# 单元测试: 共享 W/C 双路径 (融合 + 未融合)")
    rng = np.random.default_rng(42)
    W = np.eye(4, dtype=np.float32)
    C = np.ones(4, dtype=np.float32)
    w_before = W.copy()
    c_before = C.copy()
    init = {"W": W.copy(), "C": C.copy()}
    g = Graph(
        nodes=[
            _mk("g1", "Gemm", ["A", "W", "C"], ["g1_out"]),
            _mk("g2", "Gemm", ["B", "W", "C"], ["g2_out"]),
            # 活跃路径: 直接用 W 和 C 做 Add (不会被融合)
            _mk("active", "Add", ["W", "C"], ["active_out"]),
        ],
        inputs=[TensorInfo("A", shape=[2, 4]), TensorInfo("B", shape=[3, 4])],
        outputs=[TensorInfo("g1_out"), TensorInfo("g2_out"), TensorInfo("active_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check(np.allclose(init["W"], w_before), "共享 W 未被原地修改")
    check(np.allclose(init["C"], c_before), "共享 C 未被原地修改")
    feed = {"A": rng.standard_normal((2, 4)).astype(np.float32),
            "B": rng.standard_normal((3, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): 问题 5 — multi-consumer residual norm
# ===========================================================================

def test_residual_norm_multi_consumer():
    """Add 输出同时被 LN 与旁路消费 → FusedResidualNorm 融合, 外部输出存活."""
    print("\n# 单元测试: FusedResidualNorm multi-consumer")
    init = {"ln_w": np.ones(4, np.float32), "ln_b": np.zeros(4, np.float32)}
    g = Graph(
        nodes=[
            _mk("add", "Add", ["a", "b"], ["t"]),
            _mk("ln", "LayerNormalization", ["t", "ln_w", "ln_b"], ["ln_out"]),
            _mk("bypass", "Relu", ["t"], ["bypass_out"]),
        ],
        inputs=[TensorInfo("a", shape=[2, 4]), TensorInfo("b", shape=[2, 4])],
        outputs=[TensorInfo("ln_out"), TensorInfo("bypass_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedResidualNorm" in _log_patterns(log), "命中 FusedResidualNorm")
    # 外部输出 bypass_out 仍可达
    producers = opt.producer_map()
    check("bypass_out" in producers or any("bypass_out" in n.outputs for n in opt.nodes),
          "旁路输出 bypass_out 保留")
    check("ln_out" in producers or any("ln_out" in n.outputs for n in opt.nodes),
          "LN 输出 ln_out 保留")
    # 数值对齐
    rng = np.random.default_rng(42)
    feed = {"a": rng.standard_normal((2, 4)).astype(np.float32),
            "b": rng.standard_normal((2, 4)).astype(np.float32)}
    o1 = MockRuntime(g).run(feed)
    o2 = MockRuntime(opt).run(feed)
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): 问题 6 — F4 fail-closed 验证 (调用真实 score_f4)
# ===========================================================================

# ----- 直接测试 bench_c32_c33._numeric_align 的 fail-closed 行为 -----
# 这些测试调用真实的 benchmark _numeric_align 函数（非 selftest 的 score_f4），
# 验证其从 fail-open 改为 fail-closed 的正确性。

def test_bench_numeric_align_exception():
    """_numeric_align: runtime 异常 → (False, None) fail-closed.  (RED-phase test)"""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    import unittest.mock as umock
    g = Graph(
        nodes=[_mk("a", "Relu", ["x"], ["out"])],
        inputs=[TensorInfo("x", shape=[3, 4])],
        outputs=[TensorInfo("out")])
    opt = g.clone()
    feed = {"x": np.zeros((3, 4), dtype=np.float32)}
    with umock.patch.object(MockRuntime, 'run', side_effect=RuntimeError("mock-runtime-fail")):
        ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is False, "异常 → align_ok=False")
    check(md is None, "异常 → max_diff=None")


def test_bench_numeric_align_key_mismatch():
    """_numeric_align: 输出 key 集合不一致 → (False, None)."""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    rng = np.random.default_rng(42)
    shape = (2, 4)
    g = Graph(
        nodes=[_mk("a", "Relu", ["x"], ["out_a"])],
        inputs=[TensorInfo("x", shape=shape)],
        outputs=[TensorInfo("out_a")])
    opt = Graph(
        nodes=[_mk("a", "Relu", ["x"], ["out_b"])],
        inputs=[TensorInfo("x", shape=shape)],
        outputs=[TensorInfo("out_b")])
    feed = {"x": rng.standard_normal(shape).astype(np.float32)}
    ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is False, "key 不一致 → align_ok=False")
    check(md is None, "key 不一致 → max_diff=None")


def test_bench_numeric_align_shape_mismatch():
    """_numeric_align: 输出 shape 不一致 → (False, None)."""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    rng = np.random.default_rng(42)
    g = Graph(
        nodes=[_mk("mm", "MatMul", ["x", "W"], ["out"])],
        inputs=[TensorInfo("x", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers={"W": np.eye(4, 3, dtype=np.float32)})
    opt = Graph(
        nodes=[_mk("mm", "MatMul", ["x", "W"], ["out"])],
        inputs=[TensorInfo("x", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers={"W": np.eye(4, 5, dtype=np.float32)})
    feed = {"x": rng.standard_normal((2, 4)).astype(np.float32)}
    ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is False, "shape 不一致 → align_ok=False")
    check(md is None, "shape 不一致 → max_diff=None")


def test_bench_numeric_align_dtype_mismatch():
    """_numeric_align: 输出 dtype 不一致 → (False, None)."""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    rng = np.random.default_rng(42)
    g = Graph(
        nodes=[_mk("mm", "MatMul", ["x", "W"], ["out"])],
        inputs=[TensorInfo("x", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers={"W": np.eye(4, 3, dtype=np.float32)})
    opt = Graph(
        nodes=[_mk("mm", "MatMul", ["x", "W"], ["out"])],
        inputs=[TensorInfo("x", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers={"W": np.eye(4, 3, dtype=np.float64)})
    feed = {"x": rng.standard_normal((2, 4)).astype(np.float32)}
    ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is False, "dtype 不一致 → align_ok=False")
    check(md is None, "dtype 不一致 → max_diff=None")


def test_bench_numeric_align_nan_output():
    """_numeric_align: 输出含 NaN → (False, None)."""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    g = Graph(
        nodes=[_mk("id", "Relu", ["x"], ["out"])],
        inputs=[TensorInfo("x", shape=[3, 4])],
        outputs=[TensorInfo("out")])
    opt = g.clone()
    feed = {"x": np.full((3, 4), np.nan, dtype=np.float32)}
    ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is False, "NaN 输出 → align_ok=False")
    check(md is None, "NaN 输出 → max_diff=None")


def test_bench_numeric_align_inf_output():
    """_numeric_align: 输出含 Inf → (False, None)."""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    g = Graph(
        nodes=[_mk("id", "Relu", ["x"], ["out"])],
        inputs=[TensorInfo("x", shape=[3, 4])],
        outputs=[TensorInfo("out")])
    opt = g.clone()
    feed = {"x": np.full((3, 4), np.inf, dtype=np.float32)}
    ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is False, "Inf 输出 → align_ok=False")
    check(md is None, "Inf 输出 → max_diff=None")


def test_bench_numeric_align_identical():
    """_numeric_align: 完全相同的输出 → (True, 0.0)."""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    rng = np.random.default_rng(42)
    g = Graph(
        nodes=[_mk("id", "Relu", ["x"], ["out"])],
        inputs=[TensorInfo("x", shape=[3, 4])],
        outputs=[TensorInfo("out")])
    opt = g.clone()
    feed = {"x": rng.standard_normal((3, 4)).astype(np.float32)}
    ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is True, "相同输出 → align_ok=True")
    check(md == 0.0, f"相同输出 → max_diff=0.0 (实际 {md})")


def test_bench_numeric_align_threshold_exceeded():
    """_numeric_align: 有限但超 1e-3 → (False, 实际 max_diff)."""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    rng = np.random.default_rng(42)
    g = Graph(
        nodes=[_mk("mm", "MatMul", ["x", "W"], ["out"])],
        inputs=[TensorInfo("x", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers={"W": np.eye(4, dtype=np.float32)})
    opt = Graph(
        nodes=[_mk("mm", "MatMul", ["x", "W"], ["out"])],
        inputs=[TensorInfo("x", shape=[2, 4])],
        outputs=[TensorInfo("out")],
        initializers={"W": np.eye(4, dtype=np.float32) * 2.0})
    feed = {"x": rng.standard_normal((2, 4)).astype(np.float32)}
    ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is False, "超阈值 → align_ok=False")
    check(md is not None, "超阈值 → max_diff 可计算 (非 None)")
    check(md > 1e-3, f"超阈值 → max_diff>{1e-3:.1e} (实际 {md:.2e})")


def test_bench_numeric_align_empty_outputs():
    """_numeric_align: 空输出集合 → (False, None)."""
    from benchmarks.c32_c33.bench_c32_c33 import _numeric_align
    g = Graph(
        nodes=[_mk("id", "Relu", ["x"], ["out"])],
        inputs=[TensorInfo("x", shape=[3, 4])],
        outputs=[])
    opt = g.clone()
    feed = {"x": np.zeros((3, 4), dtype=np.float32)}
    ok, md = _numeric_align(g, opt, "mnist_mlp", feed_dict=feed)
    check(ok is False, "空输出 → align_ok=False")
    check(md is None, "空输出 → max_diff=None")

def test_score_f4_exception_fail_closed():
    """MockRuntime 运行异常 → score_f4 返回 F4=0, align_ok=False (fail-closed)."""
    print("\n# 单元测试: F4 fail-closed — 运行异常")
    g = Graph(
        nodes=[_mk("bad", "NonExistentOp", ["input"], ["out"])],
        inputs=[TensorInfo("input")], outputs=[TensorInfo("out")])
    opt = g.clone()
    # 调用真实 score_f4，应捕获异常并返回 F4=0 / align_ok=False
    f4, aok, md = score_f4(g, opt, "mnist_mlp",
                           feed_dict={"input": np.zeros((2, 1, 28, 28), dtype=np.float32)})
    check(f4 == 0.0, f"运行异常 → F4=0 (实际 {f4})")
    check(not aok, "运行异常 → align_ok=False")


def test_score_f4_key_mismatch_fail_closed():
    """输出 key 集合不一致 → score_f4 返回 F4=0, align_ok=False (fail-closed)."""
    print("\n# 单元测试: F4 fail-closed — 输出 key 不一致")
    # 两图输入/算子完全相同, 仅输出名不同, 使 keys_match=False
    shape = (2, 4)
    g = Graph(
        nodes=[_mk("a", "Relu", ["input"], ["out1"])],
        inputs=[TensorInfo("input", shape=shape)],
        outputs=[TensorInfo("out1")])
    opt = Graph(
        nodes=[_mk("a", "Relu", ["input"], ["out2"])],
        inputs=[TensorInfo("input", shape=shape)],
        outputs=[TensorInfo("out2")])
    # 确认 key 的确不同 (MockRuntime 按 output_names 返回)
    from runtime.mock_runtime import MockRuntime
    feed = {"input": np.zeros(shape, dtype=np.float32)}
    check(set(MockRuntime(g).run(feed).keys()) == {"out1"}, "g 输出 out1")
    check(set(MockRuntime(opt).run(feed).keys()) == {"out2"}, "opt 输出 out2")
    # 调用真实 score_f4
    feed_dict = {"input": np.random.default_rng(0).standard_normal(shape).astype(np.float32)}
    f4, aok, md = score_f4(g, opt, "mnist_mlp", feed_dict=feed_dict)
    check(f4 == 0.0, f"key 不一致 → F4=0 (实际 {f4})")
    check(not aok, "key 不一致 → align_ok=False")


def test_score_f4_per_output_nan_inf():
    """NaN/Inf 输出 → F4=0, align_ok=False."""
    print("\n# 单元测试: F4 per-output — NaN/Inf 检测")
    rng = np.random.default_rng(42)
    shape = (2, 4)
    # 原图输出正常, opt 图输出含 NaN → F4 全扣
    g = Graph(
        nodes=[_mk("a", "Add", ["input", "one"], ["out"])],
        inputs=[TensorInfo("input", shape=shape)],
        outputs=[TensorInfo("out")],
        initializers={"one": np.ones(4, dtype=np.float32)})
    # opt: 模仿一个产生 NaN 的坏融合 (直接造输出)
    opt_bad = Graph(
        nodes=[_mk("bad", "Constant", [], ["out"], attrs={"value": np.full(shape, np.nan, dtype=np.float32)})],
        inputs=[TensorInfo("input", shape=shape)],
        outputs=[TensorInfo("out")])
    feed = {"input": rng.standard_normal(shape).astype(np.float32)}
    f4, aok, md = score_f4(g, opt_bad, "mnist_mlp", feed_dict=feed)
    check(f4 == 0.0, f"NaN 输出 → F4=0 (实际 {f4})")
    check(not aok, "NaN 输出 → align_ok=False")
    # shape 不一致 → F4 全扣
    opt_wrong_shape = Graph(
        nodes=[_mk("bad", "Constant", [], ["out"], attrs={"value": np.full((3, 4), 0.0, dtype=np.float32)})],
        inputs=[TensorInfo("input", shape=shape)],
        outputs=[TensorInfo("out")])
    f4, aok, md = score_f4(g, opt_wrong_shape, "mnist_mlp", feed_dict=feed)
    check(f4 == 0.0, f"shape 不一致 → F4=0 (实际 {f4})")
    check(not aok, "shape 不一致 → align_ok=False")
    # dtype 不一致 → F4 全扣
    opt_wrong_dtype = Graph(
        nodes=[_mk("bad", "Constant", [], ["out"], attrs={"value": np.zeros(shape, dtype=np.float64)})],
        inputs=[TensorInfo("input", shape=shape)],
        outputs=[TensorInfo("out")])
    f4, aok, md = score_f4(g, opt_wrong_dtype, "mnist_mlp", feed_dict=feed)
    check(f4 == 0.0, f"dtype 不一致 → F4=0 (实际 {f4})")
    check(not aok, "dtype 不一致 → align_ok=False")


# ===========================================================================
# Part B (续): Task 5 — Dual-Conv residual add
# ===========================================================================

def test_dual_conv_residual_add_basic():
    """Two Convs → Add → Relu, each Conv consumed only by Add → FusedDualConvResidualAdd.
    
    Asserts fused_ops exactly [Conv, Conv, Add, Relu], two Conv nodes are distinct,
    no FusedConvResidualAdd, and embedded EW chain Add→Relu references same fused node.
    """
    print("\n# 单元测试: Dual-Conv residual add 微图 (含 fused_ops 精确断言)")
    rng = np.random.default_rng(42)
    init = {
        "w1": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b1": np.zeros(4, dtype=np.float32),
        "w2": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b2": np.zeros(4, dtype=np.float32),
    }
    # Both Convs take same input x, produce same-shape output; each consumed only by Add
    g = Graph(
        nodes=[
            _mk("conv_a", "Conv", ["x", "w1", "b1"], ["a_out"]),
            _mk("conv_b", "Conv", ["x", "w2", "b2"], ["b_out"]),
            _mk("add", "Add", ["a_out", "b_out"], ["add_out"]),
            _mk("relu", "Relu", ["add_out"], ["out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedDualConvResidualAdd" in pats,
          f"命中 FusedDualConvResidualAdd (pats={pats})")
    dual_recs = [r for r in log if r["pattern"] == "FusedDualConvResidualAdd"]
    check(len(dual_recs) == 1, f"恰好 1 个 DualConv 融合 (实际 {len(dual_recs)})")
    # ---- fused_ops 精确断言 ----
    fn_map = {n.name: n for n in opt.nodes}
    dual_fused = fn_map.get(dual_recs[0]["fused_node"])
    check(dual_fused is not None, "fused_node 在 opt 图中")
    if dual_fused and dual_fused.fused_ops:
        types = [m.op_type for m in dual_fused.fused_ops]
        check(types == ["Conv", "Conv", "Add", "Relu"],
              f"fused_ops == [Conv,Conv,Add,Relu] (实际 {types})")
        # Two Conv ops must be distinct (different names / weight inputs)
        convs = [m for m in dual_fused.fused_ops if m.op_type == "Conv"]
        check(len(convs) == 2, f"恰好 2 个 Conv replay (实际 {len(convs)})")
        if len(convs) == 2:
            check(convs[0].inputs[1] != convs[1].inputs[1],
                  "两个 Conv 使用不同 weight (distinct)")
    # ---- 无旧 FusedConvResidualAdd 抢占 ----
    check("FusedConvResidualAdd" not in pats,
          "没有旧 FusedConvResidualAdd 抢占")
    # ---- embedded EW chain: Add→Relu 引用同一 fused node ----
    ew_annots = [r for r in log if r["pattern"] == "FusedEWChain"
                 and r.get("annotation")]
    check(len(ew_annots) >= 1, "存在 embedded EW chain annotation")
    if ew_annots:
        add_relu_ann = next((r for r in ew_annots
                             if set(r["nodes"]) == {"add", "relu"}), None)
        if add_relu_ann is None:
            # fallback: check nodes contain add and relu
            add_relu_ann = next((r for r in ew_annots
                                 if "add" in r["nodes"] and "relu" in r["nodes"]), None)
        check(add_relu_ann is not None,
              f"embedded EW 记录精确含 Add/Relu 节点 (annots={ew_annots})")
        if add_relu_ann:
            check(add_relu_ann["fused_node"] == dual_recs[0]["fused_node"],
                  f"embedded EW 引用同一 fused node ({add_relu_ann['fused_node']})")
    # ---- 数值对齐 ----
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")
    # 无死节点
    producers = opt.producer_map()
    reachable = set()
    stack = list(opt.output_names())
    while stack:
        t = stack.pop()
        src = producers.get(t)
        if src is not None and src.name not in reachable:
            reachable.add(src.name)
            for inp in src.inputs:
                stack.append(inp)
    all_names = {n.name for n in opt.nodes}
    dead = all_names - reachable
    check(len(dead) == 0, f"无死节点 (dead: {dead})")


def test_dual_conv_bypass_consumer():
    """Conv_a output also feeds bypass Relu → no dual-conv fusion, bypass survives."""
    print("\n# 单元测试: Dual-Conv bypass consumer 回退")
    rng = np.random.default_rng(42)
    init = {
        "wa": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "ba": np.zeros(4, dtype=np.float32),
        "wb": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "bb": np.zeros(4, dtype=np.float32),
    }
    # conv_a output consumed by both Add and bypass_relu
    g = Graph(
        nodes=[
            _mk("conv_a", "Conv", ["x", "wa", "ba"], ["a_out"]),
            _mk("conv_b", "Conv", ["x", "wb", "bb"], ["b_out"]),
            _mk("add", "Add", ["a_out", "b_out"], ["add_out"]),
            _mk("bypass_relu", "Relu", ["a_out"], ["bypass_out"]),
            _mk("relu", "Relu", ["add_out"], ["out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("out"), TensorInfo("bypass_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedDualConvResidualAdd" not in pats,
          "旁路存在时不误报 FusedDualConvResidualAdd")
    # bypass_relu must survive
    bypass_nodes = [n for n in opt.nodes if "bypass_out" in n.outputs]
    check(len(bypass_nodes) >= 1, "旁路 bypass_out 存活")
    # 数值对齐
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_dual_conv_non_conv_input():
    """Add one input is Conv, other is non-conv → no dual-conv fusion."""
    print("\n# 单元测试: Dual-Conv 非 Conv 输入不误命中")
    rng = np.random.default_rng(42)
    init = {
        "w1": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b1": np.zeros(4, dtype=np.float32),
    }
    # Add gets one Conv output and one non-Conv residual tensor (same shape)
    g = Graph(
        nodes=[
            _mk("conv1", "Conv", ["x", "w1", "b1"], ["c1_out"]),
            _mk("add", "Add", ["c1_out", "residual"], ["out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8]),
                TensorInfo("residual", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    check("FusedDualConvResidualAdd" not in _log_patterns(log),
          "非双 Conv 输入不报 FusedDualConvResidualAdd")
    # 数值对齐
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    r = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x, "residual": r})
    o2 = MockRuntime(opt).run({"x": x, "residual": r})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")


def test_dual_conv_no_relu():
    """Two Convs → Add, no Relu → fused_ops = [Conv, Conv, Add]."""
    print("\n# 单元测试: Dual-Conv 无 Relu → fused_ops [Conv,Conv,Add]")
    rng = np.random.default_rng(42)
    init = {
        "w1": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b1": np.zeros(4, dtype=np.float32),
        "w2": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b2": np.zeros(4, dtype=np.float32),
    }
    g = Graph(
        nodes=[
            _mk("conv_a", "Conv", ["x", "w1", "b1"], ["a_out"]),
            _mk("conv_b", "Conv", ["x", "w2", "b2"], ["b_out"]),
            _mk("add", "Add", ["a_out", "b_out"], ["out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedDualConvResidualAdd" in pats, "命中 FusedDualConvResidualAdd")
    dual_recs = [r for r in log if r["pattern"] == "FusedDualConvResidualAdd"]
    check(len(dual_recs) == 1, "恰好 1 个 DualConv 融合")
    fn_map = {n.name: n for n in opt.nodes}
    dual_fused = fn_map.get(dual_recs[0]["fused_node"])
    if dual_fused and dual_fused.fused_ops:
        types = [m.op_type for m in dual_fused.fused_ops]
        check(types == ["Conv", "Conv", "Add"],
              f"fused_ops == [Conv,Conv,Add] (实际 {types})")
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_dual_conv_erf_not_absorbed():
    """Two Convs → Add → Erf: DualConv fuses Conv,Conv,Add but Erf stays separate."""
    print("\n# 单元测试: Dual-Conv Add→Erf — Erf 不被吸收, 保留独立")
    rng = np.random.default_rng(42)
    init = {
        "w1": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b1": np.zeros(4, dtype=np.float32),
        "w2": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b2": np.zeros(4, dtype=np.float32),
    }
    g = Graph(
        nodes=[
            _mk("conv_a", "Conv", ["x", "w1", "b1"], ["a_out"]),
            _mk("conv_b", "Conv", ["x", "w2", "b2"], ["b_out"]),
            _mk("add", "Add", ["a_out", "b_out"], ["add_out"]),
            _mk("erf", "Erf", ["add_out"], ["out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedDualConvResidualAdd" in pats,
          "DualConv 仍可融合前三项 (Conv,Conv,Add)")
    # Erf must survive as a standalone node
    erf_nodes = [n for n in opt.nodes if n.op_type == "Erf"]
    check(len(erf_nodes) == 1, "Erf 保留为独立节点")
    dual_recs = [r for r in log if r["pattern"] == "FusedDualConvResidualAdd"]
    if dual_recs:
        fn_map = {n.name: n for n in opt.nodes}
        fused = fn_map.get(dual_recs[0]["fused_node"])
        if fused and fused.fused_ops:
            types = [m.op_type for m in fused.fused_ops]
            # Must NOT include Erf
            check("Erf" not in types,
                  f"fused_ops 不含 Erf (实际 {types})")
            check(types == ["Conv", "Conv", "Add"],
                  f"fused_ops == [Conv,Conv,Add] (实际 {types})")
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_dual_conv_sqrt_not_absorbed():
    """Two Convs → Add → Sqrt: DualConv fuses Conv,Conv,Add but Sqrt stays separate."""
    print("\n# 单元测试: Dual-Conv Add→Sqrt — Sqrt 不被吸收, 保留独立")
    rng = np.random.default_rng(42)
    # Use positive-only weights + zero bias to ensure non-negative conv output
    init = {
        "w1": np.abs(rng.standard_normal((4, 4, 1, 1)).astype(np.float32)),
        "b1": np.zeros(4, dtype=np.float32),
        "w2": np.abs(rng.standard_normal((4, 4, 1, 1)).astype(np.float32)),
        "b2": np.zeros(4, dtype=np.float32),
    }
    g = Graph(
        nodes=[
            _mk("conv_a", "Conv", ["x", "w1", "b1"], ["a_out"]),
            _mk("conv_b", "Conv", ["x", "w2", "b2"], ["b_out"]),
            _mk("add", "Add", ["a_out", "b_out"], ["add_out"]),
            _mk("sqrt", "Sqrt", ["add_out"], ["out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedDualConvResidualAdd" in pats,
          "DualConv 仍可融合前三项 (Conv,Conv,Add)")
    sqrt_nodes = [n for n in opt.nodes if n.op_type == "Sqrt"]
    check(len(sqrt_nodes) == 1, "Sqrt 保留为独立节点")
    dual_recs = [r for r in log if r["pattern"] == "FusedDualConvResidualAdd"]
    if dual_recs:
        fn_map = {n.name: n for n in opt.nodes}
        fused = fn_map.get(dual_recs[0]["fused_node"])
        if fused and fused.fused_ops:
            types = [m.op_type for m in fused.fused_ops]
            check("Sqrt" not in types,
                  f"fused_ops 不含 Sqrt (实际 {types})")
            check(types == ["Conv", "Conv", "Add"],
                  f"fused_ops == [Conv,Conv,Add] (实际 {types})")
    # Use positive-only input to avoid sqrt of negative values
    x = np.abs(rng.standard_normal((2, 4, 8, 8)).astype(np.float32))
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_dual_conv_extra_consumer_consumed_earlier():
    """Add output has extra consumer (add_skip) already consumed by earlier matcher
    (_match_residual_norm consumes add_skip+LN).  Old code filtered by consumed,
    saw only Relu, and wrongfully absorbed it.  New code checks ALL original
    consumers — sees 2 consumers → does NOT absorb Relu.
    Side output and main output numerically identical.
    """
    print("\n# 单元测试: Dual-Conv Add 旁路消费者已被前序 matcher consumed → 不吸收 Relu")
    rng = np.random.default_rng(42)
    init = {
        "w1": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b1": np.zeros(4, dtype=np.float32),
        "w2": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b2": np.zeros(4, dtype=np.float32),
        # LN weight/bias shape matches last dimension (8) of 4D tensor (2,4,8,8)
        "ln_w": np.ones(8, np.float32),
        "ln_b": np.zeros(8, np.float32),
    }
    # add_out consumed by relu (main) and add_skip (bypass, gets consumed by
    # _match_residual_norm before DualConv runs).
    g = Graph(
        nodes=[
            _mk("conv_a", "Conv", ["x", "w1", "b1"], ["a_out"]),
            _mk("conv_b", "Conv", ["x", "w2", "b2"], ["b_out"]),
            _mk("add", "Add", ["a_out", "b_out"], ["add_out"]),
            _mk("relu", "Relu", ["add_out"], ["out"]),
            # bypass path: add_skip + LN, consumed by _match_residual_norm
            _mk("add_skip", "Add", ["add_out", "skip_input"], ["skip_add_out"]),
            _mk("ln", "LayerNormalization", ["skip_add_out", "ln_w", "ln_b"], ["ln_out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8]),
                TensorInfo("skip_input", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("out"), TensorInfo("ln_out")],
        initializers=init)
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedDualConvResidualAdd" in pats,
          "DualConv 仍可融合前三项 (Conv,Conv,Add)")
    # Relu must survive as standalone (total consumers of add_out = [relu, add_skip] = 2)
    relu_nodes = [n for n in opt.nodes if n.op_type == "Relu"]
    check(len(relu_nodes) == 1, "Relu 保留为独立节点 (因 add_out 有 2 个原始消费者)")
    # DualConv fused_ops must NOT include Relu
    dual_recs = [r for r in log if r["pattern"] == "FusedDualConvResidualAdd"]
    if dual_recs:
        fn_map = {n.name: n for n in opt.nodes}
        fused = fn_map.get(dual_recs[0]["fused_node"])
        if fused and fused.fused_ops:
            types = [m.op_type for m in fused.fused_ops]
            check(types == ["Conv", "Conv", "Add"],
                  f"fused_ops == [Conv,Conv,Add] (实际 {types})")
    # 旁路输出和主输出数值一致
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    skip = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x, "skip_input": skip})
    o2 = MockRuntime(opt).run({"x": x, "skip_input": skip})
    for k in o1:
        md = float(np.max(np.abs(o1[k] - o2[k])))
        check(md <= 1e-3, f"输出 {k} 数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# Part B (续): Task 5 — GlobalAveragePool→Flatten fusion
# ===========================================================================

def test_pool_flatten_basic():
    """GlobalAveragePool → Flatten with only consumer → FusedPoolFlatten.
    Asserts fused_ops == [GlobalAveragePool, Flatten].
    """
    print("\n# 单元测试: GlobalAveragePool→Flatten 融合 (含 fused_ops 断言)")
    g = Graph(
        nodes=[
            _mk("pool", "GlobalAveragePool", ["x"], ["pool_out"]),
            _mk("flat", "Flatten", ["pool_out"], ["out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("out")])
    log, opt = _run_one_pass(g)
    pats = _log_patterns(log)
    check("FusedPoolFlatten" in pats, "命中 FusedPoolFlatten")
    check(len(opt.nodes) == 1, f"折叠为 1 个融合节点 (实际 {len(opt.nodes)})")
    # fused_ops 精确断言
    pool_recs = [r for r in log if r["pattern"] == "FusedPoolFlatten"]
    if pool_recs:
        fn_map = {n.name: n for n in opt.nodes}
        fused = fn_map.get(pool_recs[0]["fused_node"])
        if fused and fused.fused_ops:
            types = [m.op_type for m in fused.fused_ops]
            check(types == ["GlobalAveragePool", "Flatten"],
                  f"fused_ops == [GlobalAveragePool, Flatten] (实际 {types})")
    # 数值对齐
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = float(np.max(np.abs(o1["out"] - o2["out"])))
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_pool_flatten_multi_consumer():
    """Pool output has 2+ consumers → no fusion."""
    print("\n# 单元测试: GlobalAveragePool multi-consumer 不融合")
    rng = np.random.default_rng(42)
    g = Graph(
        nodes=[
            _mk("pool", "GlobalAveragePool", ["x"], ["pool_out"]),
            _mk("flat", "Flatten", ["pool_out"], ["flat_out"]),
            _mk("relu", "Relu", ["pool_out"], ["relu_out"]),  # bypass
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("flat_out"), TensorInfo("relu_out")])
    log, opt = _run_one_pass(g)
    check("FusedPoolFlatten" not in _log_patterns(log),
          "multi-consumer 不报 FusedPoolFlatten")
    # Both pool and flat should remain
    pool_nodes = [n for n in opt.nodes if n.op_type == "GlobalAveragePool"]
    flat_nodes = [n for n in opt.nodes if n.op_type == "Flatten"]
    check(len(pool_nodes) == 1, "Pool 保留")
    check(len(flat_nodes) == 1, "Flatten 保留")
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


def test_pool_flatten_graph_output():
    """Pool output is graph output → no fusion."""
    print("\n# 单元测试: GlobalAveragePool 输出是 graph output 不融合")
    rng = np.random.default_rng(42)
    g = Graph(
        nodes=[
            _mk("pool", "GlobalAveragePool", ["x"], ["pool_out"]),
            _mk("flat", "Flatten", ["pool_out"], ["flat_out"]),
        ],
        inputs=[TensorInfo("x", shape=[2, 4, 8, 8])],
        outputs=[TensorInfo("pool_out"), TensorInfo("flat_out")])
    log, opt = _run_one_pass(g)
    check("FusedPoolFlatten" not in _log_patterns(log),
          "graph output Pool 不报 FusedPoolFlatten")
    pool_nodes = [n for n in opt.nodes if n.op_type == "GlobalAveragePool"]
    check(len(pool_nodes) == 1, "Pool 保留 (graph output)")
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    o1 = MockRuntime(g).run({"x": x})
    o2 = MockRuntime(opt).run({"x": x})
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    check(md <= 1e-3, f"数值对齐 max_diff={md:.2e}")
    check(opt.validate(), "优化图 validate() 通过")


# ===========================================================================
# main
# ===========================================================================
def _check_model_assertions():
    """公开模型硬门禁 (远端必两模型俱全; 任一缺失→FAIL)."""
    any_missing = False
    for key in _MODELS:
        fn, _, _, _ = _MODELS[key]
        path = os.path.join(_MODELS_DIR, fn)
        if not os.path.exists(path):
            check(False, f"{key} 模型缺失: {path} — 硬门禁 FAIL")
            any_missing = True
            continue
        from scheduler.graph import import_onnx_graph
        g = import_onnx_graph(path)
        stats, opt = _run_fusion(g)
        hits = sorted({e["pattern"] for e in stats["fusion_log"]
                       if not e.get("annotation")})
        if key == "mnist_mlp":
            # MLP: 完整 12 断言
            # raw_launches=9, raw_buffers=5
            check(stats["raw_launches"] == 9,
                  f"MLP raw_launches=9 (实际 {stats['raw_launches']})")
            check(stats["raw_buffers"] == 5,
                  f"MLP raw_buffers=5 (实际 {stats['raw_buffers']})")
            # F1: FusedMatMulBias + embedded EW chain
            all_f1 = {e["pattern"] for e in stats["fusion_log"]
                      if e["pattern"] in CANONICAL_PATTERNS}
            check("FusedMatMulBias" in all_f1, "MLP F1: FusedMatMulBias")
            check("FusedEWChain" in all_f1, "MLP F1: FusedEWChain (embedded)")
            # F2 ≥ 3.0, F3 ≥ 3.0
            F2, _ = score_f2(stats)
            F3, _ = score_f3(stats)
            check(F2 >= 3.0, f"MLP F2={F2} ≥ 3.0")
            check(F3 >= 3.0, f"MLP F3={F3} ≥ 3.0")
            # F4 = 4/4
            F4, aok, md = score_f4(g, opt, key)
            check(F4 == 4.0, f"MLP F4=4/4 (实际 {F4})")
            check(aok, f"MLP 数值对齐 max_diff={md:.2e}")
            # 总分 ~12
            total_mlp = round(len(all_f1 & CANONICAL_PATTERNS) + F2 + F3 + F4, 2)
            check(total_mlp >= 11.5, f"MLP total={total_mlp:.2f} ≥ 11.5")
        elif key == "cifar_resnet18":
            # ResNet: 明确断言 raw=69/47
            check(stats["raw_launches"] == 69,
                  f"ResNet raw_launches=69 (实际 {stats['raw_launches']})")
            check(stats["raw_buffers"] == 47,
                  f"ResNet raw_buffers=47 (实际 {stats['raw_buffers']})")
            # F1 hits 含 MatMulBias + EW, 用真实 score_f1 核验
            F1_val, strict_hits, _ = score_f1(opt, stats)
            check("FusedMatMulBias" in strict_hits,
                  "ResNet F1: FusedMatMulBias")
            check("FusedEWChain" in strict_hits,
                  "ResNet F1: FusedEWChain")
            check(F1_val == 2.0,
                  f"ResNet F1=2 (实际 {F1_val}): {sorted(strict_hits)}")
            # F2=3, F3=3, F4=4, opt_buffers<=18
            F2, (rl, ol) = score_f2(stats)
            F3, (rb, ob) = score_f3(stats)
            F4, aok, md = score_f4(g, opt, key)
            check(F2 == 3.0, f"ResNet F2=3.0 (实际 {F2})")
            check(ob <= 18, f"ResNet opt_buffers<=18 (实际 {ob})")
            check(F3 == 3.0, f"ResNet F3=3.0 (实际 {F3})")
            check(F4 == 4.0, f"ResNet F4=4/4 (实际 {F4})")
            check(aok, f"ResNet 数值对齐 max_diff={md:.2e}")
            # total = 12
            total_rn = round(F1_val + F2 + F3 + F4, 2)
            check(total_rn >= 12.0,
                  f"ResNet total>=12.0 (实际 {total_rn})")
            # 所有优化节点从 graph outputs 反向可达
            producers_rn = opt.producer_map()
            reachable_rn = set()
            stack_rn = list(opt.output_names())
            while stack_rn:
                t = stack_rn.pop()
                src = producers_rn.get(t)
                if src is not None and src.name not in reachable_rn:
                    reachable_rn.add(src.name)
                    for inp in src.inputs:
                        stack_rn.append(inp)
            all_names_rn = {n.name for n in opt.nodes}
            dead_rn = all_names_rn - reachable_rn
            check(len(dead_rn) == 0, f"ResNet 无死节点 (dead: {dead_rn})")
            # Final Gemm must NOT be FusedGemmAct
            gemmact_count = sum(1 for r in stats["fusion_log"]
                                if r["pattern"] == "FusedGemmAct")
            check(gemmact_count == 0,
                  f"ResNet FusedGemmAct 计数=0 (实际 {gemmact_count})")


def main() -> int:
    global _PASS, _FAIL
    print("=" * 72)
    print("C3.3 独立评委测试 (基于 spec.md / scoring.md, 不复用选手自评脚本)")
    print("=" * 72)

    print("\n=== Part A: 官方评测模型端到端打分 (mnist_mlp + cifar_resnet18) ===")
    totals = []
    any_model = False
    for key in _MODELS:
        t = judge_model(key)
        if t is not None:
            totals.append(t)
            any_model = True
    if totals:
        avg = float(np.mean(totals))
        print(f"\n>>> 官方评测模型 C3.3 均分 = {avg:.3f}/15")

    # Hard assertions when models exist (Task 7).
    # When all models missing, tests still run but at least main must fail.
    print("\n--- 公开模型硬断言 ---")
    _check_model_assertions()

    print("\n=== Part B: 各 pattern matcher 的 spec 触发条件单元测试 ===")
    test_fusedewchain_real_chain()
    test_fusedewchain_upper_bound()
    test_fusedewchain_single_not_match()
    test_compute_activation_not_ewchain()
    test_matmul_bias()
    test_matmul_bias_rejects_residual_add()
    test_residual_norm()
    test_softmax_dropout()
    test_conv_bn()
    test_conv_bn_exact_fused_ops()
    test_conv_bn_nonzero_bias_no_bn()
    test_conv_bn_multi_consumer()
    test_conv_bn_graph_output()
    test_conv_bn_dynamic_weight()
    test_conv_bn_bad_params()
    test_conv_bn_extra_output_consumed()
    test_conv_bn_extra_output_graph_out()
    test_prefuse_conv_bn_prefolded_annotation()
    test_correctness_numeric_align()
    # C3.3 Task 1-4 微图 / MLP 结构测试
    test_gemm_to_matmul_bias_default()
    test_gemm_to_matmul_bias_transb()
    test_gemm_to_matmul_bias_beta_only()
    test_gemm_alpha_non1_fallback()
    test_gemm_alpha_fp16_fallback()
    test_gemm_to_matmul_bias_broadcast()
    test_gemm_shared_initializer_not_mutated()
    test_gemm_no_bias_not_canonical()
    test_gemm_transa_fallback()
    test_mlp_structure_fusion()
    # 回归测试: 修复问题 1-6
    test_flatten_no_relu_not_gemmact()
    test_flatten_no_relu_gemm_matmul_bias()
    test_unique_name_allocation()
    test_init_name_clash_active_bypass()
    test_fused_name_collision()
    test_flatten_multi_consumer_not_absorbed()
    test_flatten_multi_consumer_with_relu()
    test_gemm_transa_transb_matrix()
    test_gemm_bias_shapes()
    test_shared_w_c_dual_path()
    test_residual_norm_multi_consumer()
    test_flatten_multi_consumer_bypass_consumed_earlier()
    # Item 1: Flatten replay attrs
    test_flatten_replay_copies_attrs()
    test_flatten_replay_negative_axis()
    # Item 3: replay_ops ext_inputs
    test_replay_ops_ext_inputs_derivation()
    test_replay_ops_ext_inputs_standalone_gemm()
    # Item 4: F4 per-output
    test_score_f4_per_output_nan_inf()
    # 原有 F4 fail-closed
    test_score_f4_exception_fail_closed()
    test_score_f4_key_mismatch_fail_closed()
    # 直接测试 bench_c32_c33._numeric_align fail-closed
    test_bench_numeric_align_exception()
    test_bench_numeric_align_key_mismatch()
    test_bench_numeric_align_shape_mismatch()
    test_bench_numeric_align_dtype_mismatch()
    test_bench_numeric_align_nan_output()
    test_bench_numeric_align_inf_output()
    test_bench_numeric_align_identical()
    test_bench_numeric_align_threshold_exceeded()
    test_bench_numeric_align_empty_outputs()
    # Task 5: Dual-Conv residual add
    test_dual_conv_residual_add_basic()
    test_dual_conv_bypass_consumer()
    test_dual_conv_non_conv_input()
    test_dual_conv_no_relu()
    test_dual_conv_erf_not_absorbed()
    test_dual_conv_sqrt_not_absorbed()
    test_dual_conv_extra_consumer_consumed_earlier()
    # Task 5: GlobalAveragePool→Flatten
    test_pool_flatten_basic()
    test_pool_flatten_multi_consumer()
    test_pool_flatten_graph_output()

    print("\n" + "=" * 72)
    print(f"=== { _PASS} passed, {_FAIL} failed ===")
    if totals:
        print("C3.3 维度打分汇总:")
        print(f"  {'模型':<18}{'F1':>5}{'F2':>7}{'F3':>7}{'F4':>5}{'合计':>8}")
        for key, F1, F2, F3, F4, total in _RESULTS:
            print(f"  {key:<18}{F1:>5}{F2:>7}{F3:>7}{F4:>5}{total:>8}")
        print(f"  {'均分':<18}{'':>5}{'':>7}{'':>7}{'':>5}{avg:>8}")
    # At least one model must be evaluated (远端必须执行两模型)
    if not any_model:
        print("  [WARN] 无公开模型可评测; main 将 fail 以示警示")
        _FAIL += 1  # ensure non-zero exit
    print("=" * 72)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
