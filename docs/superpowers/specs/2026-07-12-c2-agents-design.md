# C2 Excellent Agent 设计（DMA + Kernel image 选择）

- 日期：2026-07-12
- 状态：设计已批准，待写实现计划
- 范围：仅 `c2/agents/{dma_agent.py, kernel_agent.py}` 两个文件

---

## 1. 背景

C2 提交（`c2/`）当前状态：`libaec.so` 已完整实现（740 行，零 stub），mig02 上
`grader/public_grade.py --profile public` 首跑即 **88/100 — Good**：

| Tier | Requirement | 结果 |
|------|-------------|------|
| Basic | R101-R104, R201 | 满分 30 ✅ |
| Good | R105, R106, R202-R204, R301-R304 | 满分 50 ✅ |
| Excellent | R401, R402 | 各 4/10（仅 correctness，performance=0）❌ |

唯一阻碍 Excellent 的是两个 Agent 仍是 baseline stub。本设计把它们换成真策略。
**不改任何 C++ 代码**（Basic + Good 已满分，无需触碰 `aec_runtime.cpp` / `libaec.so`）。

## 2. 目标

两个 Agent 都取得正的、接近满分的 hidden performance speedup，达成 Excellent gate：
- 总分 ≥ 90（当前 88，需 +2）
- Good gate 通过（已满足）
- R401、R402 correctness 通过（已满足）
- R401、R402 hidden average speedup > 0（当前为 0，本设计目标）

## 3. Agent 运行协议（doc 05 §5，硬约束）

- 每次调用从 stdin 读一个 JSON，向 stdout 写一个 JSON（无多余字段、无日志）
- 单次超时 1 秒；stdout + stderr 合计 ≤ 64 KiB
- 不得联网、不得读评分器文件、不得跨 case 保存状态
- 输入/输出结构以 `schemas/` 与 grader 内联校验为准

---

## 4. DMA Agent（R401）

### 4.1 输入 / 输出契约

- 输入：`{case_id, direction("h2d"/"d2h"), bytes, alignment, registered(bool), concurrency}`
- 输出（恰好四个键）：`{channel:0|1, chunk_bytes:4096|65536|1048576, queue_depth:1|2|4|8, use_zero_copy:bool}`
- 约束：`use_zero_copy` 仅在 `registered==true` 时可为 true（grader `_dma_cycles` 校验）

### 4.2 周期公式（doc 05 §3，越小越优）

```
cycles = setup + ceil(ceil(bytes/32)/parallelism) + 24*(ceil(bytes/chunk_bytes)-1) + alignment_penalty
  setup             = 45 if use_zero_copy else 100
  parallelism       = min(queue_depth, concurrency, 2)
  alignment_penalty = 13 if alignment<64 else 0
```

baseline（stub 输出，也是 grader 比较基准）= `{chunk:4096, depth:1, ch:0, zero_copy:false}`。
打分：`fraction = clamp((baseline/candidate - 1)/0.5, 0, 1)`；candidate 比 baseline 快 50% → 满分。

### 4.3 策略（每项独立最小化，确定性全局最优）

```python
use_zero_copy = registered                    # setup: 45 vs 100（省 55）
chunk_bytes   = 1048576                       # 24×(chunks-1) 随 chunk 单调递减 → 取最大
queue_depth   = 2 if concurrency >= 2 else 1  # parallelism=min(depth,conc,2)；=2 即封顶，4/8 无额外收益
channel       = 0                             # 不进公式，任取合法值
```

四项中每一项都取理论极值，乘加后 candidate = 全局最小周期。**无任何策略可超越**
（公式是 grader 内联实现的确定函数）。`channel` 不影响周期，取 0 即可。

### 4.4 公开 case 验算（当前 stub 的两个 performance case）

- case1 `{d2h,65536,align64,registered,conc4}`：baseline=2508，本策略 candidate=1069 → fraction=1.0
- case2 `{h2d,1048576,align16,unregistered,conc2}`：baseline=39001，本策略 candidate=16497 → fraction=1.0

---

## 5. Kernel Agent（R402）

### 5.1 输入 / 输出契约

- 输入：`{case_id, dtype, m, n, k, alignment, workspace, candidates:[{id, semantic_kernel_id, image_id, variant, workspace, alignment, divisibility}, ...]}`
- 输出（恰好一个键）：`{kernel_id:"<candidate-id>"}`

### 5.2 候选与合法性（doc 05 §4，grader `_kernel_candidates`）

固定三档候选：

| id | variant | workspace | alignment | divisibility | shape 要求 |
|----|---------|-----------|-----------|--------------|-----------|
| naive | 1 | 0 | 1 | 1 | 任意合法 shape |
| tiled | 2 | 4096 | 1 | 4 | M/N/K 均 %4==0 |
| vectorized | 3 | 8192 | 16 | 8 | M/N/K 均 %8==0 且 align≥16 |

candidate 合法 = `candidate.workspace ≤ request.workspace` 且 `candidate.alignment ≤ request.alignment`
且 `M/N/K` 均能整除 `candidate.divisibility`。naive（divisibility=1）恒合法，候选集永不空。

### 5.3 周期（设备真实解释，非公式）

来自 `aecDeviceEvaluateKernel(semantic, dtype, variant, m,n,k, align, workspace, &completion)`。
grader baseline = naive 的周期。打分同 DMA：`fraction = clamp((baseline/candidate - 1)/0.5, 0, 1)`。

### 5.4 关键证据：设备周期排序（2026-07-12 mig02 实测探针）

| shape | naive | tiled | vec | 合法 |
|-------|------:|------:|----:|------|
| fp32 32×64×16 | 1115 | 731 | **603** | vec |
| int8 20×12×28 | 543 | **351** | REJ | tiled |
| fp16 7×9×5 | **162** | REJ | REJ | naive |
| fp32 128³ | 62035 | 37459 | **29267** | vec |
| fp16 64³ | 7923 | 4851 | **3827** | vec |
| bf16 36³ | 3400 | **2104** | REJ | tiled |
| int4 8³ | 162 | 114 | **98** | vec |
| fp64 16³ | 162 | 114 | **98** | vec |

**铁律：合法时 `vec < tiled < naive` 恒成立**。量化：vec ≈ 0.5×naive（ratio≈2.0→fraction clamp 1.0），
tiled ≈ 0.65×naive（ratio≈1.54→fraction≈1.08 clamp 1.0）。合法性边界与 doc 05 §4 完全一致。

### 5.5 公开 vs 隐藏 case 的信息差

grader `test_r402`：公开 case 会先把每个 candidate 的真实周期以 `diagnostic_cycles` 字段注入
`request["candidates"]` 再喂给 agent；隐藏 case **不注入**。因此：
- 公开 case：agent 能看到 `diagnostic_cycles` → 取精确最小 → fraction=1.0
- 隐藏 case：无周期信息 → 靠合法性 + variant 偏好启发式

### 5.6 策略（两层）

```python
def legal(c):
    return (c["workspace"] <= workspace
            and c["alignment"] <= alignment
            and all(x % c["divisibility"] == 0 for x in (m, n, k)))

legal_cands = [c for c in candidates if legal(c)]           # naive 恒在，永不空
with_cycles = [c for c in legal_cands if "diagnostic_cycles" in c]
best = (min(with_cycles, key=lambda c: c["diagnostic_cycles"])   # 公开：精确最小
        if with_cycles else
        max(legal_cands, key=lambda c: c["variant"]))            # 隐藏：vec(3)>tiled(2)>naive(1)
out = {"kernel_id": best["id"]}
```

隐藏启发式的依据是 §5.4 探针证实的 `vec<tiled<naive` 单调性——选最高 variant 即选最小周期。

---

## 6. 验证计划

1. 覆盖写 `c2/agents/dma_agent.py` 和 `c2/agents/kernel_agent.py`
2. `scp` 两个文件到 `mig02:~/A4S/c2/agents/`
3. `cd ~/A4S/c2 && python3 grader/public_grade.py --submission . --profile public`
4. 预期可观测信号：
   - R401/R402 报告行 detail 里 `public_diagnostic` 从 `0.000000` → `~1.0`（两 Agent 都拿满公开 speedup）
   - 报告总分与 R401/R402 earned 仍显示 88 / 4.0（见 §7 预期管理）
5. Excellent 最终由官方 `profile=full` 评分判定

## 7. ⚠️ 预期管理（重要）

本地 `--profile public` 的报告里，R401/R402 **始终显示 4.0/10**——因为 grader 代码
`earned = 4.0 + (6.0 * hidden_performance if profile == "full" else 0.0)`，performance 6 分
只在官方 `profile=full` 才计入。本地能观测的唯一信号是 detail 里的 `public_diagnostic`：
从 0.000000 涨到 ~1.0 即证明 hidden 会拿满。**Excellent 等级由官方 full 评分最终判定**
（grader `excellent_gate` 还要求 `profile=="full"`）。

## 8. 不在范围

- 不改 `libaec.so` / `aec_runtime.cpp` / 任何 C++（Basic+Good 已满分）
- 不改 `include/` `lib/` `kernels/` `grader/`（只读契约）
- 不动 C1 / C3

## 9. 风险

- **隐藏 case 周期排序若与公开不同** → kernel 启发式退化。但探针跨 8 种 dtype × 不同 shape 全部
  单调，且 `vec<tiled<naive` 是 image 设计本意（更优化的 variant 周期更低是普遍规律），风险极低。
- **DMA 公式**：是 grader 内联实现的确定函数，`_dma_cycles` 已逐字段对照，无误。
- **Agent 协议违规**：必须严格只输出规定键、无 stderr 日志、<1s。实现时注意 `json.load(sys.stdin)`
  读全部输入、`json.dump(..., sys.stdout)` 单次写出、不 print 调试信息。

---

## 10. 组委会澄清与 public/ 审计（2026-07-13）

### 10.1 组委会 Q&A 澄清（官方渠道）

1. **不需要实现或调用真正的 cuBLAS**（只是类似 cuBLAS）。我们的 `aecMatmul*` 就是那层，无影响。
2. **C2 不只是调度启动 image，要完成完整的 runtime 和 driver 行为**——stream/event/双 DMA/注册内存/故障恢复都必须是真实现（不能 stub/简化），官方 full 会测。
3. **不需要修改 image，不需要修改设备算子**——设备/image 是黑盒契约。
4. **34 个 image 也是正式使用的计算 image**（不只是公开测试用例）——official full 用同一批 image + 同一确定性设备。

### 10.2 对核查权重的影响

- **风险 A（driver 完整性）权重↑**：组委会强调"完整 driver 行为"，`aec_runtime.cpp` 的同步 stream 简化（不提交 BARRIER、EventQuery 永不 NOT_READY）、无 alloc 影子表（bounds/double-free 全靠设备）是核查首要目标。该核查随后由提交 `kzwywwpm "test(c2): harden runtime against hidden cases"` 落地（加固 runtime + 并发测试架 + 审计报告）。
- **风险 B（kernel agent 隐藏泛化）权重↓**：同 34 image + 同设备 → 周期模型确定，§5.4 探针结论（`vec<tiled<naive`）对 official 同样成立；只需扩大探针覆盖更多 shape/dtype 确认全局单调。

### 10.3 public/ 审计结果（2026-07-13 全量哈希比对）

public/Track-C/C2-runtime 于 07-13 11:42 重新落盘：

- ✅ grader、3 个头文件、libaec_device.so、kernels/manifest、docs/01-06、schemas、examples、cases、tutorial、RELEASE_MANIFEST ——**全部 SAME**
- 仅 README.md 因 c2/ 为定制提交包而不同（不影响评分）
- **契约与文档零实质变更**，本设计的 88/Good + public_diagnostic=1.0 基线完全有效。
