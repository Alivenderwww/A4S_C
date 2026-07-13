# R401/R402 模拟 full-profile 测试与策略改进

> **状态**：已批准，待实现

## 目标

在不拥有官方 hidden case 数据的前提下，通过设备 oracle 扫描模拟 full-profile 评分流程，实现两个目标：

1. **发现策略反例**：找出 Kernel Agent 当前"最高合法 variant"不是最快的 case，并据此改进策略。
2. **估算隐藏得分**：用独立留出集计算 grader 风格 fraction，估算 R401/R402 hidden performance。

## 背景

- 公开 grader `--profile public` 上限 88/100；R401/R402 hidden performance 未评估。
- R401：DMA Agent 策略已穷举证明在 5 组请求上最优，但未覆盖全部 direction×registered 组合，也未模拟 grader fraction 聚合。
- R402：Kernel Agent 无 diagnostic_cycles 时选择最高合法 variant（vectorized > tiled > naive），但从未调用 `aecDeviceEvaluateKernel` 验证这是否真的最快。
- 官方 hidden 数据未知，任何模拟只能估算而非等价。

## 架构

```
device oracle (远端 Linux)
    │  扫描代表性网格 × 3 variant
    ▼
scan_results.json
    │  固定 seed 70/30 划分
    ├──► 探索集 → 反例分析 → 策略改进 → 零回退验证
    └──► 留出集 → grader 风格 fraction 估算
```

## 组件

### 1. Device Oracle Scanner（远端运行）

文件：`C2/tests/device_oracle_scan.py`

扫描代表性网格：
- 10 dtype：FP4/FP8-E4M3/FP8-E5M2/FP16/BF16/FP32/FP64/INT4/INT8/INT32
- shape 类别：1×1×1、16的倍数、非16倍数余数（如 17×13×9）、极端瘦长（1×256×1、256×1×256）
- alignment tier：8、16、64
- workspace tier：0、小（如 100）、大（如 8192+）
- 每个 grid point 对 naive/tiled/vectorized 调用 `aecDeviceEvaluateKernel`，记录真实 cycles 和合法性

输出：`scan_results.json`（临时，不提交）

### 2. 探索集分析器

文件：`C2/tests/test_r402_oracle.py`

对探索集 70% 的 case：
- 找出"当前策略选择 ≠ 最快合法 variant"的反例
- 归类反例模式（某 dtype 下 tiled 比 vectorized 快？小 shape 下 naive 最快？）
- 若发现可解释规律 → 改进 `kernel_agent.py` 的无 diagnostic 启发式
- 若无规律或反例极少 → 保持当前策略

### 3. 留出集评分

对留出集 30% 的 case（策略定型前不可见）：
- 按当前 Agent 策略选 variant，用 device cycles 计算 grader 风格 fraction
- 聚合为模拟 R402 得分
- 与探索集分数对比，检查过拟合

### 4. R401 全域验证

文件：`C2/tests/test_r401_exhaustive.py`

- 扩展现有穷举测试的请求网格：覆盖全部 direction×registered×bytes×alignment×concurrency 组合
- 对每组验证 Agent 输出 == 穷举最优
- 留出集同样计算 grader 风格 fraction

## 策略改进门槛（零回退）

新策略接受条件全部满足：
- 探索集平均 fraction 提升
- 公开 R402 `public diagnostic=1.0` 不退化
- 留出集任何单 case 不比当前策略更差
- 留出集平均 fraction 不低于当前

## 数据集划分

- 固定 seed 生成代表性网格 case
- 按固定 seed 自动划分 70% 探索 / 30% 留出
- 留出集在策略定型后才运行评分
- 两组统计特征（dtype 分布、shape 类别分布）应一致

## 运行方式

- Scanner 在远端 Linux 运行（需要 `libaec_device.so`）
- 分析和评分可在本地或远端运行（纯 Python + scan_results.json）
- 集成到 `run_hidden_style.py` 发现机制，或作为独立入口

## 不做的事

- 不修改官方 grader 或 `additional_profile`
- 不声称等价于官方 hidden 数据
- 不引入 ML/决策树（如果简单规律能覆盖就不上复杂方案）
- R401 公式已公开且动作空间小，策略改动风险极低

## 文件清单

| 文件 | 职责 |
|------|------|
| `C2/tests/device_oracle_scan.py` | 远端扫描器，调用 `aecDeviceEvaluateKernel` |
| `C2/tests/test_r402_oracle.py` | 探索分析 + 反例检测 + 留出评分 |
| `C2/tests/test_r401_exhaustive.py` | R401 大规模穷举验证 |
| `C2/agents/kernel_agent.py` | 若有反例，改进启发式 |
| `scan_results.json`（临时） | 扫描输出，不提交 |
