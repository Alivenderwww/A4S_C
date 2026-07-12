# 赛道 C 阅读笔记（编译器 / Runtime / 算子调度）

> 来源：`public/Track-C/` 下的官方文档 + Wikipedia（PTX / ONNX / CUDA）补充。
> 本文档仅做重点摘要，便于快速把握赛题框架与难点。

---

## 0. 全赛道定位

赛道 C 是 **AEC GPGPU 软件栈**：编译器 → Runtime/驱动 → 算子调度。三个子题 **独立评分**，原始分各 100，最终 **等权平均** 为赛道总分（满分 100）。

```text
C1 编译器    : PTX 风格 IR  ──►  AEC ISA 机器码
C2 Runtime  : 主机侧 libaec.so + 虚拟 GPGPU 设备
C3 算子调度  : ONNX 模型  ──►  AEC GPGPU 推理
```

> 评测环境：**评测期间无网络访问**；C2 使用**确定性虚拟设备**，性能以**虚拟周期**度量。

---

## 1. C1：AEC IR 编译器

### 1.1 输入/输出

- 输入：`input.ptx`（PTX 风格 IR；见后文结构样例）
- 命令：`aec-cc input.ptx -O{0|2|3} -o output.aecbin`
- 输出：`output.aecbin` = Header + Code Section（**128-bit 定长** AEC ISA）+ Data Section + Relocation + Symbol Table
- 还需提供反汇编器：`aec-objdump output.aecbin`

### 1.2 IR 结构（PTX 风格）

- 头部：`.version 1.0` / `.target aec_sm_10` / `.kernel name(...)`
- 寄存器声明：`.reg .f32 %f<10>;` `.reg .u32 %r<10>;` `.reg .u64 %rd<10>;` `.reg .pred %p<5>;`
- 常见指令：`ld.param.u64`、`ld/st.gmem`、`add.f32`、`setp`、`@%p bra $label`、`ret` 等
- 由选手自行设计 **CFG / SSA / pass 流水线**，但不能要求修改官方测试输入

### 1.3 必须的编译 Pass

| Pass | 要点 |
|------|------|
| IR 解析 | 语法、CFG、SSA |
| 标量优化 | 常量传播、DCE、CSE、LICM、基本块合并 |
| GPGPU 优化 | 内存合并访问、谓词执行、Shared Memory 提升 |
| 多精度 GEMM | 模式检测、自动选择 FP4/FP8/FP16/BF16/FP32/FP64/INT4/INT8/INT32 + Tile 自调 |
| 寄存器分配 | Linear scan 或图着色；**每线程最多 256 reg**；可 spill |
| 指令调度 | DDG + List Scheduling；**双发射配对**；访存/计算交织 |
| 代码生成 | 128-bit 定长指令编码；aecbin 文件格式完整 |

### 1.4 评分构成（100 分）

| 类别 | 分值 |
|------|----:|
| A 编译与执行正确性 | **50** |
| B 生成代码效率（`total_cycles`） | **35** |
| C 泛化与鲁棒性（变异测试） | **5** |
| D Agent 自动优化 | **10** |

**测试类别 100 题（隐藏） + 50 题变异**：

| 类别 | 数量 | 重点 | 性能分 |
|------|----:|------|-------:|
| T1 基础 Lowering | 20 | PTX 解析、基础指令 | 0 |
| T2 控制与标量 | 20 | CFG、谓词、DCE、CSE、LICM | 5 |
| T3 内存 | 20 | 合并访存、Shared Memory | 9 |
| T4 寄存器与调度 | 20 | Live Interval、Spill、DDG、双发射 | 10 |
| T5 Tensor / GEMM | 20 | TMUL、Tensor Load/Store、Tiling | 11 |

- **正确性是门禁**：正确才进入性能打分；性能分是周期数相对 O0/O2 基线的**几何平均改进**
- **Agent 评分**：`r_agent = T_default / T_agent`；≥ 1.25 满分（8 分）
- **诊断指标（不计分）**：`instruction_count`、`spill_count`、`dual_issue_rate`、`memory_transactions`、`stall_cycles`
- **隐藏精度矩阵**：PTX-05 覆盖全部 9 种 GEMM 精度，矩阵规模含非 16 倍数

### 1.5 公开测试（5 道）

| 编号 | 文件 | 类别 | 主题 | 规模 |
|------|------|------|------|------|
| PTX-01 | vector_add | T1 | Vector Add FP32 | N=4096, 256×16 |
| PTX-02 | invariant_poly | T2 | Loop Invariant, CSE, DCE | N=256, 256×1 |
| PTX-03 | repeated_reuse | T3 | Load Reuse, Shared Memory | N=4096, 256×16 |
| PTX-04 | reg_schedule | T4 | Live Interval, DDG, Dual Issue | N=8192, 256×32 |
| PTX-05 | gemm_f16 | T5 | TMUL Lowering, FP16 Tiling | 128³, 16×8×8 |

### 1.6 环境约束

8 cores / 16 GB / **180 s 编译超时** / Docker / 无网络。

### 1.7 关键难点（个人理解）

1. **正确性门禁**：先把基础 Lowering 跑通，先保证 100 题正确性分
2. **GEMM 跨 9 种精度**：PTX-05 一个文件要支持 9 种 dtype，最容易爆零 → 参考 §6.6.3 的 tensor‑core 指令形状速查与 §6.6.4 的 CUTLASS 设计
3. **变异测试**：参数/寄存器/块顺序/循环/数据类型全部会变，不能硬编码
4. **双发射配对**：AEC ISA 双发射是性能核心，需要建模延迟与配对规则 → 参考 §6.5.1 DDG、§6.5.2 List Scheduling、§6.5.4 调度 / 寄存器分配相位
5. **Agent 闭环**：必须能读 perf report → 调配置 → 重编译 → 验证
6. **寄存器上限 256**：spill 后落 local memory 极慢，要靠 Linear Scan + spill‑cost 排序，具体见 §6.4.1、§6.4.3

---

## 2. C2：主机侧驱动与 Runtime

### 2.1 任务

实现 `libaec.so`（Host Runtime） + 可选 Agent（Excellent 必需）。

**强制执行路径**：解析固定 image → 生成 little-endian 参数块 → `AEC_DEVICE_OP_ISA_LAUNCH` 交给虚拟设备。**禁止在 Host 端直接计算**或用自定义 image 绕过。

### 2.2 Runtime API（必须导出）

- 内存：`aecAlloc/Free`、`aecCopyH2D/D2H`、`aecCopyAsync`
- Kernel：`aecLaunch(kernel, gridDim, blockDim, args, stream)`
- Stream/Event：`aecStreamCreate/Sync`、`aecEventRecord`
- 设备：`aecDeviceCount/Info`
- 错误：线程局部错误状态；未实现功能返回 `AEC_ERROR_NOT_SUPPORTED`

### 2.3 计算库（10 种 GEMM dtype）

| 精度 | API |
|------|-----|
| FP4 E2M1 | `aecMatmulF4` |
| FP8 E4M3/E5M2 | `aecMatmulF8` |
| FP16 | `aecMatmulF16` |
| BF16 | `aecMatmulBF16` |
| FP32 | `aecMatmulF32` |
| FP64 | `aecMatmulF64` |
| INT4 | `aecMatmulI4` |
| INT8 | `aecMatmulI8` |
| INT32 | `aecMatmulI32` |

向量运算：`aecAxpy` / `aecDot` / `aecNrm2`

### 2.4 虚拟驱动

- 设备 ABI 合规
- **双 DMA 通道**（H2D + D2H）
- **注册内存（零拷贝）**
- **故障恢复**

### 2.5 Agent 接口（Excellent 必需）

从 stdin 读 JSON，stdout 输出合规 JSON：

**DMA Agent** 输入示例：
```json
{"case_id":1,"direction":"h2d","bytes":4096,"alignment":64,"registered":true,"concurrency":2}
```
输出示例：`{"channel":0,"chunk_bytes":4096,"queue_depth":2,"use_zero_copy":true}`

**Kernel Agent**：只能在合法 `candidates` 中选一个 `kernel_id`。

### 2.6 Starter Kit（已公开）

| 组件 | 位置 |
|------|------|
| 头文件 | `include/aec_runtime.h`、`aec_isa.h`、`aec_device_abi.h` |
| 固定 kernel image | `kernels/images/`（**34 个**） |
| 虚拟设备 | `lib/libaec_device.so` |
| 起始代码 | `src/aec_runtime.cpp` |
| 示例 | `examples/01..06_*.c` |
| 公开测试 | `cases/test_r101.py ~ test_r402.py` |
| 评分脚本 | `grader/public_grade.py` |

```bash
cd starter-kit && make -j2 && make examples
./bin/01_device_query
python3 grader/public_grade.py --submission . --profile public
```

### 2.7 评分构成（100 分）

| 模块 | 分值 | 内容 |
|------|----:|------|
| Runtime | 30 | R101-R106：查询/错误/内存/拷贝/Stream/Event/Launch |
| 计算库 | 30 | 10 种 GEMM + AXPY + DOT + NRM2 |
| 虚拟驱动 | 20 | ABI、双 DMA、注册内存、故障恢复 |
| Agent | 20 | DMA 策略 + Kernel image 选择 |

### 2.8 等级门槛

| 等级 | 必需能力 |
|------|----------|
| **Basic** | 查询/错误、内存、同步拷贝、Vector Add、FP32/INT32 GEMM |
| **Good** | Basic + Stream/Event、异步 DMA、注册内存、全部计算、故障恢复 |
| **Excellent** | Good + **两个合法 Agent** 并取得足够性能分 |

### 2.9 关键难点（个人理解）

1. **固定 34 个 image**：Kernel Agent 只能"挑"，不能"造"——需要按 dtype + shape + alignment + workspace + divisibility 做约束求解
2. **强制路径**：所有 launch 必须走 `AEC_DEVICE_OP_ISA_LAUNCH`；写 Host 端实现会被判违规
3. **10 种 dtype**：FP4/FP8/INT4 涉及子字节打包与对齐，FP64 在 GPGPU 上少见，需注意降精度路径 → 参考 §6.3.5 Tensor Core 演进（Volta→Blackwell 的精度支持时间线）
4. **双 DMA + 零拷贝**：小数据用 zero-copy，大数据走 chunk + queue depth
5. **确定性虚拟设备**：所有性能以虚拟周期度量，要看 starter-kit 里 cycle model 怎么算
6. **SIMT 网格与 host 抽象**：Host↔Device 调度模型可对照 §6.3.2/§6.3.3 的 CUDA 工业参考

---

## 3. C3：算子调度与模型部署

### 3.1 任务总览

实现 ONNX → AEC GPGPU 的端到端推理栈：

```text
ONNX → DAG → 算子分解 → 算子融合 → 内存规划 → AEC GPGPU 推理
```

| 子任务 | 分值 | 评测方式 |
|--------|----:|----------|
| C3.1 计算图解析 | 10 | 自动（ONNX → DAG JSON） |
| C3.2 算子分解 | 15 | 微基准（MNIST MLP / CIFAR ResNet-18） |
| C3.3 算子融合 | 15 | 微基准 |
| C3.4 内存规划 | 10 | **Code Review** |
| C3.5 端到端部署 | **50** | 三个模型实测 |
| **合计** | **100** | |

### 3.2 C3.1 计算图解析（10 分）

- 命令：`<prog> --onnx <model.onnx> --output <dag.json>`
- 命令模板报名时提交：`{onnx}` / `{output}` 占位
- 输出 JSON：`format_version` / `graph_inputs` / `graph_outputs` / `nodes`(op_type, inputs, outputs) / `edges`(src_node, dst_node, tensor)
- 加载 4 分 + 正确解析 6 分

### 3.3 C3.2 算子分解与内核选择（15 分）

评测脚本 `benchmarks/c32_c33/bench_c32_c33.py`，通过**公共 API 抓信号**：

| 信号 | API |
|------|-----|
| 原始 DAG | `import_onnx_graph(model.onnx)` |
| 精度决策 | `strategy.select_precision(node, graph) → PrecisionProfile` |
| 算子分解 | `strategy.decompose(node, graph, precision) → List[KernelSpecRef]` |
| 调优参数 | `strategy.tune_kernel(ref, precision, problem_size) → KernelTuningParams`，必须填 `block_x`/`grid_x`/`smem_bytes` |
| 中间张量 | 通过 `KernelSpecRef.outputs \ node.outputs` 识别（命名 `__c3_inter_N__`） |

5 个评分维度（D1–D5 各 3 分）：

| 维度 | 关键检查 |
|------|----------|
| **D1** 多精度路由 | 敏感算子（Softmax/LN/BN/ReduceMax/Sum/Mean）**强制 FP32**（×1.5）；精度多样度 fp32/fp16/fp8/fp4 至少 4 种（×1.0）；MatMul/Conv 用硬件支持精度（×0.5）。**硬指标**：`FULL_FP32` 模式下 `max_abs_diff ≤ 1e-3`、`top1_match ≥ 0.99` |
| **D2** 内核序列完整性 | MatMul(`matmul_*`)、Softmax(`reduce_max+exp+reduce_sum+div`)、LayerNorm(`reduce_mean+sub+mul+sqrt`)、Conv2d(`winograd_forward_*` 或 `im2col_*`) |
| **D3** 中间张量跟踪 | `len(KernelSpecRef.outputs \ node.outputs) > 0`；关键算子 = Softmax/LN/Conv2d |
| **D4** 调优参数有效性 | 覆盖率 ≥ 90%；3 条断言：①`0 < block_x ≤ max_threads_per_block` ②`grid_x > 0` ③`smem_bytes ≤ hardware.smem_bytes`（`-1` 视为合规） |
| **D5** 硬件能力覆盖 | ≥ 2 种精度 0.5，3–4 种满分；`matmul_f32+f16` 必备，`f8/f4` 各 +0.25；im2col 与 Winograd 都要出现过 |

> ⚠ 强行给敏感算子开低精度导致 `max_abs_diff > 1e-3` → **D1 的 3 分全扣**。

### 3.4 C3.3 算子融合与图优化（15 分）

`GraphPassPipeline(enable_fusion=True)` 跑一遍，从 `pass_results['Fusion']['stats']['fusion_log']` 抓信号。

| 维度 | 分值 | 检查 |
|------|----:|------|
| **F1** 融合 pattern 覆盖 | 5 | 命中一个 +1（见下表） |
| **F2** Kernel launch 数减少 | 3 | `min((raw - opt)/raw × 5, 3)`；reduction ≥ 60% 满分 |
| **F3** 中间 buffer 数减少 | 3 | 同公式，reduction ≥ 60% 满分 |
| **F4** 融合正确性 | 4 | outputs/inputs 可解析 +1×2；`graph.validate()` 通过 +1；节点数不增 +1；外加 MockRuntime 数值对齐（FP32 参考 `max_abs_diff ≤ 1e-3`，任一超阈 **F4 全扣**） |

5 个目标 pattern（F1）：

| Pattern | 触发 |
|---------|------|
| `FusedMatMulBias` | MatMul → AddBias |
| `FusedConv2dBatchNorm` | Conv2d → BatchNorm |
| `FusedEWChain` | 2–5 个相邻 elementwise |
| `FusedSoftmaxDropout` | Softmax → Dropout |
| `FusedResidualNorm` | skip-Add → LayerNorm |

> ⚠ 当前 ResNet-18 ONNX 训练时 BN 已折进 conv 权重（**无 BN 节点**），必须写**预融合 pass** 才能命中 `FusedConv2dBatchNorm`。

### 3.5 C3.4 内存规划与调度（10 分，Code Review）

| 子项 | 题目要求 | 满分条件 |
|------|----------|----------|
| **A** | 设备内存池 + 权重预加载 | 设备 alloc/free + 权重经计划上传到 device buffer 并被引用 |
| **B** | 中间张量 lifetime 复用 | 生命周期不重叠张量映射到同一 slot/物理缓冲，接入执行计划 |
| **C** | 内存池碎片整理 | free-list + best-fit / size class / coalesce |
| **D** | 权重预取 | 部分层权重 H2D 前移到前序计算附近（"边算边传"） |
| **E** | 流级并行 | 无依赖算子分配到不同 compute stream，计划中可见多 stream |

**空壳不得分**：仅注释/接口声明/stub/打印日志 → 0 分；命名可不同但须能对应到功能检查点。

等级：未通过 0–3 / 基础 4–5 / 良好 6–7 / 优秀 8–10。

### 3.6 C3.5 端到端部署（50 分，核心大题）

命令：
```bash
<prog> --onnx <model.onnx> --input <input_dir> --output <output_dir> [--batch-size N]
```

输入/输出格式：`manifest.json` + `<name>.npy`，dtype=float32，N 为动态维。

三个模型：

| 模型 | 任务 | 输入形状 | 准确率阈值 |
|------|------|----------|-----------|
| MLP | MNIST | `[N, 1, 28, 28]` | ≥ 98% |
| ResNet-18（简化） | CIFAR-10 | `[N, 3, 32, 32]` | ≥ 85% |
| Transformer（decoder-only） | 合成序列 | `[N, 18]` int64 → `[N, 18, 14]` | -- |

分值分配：

| 维度 | 分值 | 性质 |
|------|----:|------|
| 精度 + 准确率 | 15 | **通过门槛** |
| 运行时间 | 25 | 排序加分 |
| 峰值显存 | 10 | 排序加分（NVML 采样） |

门禁：
- 精度：`numpy.allclose(out, golden, rtol=1e-3, atol=1e-3)`（FP32 参考）
- 准确率：MLP ≥ 98%，ResNet ≥ 85%
- 任何门禁未过 → 模型 **0 分**

支持的 17 种 ONNX 算子：
`Add`、`Constant`、`Conv`、`Div`、`Erf`、`Flatten`、`Gather`、`Gemm`、`GlobalAveragePool`、`LayerNormalization`、`MatMul`、`Mul`、`Relu`、`Reshape`、`Softmax`、`Split`、`Transpose`

> ⚠ GELU 被分解为 `Div + Erf + Add + Mul`（图中无 Gelu 节点）

### 3.7 自评门槛（C3.2 + C3.3）

| 总分 | 评语 |
|------|------|
| ≥ 25 | S |
| 20–24 | A |
| 14–19 | B |
| 8–13 | C |
| < 8 | 未达标 |

### 3.8 关键难点（个人理解）

1. **精度门禁 `1e-3` 是硬约束**：低精度（TF32/FP16/BF16）跑 ResNet 容易超阈；如要低精度换性能，必须验证仍在阈值内
2. **C3.4 是 Code Review**：5 项都要在代码里**真实落地并接入执行计划**——空壳 0 分；算法理论见 §6.8（XLA / TVM / Triton 的 buffer reuse 实现）
3. **C3.5 占 50 分**：必须跑通三个模型（MLP / ResNet / Transformer）才算拿到大头
4. **Transformer 复杂**：含 LayerNorm、Softmax、Gather（词嵌入）、GELU 分解、Split、Reshape、Transpose——支持算子最多；各算子的 ONNX opset 对应见 §6.2.3
5. **D4 / F4 一票否决**：tuning 参数漏写 / 融合数值偏差 → 整维度全扣；融合的正确性与覆盖规则见 §6.7.2（TVM FuseOps 的 Phase 划分）和 §6.7.3（cuDNN 官方 fusion pattern）
6. **ResNet‑18 BN 折进 conv**：要命中 `FusedConv2dBatchNorm` 必须先写预融合 pass 把 BN 显式展开，公式见 §6.7.4

---

## 4. 三子题横向对比

| 维度 | C1 编译器 | C2 Runtime | C3 调度 |
|------|-----------|------------|---------|
| 性质 | 系统软件 | 系统软件 + 驱动 | 应用层 + 编译优化 |
| 核心难点 | GEMM + 双发射调度 + Agent 闭环 | 10 种 dtype + 固定 image 选择 | ONNX 全栈 + 精度门禁 + 内存优化 |
| 主要语言 | C++/Rust + MLIR/LLVM 风格 | C++17 + Python (Agent) | Python (主流) |
| 性能度量 | `total_cycles` | 虚拟周期 | 实测时间 + NVML 显存 |
| 评测门槛 | 180 s 编译超时 | -- | GPU 可用 |
| 等级 | 100 分制 | Basic/Good/Excellent | S/A/B/C |

---

## 5. 公开测试 & Starter Kit 一览

| 子题 | 公开测试 | Starter Kit | 评分脚本 |
|------|----------|-------------|----------|
| C1 | `C1-compiler/testcases/` (5 个 PTX) | 无独立 starter kit，需自建 `aec-cc` + `aec-objdump` | 隐藏评测 |
| C2 | `C2-runtime/starter-kit/cases/test_r101.py ~ test_r402.py` | 完整（头文件 + 34 image + 起始代码 + 6 个 example + 6 个 doc） | `grader/public_grade.py` |
| C3 | `C3-scheduler/testcases/`（公开模型 + golden/） | 评测脚本 `benchmarks/c32_c33/bench_c32_c33.py` | 隐藏模型评测 |

---

## 6. 配套背景知识（外部资料）

> 本节为联网调研后整理的深度背景阅读，建议结合各子题官方文档交叉阅读。
> 引用源标注于子节末尾或 §6.11。

### 6.1 PTX（Parallel Thread Execution）

PTX 是 NVIDIA 为 CUDA 设计的**虚拟 ISA + 中间表示**，AEC 把它当输入 IR 是合理的工业参考。来源：<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>。

#### 6.1.1 机器模型

- **多线程 SIMT**：GPU 由一组可扩展的流多处理器（SM）组成，每个线程映射到一个标量处理器核，独立 PC 与寄存器状态。SM 中的 SIMT 单元以 **warp（32 线程）** 为单位创建、管理、调度。
- **PTX 在工具链中的位置**：`C/C++ → nvcc / clang（LLVM 后端）→ PTX → GPU driver JIT → cubin（设备二进制）`。PTX 是稳定可移植的 IR（同 compute capability 源码可在更高能力 GPU 上 forward‑compatible 运行）。

#### 6.1.2 状态空间（State Spaces）

| 空间关键字 | 名称 | 作用域与生命周期 |
|-----------|------|------------------|
| `.reg` | 寄存器 | 线程私有，标量变量 |
| `.sreg` | 特殊寄存器 | 只读：`%tid`、`%ntid`、`%ctaid`、`%nctaid` 等 |
| `.const` | 常量 | 只读、可缓存，kernel 内只读 |
| `.global` | 全局内存 | 所有线程可见，跨 kernel 持久，访存延迟最高 |
| `.local` | 局部内存 | 线程私有，寄存器溢出（spill）时自动落到这里 |
| `.param` | 参数空间 | kernel / device function 参数 |
| `.shared` | 共享内存 | CTA（block）内所有线程共享，与 block 同生命周期 |
| `.tex` | 纹理 | 旧式只读缓存路径 |

> 寄存器**虚拟无限**（`.reg .f32 %f<10>;` 声明用 10 个，实际编译期由 `aec-cc` 决定映射哪些到物理 reg、哪些 spill）。

#### 6.1.3 指令与操作数

- **三操作数格式**：`op.type dst, src1, src2;`（AEC PTX 同样使用 `op.type dst, src1, src2;`）。
- **类型后缀**：`.f32` `.f16` `.f64` `.u8` `.u16` `.u32` `.u64` `.s32` `.s64` `.b32` `.b64` `.pred`。
- **谓词执行**：`@%p bra label;` 用谓词寄存器控制分支；亦可在多数算术指令前用 `@%p` 把指令设为条件执行。谓词寄存器名 `%p<5>` 同样属于 `.reg .pred` 状态空间。
- **特殊寄存器**（仅 `.sreg` 空间内可用）：
  - `%tid.x / .y / .z` —— 线程在 block 内三维 ID
  - `%ntid.x / .y / .z` —— block 三维形状
  - `%ctaid.x / .y / .z` —— block 在 grid 内三维 ID
  - `%nctaid.x / .y / .z` —— grid 三维形状
  - `%laneid`、`%warpid`、`%clock`、`%clock64` 等

#### 6.1.4 张量/矩阵指令（tensor core）

| PTX 指令类 | 来源 ISA | 典型形状 (M,N,K) | 用途 |
|------------|---------|------------------|------|
| `wmma.load / store / mma.sync` | Volta+ | 16×16×16 等 | warp 级 fragment 抽象 |
| `mma.sync.aligned` | Turing+ | F16: 16×8×16 (HMMA.16816)；TF32: 16×16×8；FP64: 8×8×4 | 直接寄存器编程 |
| `wgmma.mma_async` | Hopper+ | 大块异步 | warp‑group |

AEC 编译器可参考上述演进做**指令选择**：`ld.global.u16` → register → `mma.sync.aligned.m16n8k16` → epilogue → `st.global`。PTX-05 `gemm_f16` 即是典型的「未用 tensor core 的 naive GEMM」，优化空间很大。

#### 6.1.5 PTX 版本兼容性

PTX 与 SASS（设备二进制）解耦：PTX 9.0 写的 kernel 可在 Ada、Hopper 上 forward‑run；为追求极致性能也可"快照"在某个 compute capability 上（牺牲可移植性）。

---

### 6.2 ONNX（Open Neural Network Exchange）

来源：<https://onnx.ai/onnx/intro.html>、<https://en.wikipedia.org/wiki/ONNX>。

#### 6.2.1 起源与治理

- **起源**：最初由 PyTorch 团队内部称为 **Toffee**；2017‑09 由 **Facebook + Microsoft** 共同发布为 ONNX。
- **治理迁移**：2019‑11 进入 Linux Foundation AI，成为 **graduate 项目**；后续成员含 IBM、Huawei、Intel、AMD、Arm、Qualcomm。
- **目标**：（1）框架互操作（PyTorch、TensorFlow、JAX、MXNet、ONNX Runtime、PaddlePaddle）；（2）共用一个 IR 让硬件厂商在跨框架上做优化。

#### 6.2.2 模型结构

- **容器格式**：Protocol Buffers（`onnx.proto3` 定义）。
- **计算图**：有向无环图（DAG）。每个 node 调用一个 operator，输入输出为 tensor。
- **顶层字段**：`ir_version`、`opset_import`、`producer_name / version`、`graph`（`name`、`node[]`、`initializer[]`、`input[]`、`output[]`）。

#### 6.2.3 Opset 版本化（与本题 17 算子对照）

| 算子 | 出现过的 opset 版本 | 备注 |
|------|--------------------|------|
| **Add** | 1, 6, 7, 13, 14 | 元素级加法；v7 加入 broadcast 与 int64 |
| **Mul / Div** | 1, 6, 7, 13, 14 | 同上 |
| **Constant** | 1, 9, 11, 12, 13, 19, 21, 23, 24, 25 | 折叠常量 |
| **Conv** | 1, 11, 22 | v11 加入 group、auto pad 行为明确化 |
| **Gemm** | 1, 6, 7, 9, 11 | `C = αAᵀ?Bᵀ? + βC` |
| **MatMul** | 9, 13 | 二维或批量矩阵乘 |
| **Relu** | 1, 6, 13, 14 | |
| **Softmax** | 1, 11, 13 | v11 把 `axis` 改为强制属性 |
| **LayerNormalization** | 17+ | `[x, scale, bias] → [y, mean, inv_std]` |
| **Gather** | 1, 11, 13 | v13 改为负 axis 也允许 |
| **Reshape** | 1, 5, 13, 14, 19, 21, 23, 24, 25 | 频繁新增"allowzero"语义 |
| **Transpose** | 1, 13, 21, 23, 25 | |
| **Flatten** | 1, 9, 11, 13, 21, 23, 25 | v9 把 `axis` 改为强制 |
| **Split** | 1, 2, 11, 13, 18 | v2 改为 `split` 为输入张量 |
| **Erf** | 9, 13 | GELU 用 `Div → Erf → Add → Mul` 实现 |
| **GlobalAveragePool** | 1, 22 | |
| **Constant** | 1, …, 25 | |

> ⚠ C3 评测的 17 种算子**与 ONNX 算子的 op_type 完全一致**，C3.1 输出 JSON 的 `op_type` 字段必须照搬。

#### 6.2.4 数据类型

`tensor(float16 / float / double / int8 / int16 / int32 / int64 / uint8 / uint16 / bool)` + 可选 `sparse_tensor`。AEC 用 float32 作 golden，对 FP16/BF16/INT4/INT8 需有专门的 cast 节点。

---

### 6.3 CUDA 编译栈与 SIMT

来源：<https://en.wikipedia.org/wiki/CUDA>、<https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/>。

#### 6.3.1 历史

- 2004：Ian Buck（Stanford PhD，BrookGPU 作者）入职 NVIDIA，联手 John Nickolls 改造出 CUDA。
- 2007‑02‑16：CUDA 首次公开发布（macOS 1.1 / 正式 Linux SDK 在 2.0）。
- 2015+：重心明显倾向机器学习与神经网络负载。
- 2026‑05‑26：最新稳定版 **CUDA 13.3.0**。

#### 6.3.2 SIMT 模型

| 层级 | 硬件 | 代码语法 | 语义 |
|------|------|----------|------|
| 设备 | GPU | Program | 单次例程调用 |
| 粗 | SM | Grid | 并发同一子例程 |
| 中 | — | Block | 单个 Block 调度到 SM |
| 细 | Warp (32 threads) | — | SIMD 指令 |
| 细 | Thread（cuda core） | — | warp 内单标量 |

- **warp size = 32**（所有 compute capability）。
- **max threads / block**：512（cc 1.x）→ 1024（cc 2.x+）。
- **max grid dim**：2（D）→ 3（D）；**x 维上限**：65535 → 2³¹−1。

#### 6.3.3 内存层次（与 AEC `libaec.so` 虚拟通道的对照）

| 类型 | 所在硬件 | 作用域 | 备注 |
|------|----------|--------|------|
| Register | L0 / reg file | 线程 | 单线程 max 63–255 regs |
| Shared | on‑chip L1 | block | 16 KiB → 256 KiB |
| Local | L1 spill | 线程 | 寄存器溢出后 |
| Global | VRAM | 设备 | 高延迟 |
| Const | VRAM+L2 | 设备 | 64 KiB，缓存 |
| Texture | VRAM+L2 | 设备 | 只读，缓存 |

AEC 虚拟设备抽象了 H2D/D2H DMA + 注册内存（零拷贝）+ 双通道，作为 RTX / Pascal 之后 PCIe 的"工业参考简化版"。

#### 6.3.4 编译管线

`CUDA C/C++ (.cu) → nvcc/clang LLVM → PTX → 设备驱动 JIT → cubin`。AEC 把 PTX 当"前端 IR"，自己写后端到 128‑bit 定长 ISA。

#### 6.3.5 Tensor Core 演进

| 架构 | CC | 数据类型 | 典型 HMMA 形状 |
|------|----|---------|----------------|
| Volta | 7.0/7.2 | FP16 → FP16 | 16×16×16 |
| Turing | 7.5 | FP16/BF16 → FP32 | 16×16×16, 32×8×16, 8×32×16 |
| Ampere | 8.0/8.6 | TF32 / FP64 / BF16 / INT8 / 稀疏 | 16×16×8 (TF32), 8×8×4 (FP64) |
| Ada | 8.9 | FP8 (E4M3, E5M2) | 新形状 |
| Hopper | 9.0 | + FP8 + wgmma 异步 | warpgroup size 128 |
| Blackwell | 10.0/10.1 | + FP4 (E2M1), FP6 | 大幅扩展 mma |

> ⚠ AEC GEMM 跑在**自己的虚拟设备**上，与 NVIDIA 硬件不直接对应；但 AEC 的 9 种 dtype（FP4 E2M1 / FP8 E4M3+E5M2 / FP16 / BF16 / FP32 / FP64 / INT4 / INT8 / INT32）和**硬件支持的精度**一致（不包括 FP6 / 块缩放格式）。C3.2 D5 的硬件能力覆盖就是按这份"事实标准"评分。

来源：<https://en.wikipedia.org/wiki/CUDA> § "Chips with Tensor Cores"。

---

### 6.4 寄存器分配算法

来源：<https://en.wikipedia.org/wiki/Register_allocation>。

#### 6.4.1 Linear Scan（Poletto 1999）

- **核心**：把每个变量的活跃区间（live interval）按起点排序后**线性扫描**；不构造干涉图，**贪心**分配寄存器。
- **缺点**：不考虑区间"洞"（lifetime hole）；一旦 spill 整个区间都得 spill。
- **SSA 改进**（Wimmer & Franz, 2010）：每次定值是新的 live interval，能保留洞，活跃区间更短。
- **应用**：HotSpot client、V8、Jikes RVM、ART。**AEC 编译器推荐 Linear Scan**——单线程实现简单，O(n) 编译时间够用。

#### 6.4.2 图染色（Chaitin 1981, Briggs 改进）

- **核心**：节点 = 活跃区间；边 = 同时活跃 → **图染色**问题（NP‑complete 但有启发式可用）。
- **流水线**：`Renumber → Build → Coalesce → Spill cost → Simplify → Spill code → Select`。
- **质量**：代码质量通常优于 Linear Scan，但图在最坏情况下 O(n²)。
- **Aggressive / Conservative / Iterated / Optimistic** 四类 Coalescing 启发式。

#### 6.4.3 AEC 约束

- 每线程 **最多 256 regs**；超出必须 spill（写到 local memory）。
- 调试技巧：`spill_count` 是诊断指标，`T4` 类别 20 题评分含 Spill。
- 推荐 Simple Linear Scan + 按 spill cost 选 spill point（"def 之后尽快 store、use 之前尽快 load"）。

---

### 6.5 指令调度与双发射

来源：<https://en.wikipedia.org/wiki/Instruction_scheduling>。

#### 6.5.1 数据依赖图（DDG）

边类型（hazards）：

- **RAW**（Read After Write）：True 依赖，必须保序。
- **WAR**（Write After Read）：Anti 依赖，寄存器分配后产生。
- **WAW**（Write After Write）：Output 依赖。

每条边带 **latency**（流水线间隔）。剔除循环携带依赖后是 DAG，任意拓扑排序都有效。

#### 6.5.2 List Scheduling

1. 维护 ready set（所有前驱已调度）。
2. 优先级启发式：（a）资源跟踪（占用的执行单元扣分）、（b）latency proximity（早于 latency 调度扣分）、（c）critical path（关键路径加分）、（d）source creation（解锁后续的源加分）。
3. 调度最高优先级的 ready 指令；重复。

#### 6.5.3 软件流水线（Software Pipelining / Modulo Scheduling）

- 让循环不同迭代交错起来，提升 ILP。
- 关键参数：**II（Initiation Interval）** 决定每迭代启动间隔。
- Triton / CUTLASS `num_stages` 等价于这个深度。

#### 6.5.4 Schedule ↔ Register Allocation 相位

- **先调度后 RA**：最大化 ILP，但可能需更多寄存器触发 spill。
- **先 RA 后调度**（包含 AEC 双发射）：避免组合非法，但 RA 引入的 false 依赖限制调度自由度。
- AEC ISA 是"双发射配对"模型（128‑bit 定长指令，理论上一拍可发两条），需要**先做配对规则建模**（哪些功能单元可配对、共享资源），再做 list scheduling。

---

### 6.6 GEMM 优化与 Tensor Core

来源：<https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>、<https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/>、<https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wmma_programming.html>。

#### 6.6.1 GEMM 形式

`C ← α · op(A) · op(B) + β · C`，其中 A、B 可转置或共轭，全部矩阵可跨步（strided）。这是 BLAS level‑3；常通过"分成块矩阵"递归实现以获得 cache locality。

#### 6.6.2 经典 tiled GEMM（GotoBLAS / BLIS）

**微内核（microkernel）** 通常长这样：

1. **Pack A** 到一块连续的 micro‑panel（K×Mc）放进 L2/L1。
2. **Pack B** 到连续 K×Nc 面板。
3. 内层循环从 Mc/Nc 寄存器分块中累加（GEMM kernel）。
4. 这样把反复访存的 A/B tile 钉在 cache，外层 K 维再分块更新。

> AEC 没有真 cache 层，但**"寄存器分块 + shared memory pack + 标量 epilogue"** 的思路可以直接借鉴。

#### 6.6.3 Tensor Core 指令形状速查

| 精度 | 形状 (M, N, K) | 来源 |
|------|----------------|------|
| FP16 (Volta+) | 16×16×16 / 32×8×16 / 8×32×16 | wmma |
| FP16 (Turing+ SASS) | 16×8×16 (HMMA.16816) | mma.sync |
| TF32 (Ampere) | 16×16×8 | mma.sync |
| FP64 (Ampere) | 8×8×4 | mma.sync |
| FP8 (Ada) | 取决于架构 | mma.sync |
| FP4 (Blackwell) | NVFP4（块缩放） | cute DSL |

> AEC GEMM 调用图（C3.2）允许 tile 大小为**16×8×8** 等非标准值，但**必备 matmul_f32 + matmul_f16** 才能拿 D5 满分。

#### 6.6.4 CUTLASS 设计哲学（参考实现）

来源：<https://github.com/NVIDIA/cutlass>。

- **分层分解 + 可组合**：

  | 目录 | 抽象 | 描述 |
  |------|------|------|
  | `include/cutlass/arch/` | 架构原语 | 指令级 GEMM primitive |
  | `include/cutlass/thread/` | 线程 | 单 CUDA 线程内实现 |
  | `include/cutlass/epilogue/` | Epilogue | GEMM 后 bias、activation、转换 |
  | `include/cutlass/gemm/` | GEMM kernel | 通用矩阵乘 |
  | `include/cutlass/conv/` | 2D/3D conv | implicit GEMM |
  | `include/cute/` | CuTe DSL | layout / tensor / MMA atom |

- **CuTe** 把数据布局（Layout/Tensor）与执行索引耦合，能代数地**组合跨层级分块与分区**。`atom_layout_mnk = (2,2,1)` 代表把 (16,8,16) 的 atom 在 M/N 维各铺 2 倍用 4 warps。
- **混合精度支持**（与 AEC 完全对应）：FP64/FP32/TF32/FP16/BF16/FP8 (E5M2, E4M3)/MXFP4/MXFP6/MXFP8/INT4/INT8/FP32 仿真。

#### 6.6.5 关键启示（落到 AEC）

1. **指令选择**：`ld.global` → register fragment → `mma.sync` → epilogue → `st.global`。
2. **多精度**：要兼顾 9 种 dtype tile 大小，对一致问题（F32/F16/BF16）做 dtype‑lift 的 tile cache。
3. **性能节奏**：访存绑 kernel（elementwise/reduction）`Compute Bound` 反过来变访存依赖；GEMM 永远是 compute‑bound；fusion 才是救带宽的关键。

---

### 6.7 算子融合（Operator Fusion）

来源：<https://apxml.com/courses/intro-ml-compiler-optimization/chapter-3-graph-level-optimizations/operator-fusion-strategies>、<https://tvm.apache.org/docs/arch/fusion.html>、<https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html>、<https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/developer-guide/index.html>。

#### 6.7.1 三大融合类别

| 类型 | 典型例子 | 收益 | 难度 |
|------|----------|------|------|
| **Element‑wise 链** | Add→Mul→ReLU | 消除中间 tensor 落 HBM | 易 |
| **Reduction 之前 elementwise** | Square + Sum（平方和） | 与 reduce kernel 合并 | 中 |
| **GEMM/Conv 之后 elementwise**（最值钱） | Conv→Bias→ReLU；MatMul→Bias→GELU | 几乎免费 | 中等 |

> ⚠ 注意：reduction 不能"向前 fuse"——因为 sum 求全后才能做除法（Softmax 分母）。但 sum 之前的 broadcast elementwise 可融合。

#### 6.7.2 TVM FuseOps / FuseTIR

- 给每个算子打 `OpPatternKind`：`kElemWise`、`kBroadcast`、`kInjective`、`kReduction`、`kOutEWiseFusable`、`kOpaque`。
- **FuseOps** 在 DataflowBlock 内做：
  1. 构造 `IndexedForwardGraph`；
  2. **后支配树**（post‑dominator tree）通过 LCA 求出；
  3. 对每个节点判断是否可与 immediate post‑dominator 融合；
  4. 用 **Union‑Find** 把通过的中间节点并入同一组。
- **FuseTIR** 把同组的 PrimFunc 合并成单一 TIR，**消除中间 buffer**。
- 三个 Phase：
  - Phase 0：`kOutEWiseFusable`（Conv/MatMul）与 epilogue 融合。
  - Phase 1：injective / tuple 处理。
  - Phase 2：补融合此前已分组的 tuple 中间节点。
- **FuseOpsByPattern**（如 `matmul_bias → cutlass`）走外部 backend。

#### 6.7.3 cuDNN 官方 fusion pattern

- **ConvBiasAct**：Conv + elementwise `mul`、`add`、ReLU 三段点式（顺序固定）。FP16；mul 只能 per‑channel 标量张量；add 只能 column broadcast。
- **BnAddRelu**：BatchNorm + Add（skip connection）+ ReLU，针对 ResNet。仅 FP16/BF16/FP32；channel 必须是 8（FP16/BF16）或 4（FP32）的倍数。NHWC packed layout。
- **DReluForkDBn**：对应 backward 的 BN bias/scale 梯度计算。

#### 6.7.4 Torch.compile Conv+BN fuser

- 推理态 BN = (x − μ)/√(σ² + ε) × γ + β，可**直接折进 Conv 的 weight / bias**（无需新算子）：
  ```text
  fused_w  = conv_w   * (γ * rsqrt(σ² + ε)).reshape(-1, 1, 1, 1)
  fused_b  = (conv_b − μ) * rsqrt(σ² + ε) * γ + β
  ```
- 用户只需注册 `PatternMatcherPass`，举例输入 trace 模板，对**任意 shape** 都生效。

> ⚠ 与 C3.3 的关系：`benchmarks/c32_c33/bench_c32_c33.py` 里 ResNet‑18 ONNX 是推理图、BN 已折进 conv 权重（**无 BN 节点**）。要命中 `FusedConv2dBatchNorm`，必须写**预融合 pass** 把 BN 重新显式展开。

---

### 6.8 内存规划（Memory Planning）

C3.4 的核心算法理论。来源：<https://arxiv.org/pdf/2504.04874>（Futureproof Static Memory Planning）、XLA `buffer_assignment.cc`、TVM `static_plan_block_memory.cc`、Triton AutoWS commit。

#### 6.8.1 形式化问题（DSA, Dynamic Storage Allocation）

给定 N 个 buffer，每个 `(size, start_time, end_time)`，求给每个 buffer 分配 offset，使得分配总长度最短。**NP‑complete**（一般情形）；同尺寸时**退化为区间图染色（IGC）**，可用贪心 first‑fit / interval graph coloring 高效求解。

#### 6.8.2 XLA `BufferAssignment`

- 跑 `HloDataflowAnalysis` 得出**lifetime（definition_time, end_time）**；
- 全局按 size/对齐聚合到 `BufferAllocation`，每个 buffer 给一个 `(offset, size)`；
- 可以保留 `color` 字段区分 memory space（参数、常量、output、intermediate）。

#### 6.8.3 TVM `StaticPlanBlockMemory`

三阶段：

1. **init**：对每个 `alloc_tensor` 建 `StorageToken`（包含引用计数）。
2. **planning**：在可用 pool 中匹配能复用的 token（按 size + alignment + dtype + scope）。
3. **rewrite**：插入 `memory.alloc_storage` / `memory.alloc_tensor`。

只对"可被复用"的 tensor 建 token：函数参数、返回值、跨 BindingBlock 用户、被 If/Seq 用作条件 / body 的 tensor 都不参与。

#### 6.8.4 Triton AutoWS — Interval Graph Coloring

- **ScheduleGraph** 上构造每个 buffer 的 cycle‑级 lifetime。
- **合并规则**（`mergeNonOverlappingBuffers`）：modular overlap 判定、`same storage kind`、`shouldMerge`（只在 `max(size) × max(count) < Σ(size × count)` 时才合并，否则合并后浪费更大）、`mergeIntroducesCycle`（BFS 检查）。
- 每个 color 落一个 `PhysicalBuffer`（`sizeBytes = max(member.sizeBytes())`、`count = max(member.count)`）。
- 超预算（SMEM/TMEM）时调 `reduceBuffersForBudget`，按 cost 排序扣 buffer 数量，cost = `II_increase × tripCount / size_bytes_saved`。

#### 6.8.5 落地到 C3.4 五项检查

| 子项 | 库等价实现 |
|------|-----------|
| **A** 设备池 + 权重预加载 | TVM alloc_storage + block hoist |
| **B** lifetime 复用 | XLA `BufferAssignment` / TVM pool |
| **C** 碎片整理 | free‑list + best‑fit + coalesce |
| **D** 权重预取 | "compute‑transferring" H2D 前移；Triton AutoWS 的 `reduceBuffersForBudget` 思路 |
| **E** 流级并行 | 算子 DAG 切多 stream |

> **空壳判 0 分**：只留注释或打印的实现不算。命名可不同但要能在代码里找到对应的数据结构和算法。

---

### 6.9 MLIR（Multi‑Level Intermediate Representation）

来源：<https://mlir.llvm.org/>。

#### 6.9.1 核心思想

- **多级 IR**：在同一基础设施里表达**多抽象层级**的 IR。
- **Dialect**：用户可扩展算子生态（`gpu`、`nvvm`、`rocdl`、`spirv`、`tosa`、`llvm`、`affine`、`scf`、`linalg`、`vector`、`arith`、`memref`、`tensor`、`quant`、…）。
- **Progressive lowering**：高层算子逐级降到 vector→affine→llvm dialect→ISA。
- **SSA + symbol reference**：限制 SSA 作用范围，把跨函数引用做成显式 `symbol_ref` 属性，绕开 LLVM 多线程编译器限制。

#### 6.9.2 TableGen / ODS

- 操作定义用 **ODS（Operation Definition Specification）**——基于 TableGen 的声明式系统。
- 文档、verifier、printer/parser、rewrite 自动生成。

#### 6.9.3 与 AEC 的关系

AEC 编译器**没有规定**后端必须用 MLIR，但选手若选用：

- 可用 `gpu.thread_id`、`gpu.launch` 把 IR 归约；
- 用 `affine.for` 处理嵌套循环；
- 用 `memref.alloc` 配合 `BufferReusePass`（`openxla/xla/mlir_hlo/deallocation/transforms/buffer_reuse.cc`，保守/激进两档）做静态缓冲合并；
- 写一个 `aec-isa` dialect 承载 128‑bit 定长指令编码。

MLIR 是工业上**最低成本的"自建编译器骨架"路径**，比从 0 写 LLVM Pass 框架要快。

---

### 6.10 SSA 形式

来源：<https://en.wikipedia.org/wiki/Static_single_assignment_form>。

#### 6.10.1 定义与历史

- **性质**：每个变量只被赋值一次；控制流汇合点用 **φ‑function** 选择值。
- 由 IBM researchers（Alpern、Wegman、Zadeck）于 1980s 末提出，Cytron 等人给出 dominance frontier 算法构造 SSA。

#### 6.10.2 优化收益

SSA 简化了 use‑def 链，带来：

- **常量传播**
- **死代码消除（DCE）**
- **公共子表达式消除（CSE）**
- **全局值编号（GVN）**
- **条件常量传播**
- 改进的 **稀疏条件常量传播**

#### 6.10.3 与寄存器分配的协同

SSA 形式让每个定值都是独立活跃区间 → Linear Scan 分配更准（AEC 推荐）。LLVM、HotSpot、GCC GIMPLE、Go、V8、SpiderMonkey、HHVM、OCaml、HotSpot JVM、MSVC、SPIR‑V、CUDA 都已采用 SSA。

---

### 6.11 推荐参考链接

| 主题 | 链接 |
|------|------|
| PTX ISA | <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html> |
| WMMA / mma programming | <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wmma_programming.html> |
| Programming Tensor Cores (CUDA 9) | <https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/> |
| Using Tensor Cores in CUDA Fortran | <https://developer.nvidia.com/blog/using-tensor-cores-cuda-fortran/> |
| CUTLASS GitHub | <https://github.com/NVIDIA/cutlass> |
| ONNX opset | <https://onnx.ai/onnx/operators/index.html> |
| ONNX intro | <https://onnx.ai/onnx/intro.html> |
| ONNX Wiki | <https://en.wikipedia.org/wiki/ONNX> |
| CUDA Wiki | <https://en.wikipedia.org/wiki/CUDA> |
| Register allocation | <https://en.wikipedia.org/wiki/Register_allocation> |
| Instruction scheduling | <https://en.wikipedia.org/wiki/Instruction_scheduling> |
| BLAS | <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms> |
| MLIR | <https://mlir.llvm.org/> |
| SSA form | <https://en.wikipedia.org/wiki/Static_single_assignment_form> |
| TVM fusion | <https://tvm.apache.org/docs/arch/fusion.html> |
| XLA BufferAssignment | <https://github.com/openxla/xla/blob/main/xla/service/buffer_assignment.cc> |
| TVM static memory plan | <https://github.com/apache/tvm/blob/main/src/relax/transform/static_plan_block_memory.cc> |
| Triton AutoWS merge | <https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/AutoWSBufferMerging.cpp> |
| cuDNN fusion patterns | <https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/developer-guide/index.html> |
| Torch compile Conv+BN fuser | <https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html> |
| Futureproof Static Memory Planning | <https://arxiv.org/pdf/2504.04874> |
| Operator Fusion Strategies | <https://apxml.com/courses/intro-ml-compiler-optimization/chapter-3-graph-level-optimizations/operator-fusion-strategies> |
| XLA buffer reuse pass | <https://github.com/openxla/xla/blob/main/xla/mlir_hlo/deallocation/transforms/buffer_reuse.cc> |

---

## 7. 准备建议（按性价比排序）

### C2 性价比最高
- Starter Kit 最完整（头文件 + 起始代码 + 34 个 image + example + 公开测试 + 评分脚本 + 6 篇 doc）
- 先拿 Basic → Good（不写 Agent 就有 80 分）
- Excellent 需要 Agent：DMA + Kernel 一起做

### C3 大头（50 分在 C3.5）
- 先 C3.5 把 MLP / ResNet 跑通（拿 15 分门禁 + 部分排序分）
- C3.1 + C3.4 都是相对独立的小任务
- C3.2 + C3.3 是微基准，自动评测较容易拿分

### C1 难度最高
- 需要完整的编译器栈（IR → Opt → RegAlloc → Schedule → Codegen）
- 优先 T1 + T5：基础 Lowering + GEMM 是大头
- Agent 自动优化是 10 分中的 8 分，需要做闭环

---

*文档生成时间：2026-07-12，基于 public/Track-C/ 内 spec.md 与 scoring.md 及 Wikipedia 资料。*