# 赛道 C 作战手册 · 知识库 + 赛题解析对齐

> 面向 3 人小组（C1/C2/C3 各一人并行），3 天完成。
> 本文合并：官方公开 spec（`public/Track-C/`）+ 根 `README.md` 增强文档 + 实际交付物核对 + 联网检索到的可复用知识库。
> 归一化：`赛道总分 =(C1+C2+C3)/3`，**三题都必须交**，缺一题 = 丢 33 分。

---

## 0. 增强文档补充的关键信息（公开 spec 里没有）

| 项 | 内容 | 影响 |
|---|---|---|
| **C1 正确性权重** | 50 分按类别加权：**T1×4 + T2×8 + T3×10 + T4×12 + T5×16** | T5(TMUL/GEMM) 权重最高，别把 GEMM 放最后 |
| **C1 性能归一化** | `r=T_base/T_你`；`p(r)=clip((log r−log0.5)/(log2−log0.5),0,1)`：0.5×→0%、1.0×→50%、2.0×→100% | 只要不比基线更差就有分，追平基线拿一半 |
| **C1 Agent** | `GM=(∏ r_i)^(1/10)`，≥1.25× 满 8 分；闭环 2 分 | Agent 只需在自己默认输出上再优化 25% |
| **C2 固定平台** | 设备数 1 / 显存 **64MiB** / DMA 通道 2 / 最大线程/块 **1024** | 与 `aec_device_abi.h` 完全一致，可硬编码这些常量做校验 |
| **C2 Agent 拆分** | 20 分 = DMA 策略 10 + 内核镜像选择 10 | |
| **⚠️ C3.5 口径冲突** | 增强文档写"时间25+显存15"；公开 `scoring.md` 写"时间25+显存10+精度门槛15分" | **以官网公开 scoring.md 为准**，存疑向组委会确认 |

---

## 1. 实际交付物 vs 需自建（务必先看）

对比过 spec 描述和仓库里真正 ship 的文件，三题"起点"差异极大：

### C1 —— 从零自建，正确性验证依赖自建 oracle
- ✅ 给了：5 个公开 `.ptx`、ISA 文档（`docs/03`＋`Track-B/spec.md`）、ISA 编码金标准向量 `starter-kit/golden/b_isa_public.json`
- ✅ 设计目标：自建本地 oracle（`C1/sim/`：AEC 功能模拟器 + numpy 参考对拍），填补正确性验证缺口
- ❌ 没给：**golden model、cycle model、参考 aec-cc**
- ❗ 输入是**真实 NVIDIA PTX**（非 spec.md 示例简化语法）
- 当前状态见 [`docs/C1-完成度审计.md`](../docs/C1-完成度审计.md)（工程状态唯一事实源，本文不再重复枚举）

### C2 —— 高度脚手架化，**本地可自测打分（最高性价比）**
- ✅ 给了：`lib/libaec_device.so`（真正算数的受控设备）、全套头文件、`src/aec_runtime.cpp`（**只实现了设备查询+错误处理，其余全 `return AEC_ERROR_NOT_SUPPORTED`**）、评分脚本 `grader/public_grade.py`、公开测试 `cases/test_r101~r402.py`
- 你要做：把 stub 接到设备 ABI —— `aecAlloc→aecDeviceAlloc`、拷贝/launch→填 `aecDeviceCommand`→`aecDeviceSubmit`、GEMM→`aecDeviceResolveKernel`+submit
- 自测：`make -j2 && python3 grader/public_grade.py --submission . --profile public`
- **建议 Day1 先把 C2 打通拿稳分**

### C3 —— C3.1+C3.5(60分) 自包含可自测；C3.2/C3.3 框架**未随包发布**
- ✅ 给了：3 个 ONNX 模型 + golden logits + labels + thresholds（`testcases/release_to_competitors/`）
- ❌ 没给：C3.2/C3.3 评测脚本 `benchmarks/c32_c33/bench_c32_c33.py` **和它 import 的整套框架**
- ❗ 评测会 import 这些**指定 API**（必须按契约实现同名包）：
  - `import_onnx_graph(model.onnx)`
  - `strategy.select_precision(node, graph)` / `strategy.decompose(...)` / `strategy.tune_kernel(...)`（tune 必须填全 `block_x/grid_x/smem_bytes`）
  - `hardware.supported_precisions()` / `hardware.smem_bytes` / `hardware.max_threads_per_block`
  - `GraphPassPipeline(enable_fusion=True)`，从 `pass_results['Fusion']['stats']['fusion_log']` 读命中
  - `scheduler/graph_passes/fusion.py`
- **待确认**：eval 时框架是否会提供，还是要我们建。默认按"要自建同名包"准备

---

## 2. 知识库（按子题，均为可合法使用的开源/公开资料，须在原创性声明中披露）

### 通用 / 竞赛背景
- **`simple-gpgpu` 仓库检索不到** → 判断为组委会私有 ISA 基线，拿不到；ISA 一切以 `golden/b_isa_public.json` + `docs/03_AEC_ISA规范.md` + `Track-B/spec.md` 为准。
- A4S 暑期学校确为真实项目（深圳河套/上海AI Lab/PKU/Fudan/ICT），但竞赛本身无公开题解知识库。

### C1 编译器
| 用途 | 资源 |
|---|---|
| PTX 语法参考（写解析器） | NVIDIA PTX ISA 官方文档；[PTX 维基](https://en.wikipedia.org/wiki/Parallel_Thread_Execution)；[NVIDIA 博客: Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) |
| 编译器后端整体 | [LLVM NVPTX Backend 指南](https://prereleases.llvm.org/18.1.0/rc3/docs/NVPTXUsage.html)（**参考架构即可，别真上 LLVM**，opcode 子集很小手写更快） |
| 寄存器分配（256 GPR） | **Poletto & Sarkar 线性扫描**（[UCLA PDF](http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf)，[SFU 讲义](https://anoopsarkar.github.io/compilers-class/assets/lectures/opt3-regalloc-linearscan.pdf)，[Max Bernstein: SSA 上的线性扫描带代码](https://bernsteinbear.com/blog/linear-scan/)） |
| 指令调度 / 双发射 | List Scheduling + DDG（数据依赖图）；[组合式寄存器分配+调度综述](https://arxiv.org/pdf/1409.7628) |
| 参考 GPU 模拟器（周期模型思路） | [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution)、[Vortex RISC-V GPGPU](https://github.com/vortexgpgpu/vortex)、[tiny-gpu](https://github.com/adam-maj/tiny-gpu)（理解 warp/lane/调度延迟，用于自建 cycle 估计） |

### C2 Runtime（外部依赖最少，主要吃 starter-kit 自带资料）
| 用途 | 资源 |
|---|---|
| API 心智模型 | CUDA Runtime API（`cudaMalloc/Memcpy/StreamCreate/EventRecord` 与 `aec*` 一一对应）；cuBLAS 的 GEMM 接口形状对照 10 个 `aecMatmul*` |
| 权威依据 | starter-kit `include/*.h` + `docs/02_Runtime与设备规范.md` + `golden/b_isa_public.json`（`examples/02_isa_encoding.c` 可本地验编码） |
| 核心做法 | 阅读 `aec_device_abi.h`：`aecDeviceSubmit`（提交命令）、`aecDeviceResolveKernel`（解析固定镜像）、`aecDeviceEvaluateKernel`（只读周期 oracle，供 kernel_agent 打分选镜像） |

### C3 调度器（可复用资料最丰富）
| 用途 | 资源 |
|---|---|
| **C3.1 图解析** | **`onnx` Python 包**：`onnx.load` + `graph.node/initializer/input/output` 直接产 DAG JSON。[Python API 概览](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)、[ONNX with Python](https://onnx.ai/onnx/intro/python.html) |
| **C3.5 正确性 oracle** | **`onnx.reference.ReferenceEvaluator`**（纯 Python 参考运行时，逐算子语义对照，开发期做 golden 自测）；[ONNX 概念](https://onnx.ai/onnx/intro/concepts.html) |
| **C3.5 GPU 推理** | PyTorch（最省事过 98%/85% 精度门禁，可 onnx2torch 导权重）；**CuPy**（NumPy 兼容 GPU + 自定义 CUDA kernel + cuBLAS，兼顾速度/显存排名）[cupy.dev](https://cupy.dev/) |
| 自定义 kernel/tiling（C3.2 D2/D4/D5） | [CuPy 自定义 kernel + tiling 实战](https://medium.com/@ThinkingLoop/7-numba-cupy-boosts-that-give-gpus-to-plain-python-bb6b931e0cc9)、[GPU 计算实战(CuPy+streams)](https://www.marktechpost.com/2026/05/14/a-coding-implementation-to-master-gpu-computing-with-cupy-custom-cuda-kernels-streams-sparse-matrices-and-profiling/) |
| 卷积策略（im2col + Winograd，C3.2 需两者都被选过） | [Im2col-Winograd 融合卷积论文](https://dl.acm.org/doi/fullHtml/10.1145/3673038.3673039)、[GPU 批量 Winograd 优化](https://www.researchgate.net/publication/339371000) |
| 算子融合（C3.3 五模式） | [图融合科普(BN 折叠/逐元素链)](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/)、[TVM Relay 论文](https://arxiv.org/pdf/1904.08368)、[TVM 算子融合优化](https://www.researchgate.net/publication/402276102) |
| numpy 版算子参考 | [onnxruntime-numpy](https://github.com/gf712/onnxruntime-numpy)、[ONNX Operators 规范](https://onnx.ai/onnx/operators/) |

---

## 3. 三天节奏（3 人各一题）

| | C2（C++，稳分优先） | C3（Python，分最多） | C1（C++，最难先起手） |
|--|--|--|--|
| **Day1** | R101–R106 全过（内存/拷贝/stream/event/launch）→ 30 分到手 | `onnx` 出 C3.1 DAG JSON；PyTorch/CuPy 跑通 MLP 过门禁 | 前端解析 + 基础 lowering（历史赛题需求；当前状态见 [`docs/C1-完成度审计.md`](../docs/C1-完成度审计.md)） |
| **Day2** | GEMM 10 dtype + AXPY/DOT/NRM2 + 双DMA/注册内存/故障恢复 → 冲 Good | 打通 ResNet-18 + Transformer 全过门禁（60分）；搭 `scheduler` 框架 API | 标量优化 + 寄存器分配（历史赛题需求；当前状态见 [`docs/C1-完成度审计.md`](../docs/C1-完成度审计.md)） |
| **Day3** | 两个 Agent（DMA+kernel 选镜像）冲 Excellent + 全量 grader | C3.2/3.3 五融合模式 + C3.4 内存规划（要经得起 code review）+ 时间/显存调优 | 指令调度 + GEMM lowering + Agent（历史赛题需求；当前状态见 [`docs/C1-完成度审计.md`](../docs/C1-完成度审计.md)） |

人手不足时优先级：**C2 全量 > C3.1+C3.5(60) > C3.2/3.3/3.4**。C1 优先级及计划见 [`docs/C1-完成度审计.md`](../docs/C1-完成度审计.md)。

---

## 4. 待向组委会确认的问题
1. C3.2/C3.3 的 `bench_c32_c33.py` 及其 import 的 scheduler 框架，eval 时是否提供？还是选手按 API 契约自建同名包？
2. C3.5 分值口径：显存 10 还是 15？精度门槛是否计入 15 分？
3. ~~C1 是否会提供 golden/cycle model 用于选手自测，还是只在评测端？~~ **[已确认不提供]** → 自建 sim/oracle 已填补正确性验证缺口；若后续发布二进制参考 C-model 再校准。
4. 跨赛道 ISA：C1 需要的 TMUL 等指令，编码是否完全等同 `b_isa_public.json` 定义？

---

## 附：检索来源
- 竞赛/基线：[A4S 暑期学校介绍](https://www.slai.edu.cn/zh-hans/article/502)、[GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution)、[Vortex](https://github.com/vortexgpgpu/vortex)、[tiny-gpu](https://github.com/adam-maj/tiny-gpu)
- C1：[PTX 维基](https://en.wikipedia.org/wiki/Parallel_Thread_Execution)、[LLVM NVPTX](https://prereleases.llvm.org/18.1.0/rc3/docs/NVPTXUsage.html)、[线性扫描 Poletto](http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf)、[线性扫描讲义](https://anoopsarkar.github.io/compilers-class/assets/lectures/opt3-regalloc-linearscan.pdf)、[SSA 线性扫描博客](https://bernsteinbear.com/blog/linear-scan/)、[寄存器分配+调度综述](https://arxiv.org/pdf/1409.7628)
- C3：[ONNX Python API](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)、[ONNX with Python](https://onnx.ai/onnx/intro/python.html)、[CuPy](https://cupy.dev/)、[Im2col-Winograd](https://dl.acm.org/doi/fullHtml/10.1145/3673038.3673039)、[图融合科普](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/)、[TVM Relay](https://arxiv.org/pdf/1904.08368)、[onnxruntime-numpy](https://github.com/gf712/onnxruntime-numpy)
