# A4S_C · 赛道 C（编译器 / Runtime / 算子调度）

> AEC GPGPU 软件栈：PTX 风格 IR → 主机侧 Runtime → ONNX 端到端推理部署。
> 三子题独立评分（各 100 分），等权平均为赛道总分（满分 100）。

```text
C1 编译器   : PTX 风格 IR  ──►  AEC ISA 机器码
C2 Runtime : 主机侧 libaec.so + 虚拟 GPGPU 设备
C3 算子调度 : ONNX 模型    ──►  AEC GPGPU 推理
```

## 子题概览

| 子题 | 主题 | 关键内容 | 分值构成 |
|------|------|----------|----------|
| **C1** AECIR 编译器 | PTX 风格 IR → AEC ISA 机器码 | IR/SSA/CFG、必需优化 pass（常量传播/DCE/CSE/LICM/内存合并/谓词）、寄存器分配（256 GPR）、指令调度与双发射、多精度 GEMM | 正确性 50 + 性能 35 + 鲁棒 5 + Agent 10 |
| **C2** 主机侧驱动 libaec.so | 内存/Kernel/Stream/Event、10 种 GEMM、虚拟设备驱动 | Runtime API、计算库（10 dtype + 向量运算）、双 DMA + 注册内存 + 故障恢复、DMA/Kernel Agent | Runtime 30 + 计算库 30 + 驱动 20 + Agent 20 |
| **C3** 算子调度 | ONNX 模型 → AEC GPGPU 推理 | 计算图解析、算子分解、算子融合（5 模式）、内存规划、端到端部署（MLP/ResNet-18/Transformer） | 图解析 10 + 分解 15 + 融合 15 + 内存 10 + 端到端 50 |

## 文档索引

| 文档 | 用途 |
|------|------|
| [`docs/Track-C-Reading-Notes.md`](./docs/Track-C-Reading-Notes.md) | 主体笔记：C1/C2/C3 摘要 + §6 联网调研背景（PTX / ONNX / CUDA / GEMM+Tensor Core / 算子融合 / 内存规划 / MLIR / SSA） |
| [`docs/赛道C-知识库与作战手册.md`](./docs/赛道C-知识库与作战手册.md) | 知识库 + 赛题解析对齐：实际交付物核对、可复用资料清单、三天节奏、待向组委会确认的问题 |
| [`public/Track-C/README.md`](./public/Track-C/README.md) | 官方赛题入口 |
| `public/Track-C/C1-compiler/{spec,scoring}.md` | C1 官方赛题与评分 |
| `public/Track-C/C2-runtime/{spec,scoring}.md` | C2 官方赛题与评分 |
| `public/Track-C/C2-runtime/starter-kit/` | C2 完整 starter kit（头文件、34 image、文档） |
| `public/Track-C/C3-scheduler/{spec,scoring}.md` | C3 官方赛题与评分 |

## 推荐阅读顺序

1. 先读本文「子题概览」与官方 `public/Track-C/*/spec.md`、`scoring.md`——熟悉题型与评分构成。
2. 再读 [`docs/赛道C-知识库与作战手册.md`](./docs/赛道C-知识库与作战手册.md) §1，搞清「实际给了什么 / 要自建什么」。
3. 最后按子题精读 [`docs/Track-C-Reading-Notes.md`](./docs/Track-C-Reading-Notes.md)（§1–§3 摘要 + §6 配套背景，每节标注来源 URL 与对应子题）。
