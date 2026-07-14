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
| [`docs/术语表.md`](./docs/术语表.md) | **专业词汇通俗解释**（编译器/GPU/ONNX/精度/框架…），读文档卡词时先查这里 |
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

## 远程执行工具

`scripts/remote_exec.py` 在 mig02 服务器（Linux, gcc 13.3）上执行命令，供远程构建/测试使用。
连接配置固定（`mig02@39.107.68.147:1102`，私钥 `./mig02`）；下方打包/校验脚本复用其配置。
服务器公网流量有限——禁止经其上传/下载大文件、禁止隧道/端口转发。

## 提交打包与校验（pack / verify）

`scripts/pack_trackc.py` 与 `scripts/verify_trackc.py` 负责把 C1/C2/C3 装配成官方要求的提交压缩包并校验。
仅依赖 Python 标准库，跨平台（在 Windows 开发机即可运行；`libaec.so` 构建与运行时校验在 mig02 上完成）。

### 产出的提交结构

```text
TrackC-<编号1姓名1>-<编号2姓名2>-<编号3姓名3>.zip
└── TrackC-<编号1姓名1>-<编号2姓名2>-<编号3姓名3>/
    ├── C1/compiler/{aec-cc, src/{Makefile,src,include,tools}}   # 入口 wrapper + 源码文件夹
    ├── C2/{libaec.so, lib/libaec_device.so, agents/}            # Runtime + 依赖 + 可选 Agent
    └── C3/{src/{scheduler,runtime,tools,benchmarks}, requirements.txt, readme.md}
```

关键处理（开发仓库本身不动，全部发生在打包暂存副本里）：

- **C1**：wrapper `compiler/aec-cc` 的 `root` 改写为 `./src`，源码收入 `compiler/src/`；评测机首次调用自动 `make build`（build-on-first-use）。
- **C2**：`libaec.so` 缺失时**自动在 mig02 远程构建并回传**（仅传 KB 级源码 + 回传 .so）；`libaec_device.so` 作为 rpath 依赖一并附带（公共资产，原样）。
- **C3**：框架源码收入 `C3/src/`（Q&A 要求）；`README.md` → 小写 `readme.md`，命令为 `python src/tools/...`（评测以 `C3/` 为工作目录，脚本内部 `sys.path` 自注入 `src/` 使 `from scheduler import` 生效）。

### 前置条件

- 私钥 `./mig02` 就绪（远程构建 `libaec.so` 与 `--remote` 校验都需要；`remote_exec.py` 会自动从 `~/.ssh/mig02` 复制并 `chmod 600`）。
- 本机有 `ssh`（Windows 用 OpenSSH 或 msys2 的 ssh 均可；脚本经 tar/ssh 管道传文件，不依赖 scp）。

### 快速开始

```bash
# 1) 打包（默认占位成员；libaec.so 缺失会自动远程构建）
py -3.13 scripts/pack_trackc.py

# 2) 真实成员信息打包
py -3.13 scripts/pack_trackc.py --members "20260001张三,20260002李四,20260003王五"

# 3) 静态校验（本地、快、无网络）
py -3.13 scripts/verify_trackc.py --zip TrackC-20260001张三-20260002李四-20260003王五.zip

# 4) 完整校验：静态 + 远程 Linux 运行时（与评测同构，正式提交前必跑一次）
py -3.13 scripts/verify_trackc.py --zip TrackC-20260001张三-20260002李四-20260003王五.zip --remote
```

### 两层校验

| 层 | 触发 | 内容 |
|----|------|------|
| **Layer A**（静态，本地） | 默认即跑 | zip 命名 / 结构 / 路径 / 可执行位 / ELF magic / readme / 体积 / 洁净度，无需解压 |
| **Layer B**（运行时，mig02） | `--remote` | 上传 zip → 解压 → C1 `aec-cc --selftest`（build-on-first-use）、C2 `dlopen libaec.so`、C3.1 `export_dag`（无公共模型时自动生成探针模型）、C3.5 worker `READY` |

任一硬性 FAIL 退出码非 0；Layer A 有硬性 FAIL 时自动跳过 Layer B。可选公共资产（官方 grader、onnx 模型）在 mig02 上缺失记为 WARN，不计入失败。

### 常用参数

| 脚本 | 参数 | 说明 |
|------|------|------|
| pack | `--members` | 三位成员 `编号1姓名1,编号2姓名2,编号3姓名3`（默认占位 `00000000成员1,...`） |
| pack | `--out` | 输出目录（默认仓库根） |
| pack | `--build-c2 {auto,remote,wsl,none}` | `libaec.so` 构建方式（默认 `auto`：先 remote 再 wsl） |
| verify | `--zip` | 待校验 zip（缺省自动找仓库内最新 `TrackC-*.zip`） |
| verify | `--remote` | 追加 Layer B 远程运行时校验 |
| verify | `--remote-public` | 服务器公共资料路径（默认 `~/A4S/public`） |

> 备注：远程构建回传的 `C2/libaec.so` 是构建产物，落在工作区——是否提交请按本地策略（如需忽略可加入 `.gitignore`）。GitHub Private Repo 上传与赛道邮箱（`zdhuang24@m.fudan.edu.cn`）提交为后续手动步骤，脚本只负责产出并验证合规 zip。

## C1 状态快照

当前实现状态、验证数字、完成度、风险、优先级等全部维护在 [`docs/C1-完成度审计.md`](./docs/C1-完成度审计.md)（工程状态唯一事实源）。本文不再重复维护可漂移的状态断言。

**稳定测试入口**：
```bash
cd C1 && make selftest && make test
cd C1/sim && py -3.13 dogfood.py && py -3.13 bench.py
```
