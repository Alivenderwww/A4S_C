# 提交与公开测试指南

## 1. 构建

需要 GNU Make、C++17 编译器和 Python 3.10+。

```bash
make -j2
make examples
```

起始 Runtime 可编译，但除 query/error/stats 外的大部分接口会返回
`AEC_ERROR_NOT_SUPPORTED`。可以修改或拆分 `src/` 中的实现并调整自己的构建规则；
`include/`、设备库、Kernel bundle 和 grader 是只读评分契约。

## 2. 建议实现顺序

1. TLS last error、device query，并运行 ISA golden encoding 示例
2. allocation/free、同步复制、allocation-relative bounds
3. Vector Add 的 resolve、参数序列化和 ISA launch
4. Stream FIFO、Event generation、异步错误
5. FP32/INT32 GEMM，再实现其他 dtype
6. registration、zero-copy 和双 DMA
7. AXPY、DOT、NRM2
8. 两个 Agent

## 3. 示例

| 程序 | 内容 |
|---|---|
| `01_device_query` | 设备、ABI、ISA |
| `02_isa_encoding` | 128-bit 指令编码 |
| `03_vector_add` | alloc/copy/launch |
| `04_stream_event` | Stream FIFO 与 Event |
| `05_fp32_gemm` | FP32 GEMM |
| `06_registered_copy` | registration 与 zero-copy |

```bash
./bin/01_device_query
./bin/02_isa_encoding
./bin/03_vector_add
./bin/04_stream_event
./bin/05_fp32_gemm
./bin/06_registered_copy
```

## 4. 公开评分

```bash
python3 grader/public_grade.py \
  --submission . \
  --profile public \
  --json-out public-report.json
```

公开 grader 不支持 full profile。Agent 的公开性能只用于诊断。

运行单项或全部 requirement：

```bash
python3 cases/test_r101.py --submission .
python3 cases/test_r201.py --submission .
make public-cases
```

## 5. 公开测试索引

| ID | 主要检查 |
|---|---|
| R101 | metadata、error、TLS |
| R102 | allocation reuse、OOM、double free |
| R103 | 同步复制与 span 错误 |
| R104 | Vector Add image 与 launch |
| R105 | Stream FIFO |
| R106 | Event、cycles、异步错误 |
| R201 | FP32/INT32 GEMM |
| R202 | 浮点存储格式 |
| R203 | INT4/INT8 packed 输入 |
| R204 | AXPY、DOT、NRM2 |
| R301 | command 与 stats accounting |
| R302 | 双通道和异步恢复 |
| R303 | registration 与 zero-copy |
| R304 | fault propagation 与恢复 |
| R401 | DMA Agent JSON 与公开性能 |
| R402 | Kernel Agent 合法性与公开性能 |

机器可读定义位于 `cases/manifest.json` 和 `cases/public_cases.json`。

## 6. 常见问题

- 结果正确但计算 requirement 失败：确认实际提交了正确 AEC image，并产生 retired/digest。
- 参数错误：不要直接复制 native struct；按规定 offset 写 little-endian 参数块。
- 边界错误：device span 必须完整属于一个 live allocation。
- 异步结果不稳定：同一 Stream 必须 FIFO，异步 launch 必须复制参数内容。
- Event 读到旧结果：query/sync 应观察最新 record generation。
- vectorized candidate 被拒绝：检查 shape divisibility、alignment 和 workspace。
- Agent 输出被拒绝：stdout 只能包含一个 JSON，不能打印调试日志。

## 7. 提交前检查

```bash
nm -D --defined-only libaec.so
python3 grader/public_grade.py --submission . --profile public --quiet
```

确认全部 Runtime 符号已导出，提交不依赖非公开路径，且没有修改公共契约、设备或 image。

## 8. 提交目录

```text
submission/
├── libaec.so
└── agents/
    ├── dma_agent.py
    └── kernel_agent.py
```

Basic/Good 不要求 Agent 性能，但 `libaec.so` 仍须导出全部 Runtime 符号。
两个 Agent 文件可以省略；省略时 R401/R402 得 0 分，不影响 Basic/Good gate。

主办方只读取上述三个文件。提交中不得包含 symlink、device file、setup script，且不得依赖
赛事环境未声明的第三方动态库、绝对路径或网络服务。压缩包格式、大小上限、提交次数和
截止时间以赛事平台公告为准。
