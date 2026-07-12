# 评分标准与 Agent

## 1. 总分

总分 100：Runtime 30、计算库 30、虚拟 Driver 20、Agents 20。

除 R401、R402 外，每个 requirement 为 all-or-zero：该项的 mandatory public/hidden 检查
全部通过才得分。计算项还必须提供受控设备上的有效 ISA 执行证据。

| ID | 内容 | 分值 | Gate |
|---|---|---:|---|
| R101 | Device/ISA query、错误名、TLS last error | 4 | Basic |
| R102 | allocation/free、OOM、reuse、非法 free | 6 | Basic |
| R103 | 同步 H2D/D2H 与 allocation bounds | 6 | Basic |
| R104 | Vector Add image 与 launch mapping | 4 | Basic |
| R105 | Stream FIFO 与异步操作 | 5 | Good |
| R106 | Event generation、cycles、异步错误 | 5 | Good |
| R201 | FP32/INT32 GEMM | 10 | Basic |
| R202 | FP4、FP8、FP16、BF16、FP64 GEMM | 10 | Good |
| R203 | INT4/INT8 与 INT32 饱和输出 | 4 | Good |
| R204 | FP32 AXPY、DOT、NRM2 | 6 | Good |
| R301 | ABI sequence、resolve、completion、stats | 6 | Good |
| R302 | 双 DMA、异步边界与恢复 | 6 | Good |
| R303 | host registration 与 zero-copy | 4 | Good |
| R304 | DMA/ISA fault propagation 与恢复 | 4 | Good |
| R401 | DMA policy Agent | 10 | Excellent |
| R402 | Kernel-image policy Agent | 10 | Excellent |

## 2. 等级 gate

- Basic：总分至少 30，且 R101–R104、R201 全部通过。
- Good：总分至少 75，Basic 通过，且 R105、R106、R202–R204、R301–R304 全部通过。
- Excellent：总分至少 90，Good 通过，R401/R402 correctness 通过，且两个 Agent 都有正的
  hidden average speedup。

## 3. DMA Agent

文件：`agents/dma_agent.py`。

输入包含 `case_id`、direction、bytes、alignment、registered 和 concurrency。输出必须只含：

```json
{"channel":0,"chunk_bytes":65536,"queue_depth":2,"use_zero_copy":true}
```

合法值：

- channel：0 或 1
- chunk bytes：4096、65536、1048576
- queue depth：1、2、4、8
- zero-copy：仅 registered range 可使用

DMA 虚拟周期为：

```text
setup + ceil(ceil(bytes / 32) / parallelism)
      + 24 × (ceil(bytes / chunk_bytes) - 1)
      + alignment_penalty
```

其中 registered zero-copy 的 setup 为 45，否则为 100；
`parallelism=min(queue_depth,concurrency,2)`；低于 64-byte alignment 时 penalty 为 13。

## 4. Kernel-image Agent

文件：`agents/kernel_agent.py`。

输入包含 dtype、M/N/K、alignment、workspace 和 candidates。输出必须只含：

```json
{"kernel_id":"<candidate-id>"}
```

- naive：所有合法 shape
- tiled：M/N/K 均可被 4 整除
- vectorized：M/N/K 均可被 8 整除，且 alignment 至少 16 bytes
- 选择结果还必须满足 workspace 和 candidate 自身约束

Kernel 周期来自实际 AEC image interpretation，不使用 grader 侧估算公式。

## 5. Agent 运行协议

- 每次调用从 stdin 读取一个 JSON，只向 stdout 输出一个 JSON。
- 输出不能带额外字段或日志。
- 单次超时 1 秒，stdout 与 stderr 合计不超过 64 KiB。
- 不得访问网络、评分器文件或在 case 间保存状态。
- 输入输出结构以 `schemas/` 为准。

## 6. Agent 分数

每个 Agent 的 10 分由两部分组成：

- 4 分：全部 case 的 JSON、合法性与 correctness
- 6 分：hidden performance

```text
fraction = clamp((baseline / candidate - 1) / 0.50, 0, 1)
points = 4 + 6 × average(hidden performance fractions)
```

50% speedup 对应满 performance 分。公开性能仅用于诊断，不计入最终 performance 分。
两个 Agent 都合法但没有隐藏加速时，总分上限为 88，等级为 Good。

## 7. 公开与隐藏测试

公开测试给出正常路径和代表性错误，便于定位问题。最终评分会换用不同输入，并增加边界、
生命周期、并发、故障、ISA trap、参数编码和 Agent 泛化检查。

报告包含总分、等级、gate、逐 requirement 结果、Agent 周期与受控设备统计。性能分只使用
虚拟周期，不使用 wall-clock。

## 8. 最终评分流程

1. 主办方按提交截止时保存的 artifact 计算 SHA-256，并只评分该 artifact。
2. 每个 requirement 在新的安全 worker 中执行；不同 requirement 不共享进程状态。
3. 公共和隐藏步骤使用相同公共契约。非 Agent requirement 任一步失败，该项得 0 分。
4. 计算项同时核对独立数值 oracle 和受控设备的 image handle、retired count、trace digest。
5. 汇总 requirement 分数后应用 Basic/Good/Excellent gate，生成最终 JSON 报告。
6. 只有平台故障、grader 缺陷或规范歧义触发复评；确认缺陷后对全部受影响提交统一重跑。

参赛者可见报告包含总分、等级、逐项得分和不泄露隐藏输入的错误类别。隐藏输入、其他队伍
提交和主办方内部 oracle 不公开。
