# Runtime、Device 与 Kernel ABI 规范

公共布局和导出符号以 `include/` 中的三个头文件为准。本文说明可观察行为。

## 1. 设备与错误

- `aecDeviceCount` 返回 1。
- device 0 名称为 `AEC Deterministic Virtual Device`。
- 设备内存 64 MiB，DMA 通道 2，max threads/block 1024。
- Runtime ABI 2，AEC ISA 2/profile 1，参数块最大 64 字节。

错误状态按 host thread 保存。失败调用更新 last error，成功调用不清除旧错误。

- `aecPeekAtLastError`：读取但不清除。
- `aecGetLastError`：读取后清为 `AEC_SUCCESS`。
- `aecGetErrorName`：已知枚举返回稳定名称，未知值返回 `AEC_ERROR_UNKNOWN`。

异步错误属于对应 Stream。`aecStreamSync` 返回并清除第一个未报告错误。ISA trap 映射为
`AEC_ERROR_ISA_TRAP`。

## 2. Device allocation

- device pointer 是 opaque 64-bit offset，0 非法。
- allocation 按 64 字节对齐，使用 deterministic lowest-address first-fit。
- free block 合并；中间没有其他 allocation 时，同尺寸重分配返回同一 offset。
- 前 64 字节保留，因此申请完整 64 MiB 必须 OOM。
- bounds 必须相对于单个 live allocation，不能只检查全局地址范围。

`aecFree` 等待此前 enqueue 的工作。zero、unknown、interior、stale 和 double-free 均失败。

## 3. Copy

同步复制在返回前完成。以下输入失败：

- host pointer 为 null；
- bytes 为 0；
- device span 不完整属于一个 live allocation；
- offset/size 加法 overflow。

`aecCopyAsync` 要求有效 Stream，enqueue 后即可返回。H2D source 在完成前必须保持存活且
不变；D2H destination 在完成前必须保持可写。

## 4. Stream 与 Event

同一 Stream 严格 FIFO；并发 enqueue 按 Runtime sequence 排序。不同 Stream 不提供隐式
顺序。destroy 先将 handle 从 live registry 移除，再等待队列和 worker；后续使用必须返回
invalid handle。

Event record 在 Stream 当前队尾插入 marker。重复 record 产生新 generation，query、sync、
destroy 和 elapsed 都观察最新 generation。

- 未 record Event 的 query/sync 返回 invalid argument。
- 未完成的 query 返回 `AEC_ERROR_NOT_READY`。
- destroy 已 record Event 时等待最新 generation。
- elapsed 要求两个 Event 已完成，且 end cycle 不小于 start cycle。

## 5. Host registration

`aecHostRegister` 注册一个精确、非空 interval。duplicate、overlap 和 overflow 均失败。
完整位于一个 registered interval 内的 transfer 使用 REGISTERED 和 ZERO_COPY flags。
registration 只改变虚拟周期，不改变数据结果。

`aecHostUnregister` 要求精确 base pointer，并等待此前使用该 interval 的工作。

## 6. Runtime statistics

统计镜像受控设备 counters，包括：

- submitted、DMA、kernel、zero-copy 和 channel counts；
- total/last virtual cycles；
- ISA launches、instructions retired、traps；
- last kernel handle 和 last trace digest。

`aecResetRuntimeStats` 只清统计，不重置 allocation、registration、sequence、handle 或 image
registry。

## 7. Launch

`aecLaunch` 只接受公共 Kernel ID 和精确参数结构：

- grid/block 各维大于 0；
- block volume 不超过 1024；
- args 非空，`args_size` 必须精确；
- null Stream 表示同步；非空 Stream 复制 args bytes 后 enqueue。

Runtime 必须调用 `aecDeviceResolveKernel`，生成 canonical 参数块，再提交
`AEC_DEVICE_OP_ISA_LAUNCH`。不得从 Kernel ID 推导 handle，也不得把 Kernel ID 当作 opcode
或 code address。

## 8. Device ABI command

每个 command 必须使用 ABI version 2。sequence 非零，并严格大于进程中此前所有已接受的
sequence；completion 回显相同 sequence。

DMA command：

- H2D 的 host address 是 source，`dst` 是 device offset。
- D2H 的 `src` 是 device offset，host address 是 destination。
- bytes/chunk 非零；queue depth 为 1、2、4 或 8；channel 为 0 或 1。
- ZERO_COPY 只能与 REGISTERED 同时使用。

ISA launch command 必须使用已解析的非零 handle，并与 manifest 中的 ISA version、entry、
parameter bytes、dtype 和 variant 一致。unused parameter bytes 必须为 0；grid、block、shape、
alignment 和 tier 必须合法。

preflight 失败时不退休指令，也不修改计算数据。completion 包含 status、cycles、retired、
digest、fault code 和 trap PC。

## 9. Fault 与 reset

fault injection 影响下一条匹配的 DMA、ISA launch 或任意 command，只消费一次。injected
fault 不退休指令；ISA trap 与 injected fault 是不同错误。错误后设备必须能继续处理合法
command。

`aecDeviceReset` 清 `.gmem`、allocation、sequence、fault 和 stats，但保留 image registry。
`aecDeviceResetStats` 只清统计。

## 10. Kernel image

image 格式：

```text
64-byte aecIsaImageHeader
instruction_count × 16-byte aecIsaInstruction
```

header 包含 magic、版本、image ID、semantic Kernel ID、dtype、variant、entry、参数大小、
flags 和 instruction FNV-1a hash。reserved 必须为 0。v2 image 不包含 relocation、import、
symbol 或 writable code。

bundle 中的 34 个 tuple 会与受控设备逐一核对。handle 非零、不可变、仅在当前进程有效，
并在 device/stat reset 后保持稳定。

## 11. Canonical 参数块

所有字段 little-endian、紧密排列、无 native padding。

| Kernel | 大小 | 字段顺序 |
|---|---:|---|
| Vector Add FP32 | 32 | A:u64, B:u64, C:u64, count:u64 |
| GEMM | 40 | A:u64, B:u64, C:u64, M:u32, N:u32, K:u32, dtype:u32 |
| AXPY FP32 | 28 | X:u64, Y:u64, count:u64, alpha:f32 bits |
| DOT FP32 | 32 | X:u64, Y:u64, result:u64, count:u64 |
| NRM2 FP32 | 24 | X:u64, result:u64, count:u64 |

## 12. Tier

| Tier | 必需能力 |
|---|---|
| Basic | query/error、allocation/free、同步 copy、stats、Vector Add、FP32/INT32 GEMM |
| Good | Basic + Stream/Event、async copy、registration、其余 GEMM 和 vector API |
| Excellent | Good + 两个 Agent 和 vectorized image 选择 |

Basic 可以对高等级操作返回 `AEC_ERROR_NOT_SUPPORTED`，但必须导出全部符号。

## 13. API 索引

Runtime API：

| 类别 | API |
|---|---|
| Device/error | `aecDeviceCount`、`aecDeviceInfo`、`aecGetLastError`、`aecPeekAtLastError`、`aecGetErrorName` |
| Memory/copy | `aecAlloc`、`aecFree`、`aecCopyH2D`、`aecCopyD2H`、`aecCopyAsync` |
| Stream | `aecStreamCreate`、`aecStreamDestroy`、`aecStreamSync` |
| Event | `aecEventCreate`、`aecEventDestroy`、`aecEventRecord`、`aecEventSynchronize`、`aecEventQuery`、`aecEventElapsedCycles` |
| Registration/stats | `aecHostRegister`、`aecHostUnregister`、`aecGetRuntimeStats`、`aecResetRuntimeStats` |
| Launch | `aecLaunch` |
| GEMM | `aecMatmulF4`、`aecMatmulF8`、`aecMatmulF16`、`aecMatmulBF16`、`aecMatmulF32`、`aecMatmulF64`、`aecMatmulI4`、`aecMatmulI8`、`aecMatmulI32` |
| Vector | `aecAxpy`、`aecDot`、`aecNrm2` |

Device ABI：`aecDeviceGetCaps`、`aecDeviceReset`、`aecDeviceResetStats`、
`aecDeviceAlloc`、`aecDeviceFree`、`aecDeviceResolveKernel`、`aecDeviceEvaluateKernel`、
`aecDeviceSubmit`、`aecDeviceGetStats`、`aecDeviceInjectFault`。

`aecDeviceEvaluateKernel` 是只读 policy oracle，不提交 command、不修改 stats。Runtime 的
计算路径仍必须使用 resolve + submit。
