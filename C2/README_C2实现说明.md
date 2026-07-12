# 赛道 C - C2 主机运行时实现说明（libaec.so）

本文档说明 `C2/` 目录下**主机侧运行时** `libaec.so` 的实现方式、构建/自测流程、
每个 requirement 的完成状态与后续 TODO 优先级。

- 唯一需要我们实现的源文件：`src/aec_runtime.cpp`
- 只读评分契约（**不得修改**）：`include/`、`lib/libaec_device.so`、`kernels/`、
  `cases/`、`grader/`、以及起步包自带的 `README.md`
- 冲击 Excellent 时的两个策略 Agent：`agents/dma_agent.py`、`agents/kernel_agent.py`
  （当前为合法 baseline stub）

---

## 0. ⚠️ 重要：Windows 本机无法构建

`lib/libaec_device.so` 是 **Linux ELF 共享库**。`libaec.so` 必须链接并在运行期由
`ctypes` 与它一起加载（见 `grader/public_grade.py` 的 `Runtime.__init__`）。因此：

> **在 Windows 本机无法 `make`、无法运行 grader。必须在 WSL 或 Linux 环境构建与自测。**

Windows 端只能做**语法自检**（不链接设备库）：

```bash
# 仅语法检查，不产出可用产物；真正评分请在 WSL/Linux 上做
g++ -fsyntax-only -std=c++17 -Iinclude src/aec_runtime.cpp
```

本实现已通过语法自检（唯一的 C++17 特性是继承自起步包的无消息 `static_assert`，
在 `Makefile` 的 `-std=c++17` 下合法）。

---

## 1. 构建与自测（WSL / Linux）

```bash
cd C2
make -j2                 # 编译 libaec.so（链接 lib/libaec_device.so）
make examples            # 编译 bin/01_..bin/06_ 六个示例
./bin/01_device_query    # 冒烟测试：打印设备名/ABI/ISA/内存

# 公开评分（诊断用；不支持 full profile）
python3 grader/public_grade.py --submission . --profile public

# 单个 requirement 自测
python3 cases/test_r101.py --submission .
python3 cases/test_r201.py --submission .
# 或全部：
make public-cases        # == python3 cases/run_all.py --submission .

# 提交前确认符号齐全
nm -D --defined-only libaec.so
```

`Makefile` 会用 `-Wl,-rpath,'$ORIGIN/lib'` 把 `libaec.so` 指向同目录的
`lib/libaec_device.so`，不要移动该库。

---

## 2. 核心机制：设备 ABI 转发模式

运行时**不做任何主机侧数值计算**。它只是把公共 Runtime API 翻译成设备 ABI
（`include/aec_device_abi.h`）调用：

| Runtime 概念 | 设备 ABI 调用 |
|---|---|
| `aecAlloc` / `aecFree` | `aecDeviceAlloc(bytes, 64, &ptr)` / `aecDeviceFree(ptr)` |
| 同步/异步拷贝 | 构造 `aecDeviceCommand{opcode=OP_H2D/OP_D2H}` → `aecDeviceSubmit` |
| `aecLaunch` / GEMM / 向量库 | `aecDeviceResolveKernel` → 打包参数块 → `OP_ISA_LAUNCH` → `aecDeviceSubmit` |
| `aecGetRuntimeStats` | `aecDeviceGetStats`（逐字段镜像） |

关键不变量（`docs/02` 第 8 节）：

- 每条命令 `abi_version = 2`；`sequence` 非零且**进程内严格递增**
  （实现用单个 `std::atomic<uint64_t> g_sequence` 全局计数器保证）。
- 同步命令直接返回错误；异步（带 Stream）命令把错误**暂存到 Stream**，由
  `aecStreamSync` 返回并清除（"异步错误属于 Stream"）。
- 提交后**同时**检查 `aecDeviceSubmit` 的返回码（preflight 失败）和
  `completion.status`（执行期故障 / ISA trap）。

### 设备状态 → Runtime 错误映射（`from_device()`）

| 设备状态 | Runtime 错误 |
|---|---|
| `AEC_DEVICE_SUCCESS` | `AEC_SUCCESS` |
| `AEC_DEVICE_INVALID_ARGUMENT` | `AEC_ERROR_INVALID_ARGUMENT` |
| `AEC_DEVICE_OUT_OF_MEMORY` | `AEC_ERROR_OUT_OF_MEMORY` |
| `AEC_DEVICE_INVALID_ADDRESS` | `AEC_ERROR_INVALID_ADDRESS` |
| `AEC_DEVICE_UNSUPPORTED` | `AEC_ERROR_NOT_SUPPORTED` |
| `AEC_DEVICE_INJECTED_FAULT` | `AEC_ERROR_DEVICE` |
| `AEC_DEVICE_ISA_TRAP` | `AEC_ERROR_ISA_TRAP` |
| `AEC_DEVICE_INTERNAL` | `AEC_ERROR_INTERNAL` |

### H2D 拷贝完整代码范例

以下是 `aecCopyH2D` 的实际实现路径（构造命令 → 提交 → 检查 completion）：

```cpp
// 1) 构造 DMA 命令：H2D 时 host 地址是 source，device offset 是 dst
aecDeviceCommand make_dma(uint16_t opcode, aecDevicePtr device,
                          uint64_t host_address, uint64_t bytes, uint16_t flags) {
    aecDeviceCommand command{};                 // {} 值初始化：所有保留字段清零
    command.opcode = opcode;                    // OP_H2D = 1
    command.flags  = flags;                     // registered range 命中则 REGISTERED|ZERO_COPY
    command.dst    = device;                    // device offset（H2D 目标）
    command.host_address = host_address;        // 主机源指针
    command.bytes  = bytes;
    uint64_t chunk = bytes > (1u << 20) ? (1u << 20) : bytes;
    command.chunk_bytes = (uint32_t)chunk;      // 必须非零
    command.queue_depth = 1;                    // 合法值 1/2/4/8
    command.channel     = 0;                    // 合法值 0/1
    return command;
}

// 2) 填 abi_version + sequence，提交，检查两处状态
aecError_t submit_command(aecDeviceCommand &command, uint64_t *out_cycles) {
    command.abi_version = AEC_DEVICE_ABI_VERSION;                 // 2
    command.sequence    = g_sequence.fetch_add(1);               // 非零、严格递增
    aecDeviceCompletion completion{};
    const aecDeviceStatus rc = aecDeviceSubmit(&command, &completion);
    if (out_cycles) *out_cycles = completion.virtual_cycles;
    if (rc != AEC_DEVICE_SUCCESS) return from_device(rc);        // preflight 失败
    if (completion.status != AEC_DEVICE_SUCCESS)                 // 执行期故障 / trap
        return from_device((aecDeviceStatus)completion.status);
    return AEC_SUCCESS;
}

// 3) 对外接口：仅做参数校验后转发
aecError_t aecCopyH2D(aecDevicePtr dst, const void *src, size_t bytes) {
    if (src == nullptr)   return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (bytes == 0)       return finish(AEC_ERROR_INVALID_ARGUMENT);
    const uint64_t host = reinterpret_cast<uint64_t>(src);
    const uint16_t flags = host_span_registered(host, bytes)
                         ? (AEC_DEVICE_FLAG_REGISTERED | AEC_DEVICE_FLAG_ZERO_COPY)
                         : AEC_DEVICE_FLAG_NONE;
    aecDeviceCommand cmd = make_dma(AEC_DEVICE_OP_H2D, dst, host, bytes, flags);
    return dispatch(cmd, /*stream=*/nullptr);   // 同步：直接返回映射后的错误
}
```

> **allocation 越界不在运行时侧检查**：设备是 span/bounds 的唯一权威
> （`docs/02` 第 2/3 节）。运行时不维护 `ptr→size` 影子表，以免与设备判定分歧。
> 越界拷贝由设备 preflight 返回 `INVALID_ADDRESS`，运行时只做映射。

---

## 3. Launch 参数块打包（little-endian，紧密无 padding）

`aecLaunch` 与计算接口统一走 `launch_kernel()`：`resolve → 打包 → OP_ISA_LAUNCH`。
参数块按 `docs/02` 第 11 节严格 little-endian、紧密排列写入 `command.parameters`
（未用字节保持 0）：

| Kernel | 字节 | 字段顺序 | image flags |
|---|---:|---|---|
| Vector Add FP32 | 32 | A,B,C,count (u64×4) | 5 = SPMD |
| GEMM | 40 | A,B,C (u64), M,N,K,dtype (u32) | 6 = SINGLE_INVOCATION |
| AXPY FP32 | 28 | X,Y,count (u64), alpha (f32 bits) | 5 = SPMD |
| DOT FP32 | 32 | X,Y,result,count (u64) | 6 = SINGLE_INVOCATION |
| NRM2 FP32 | 24 | X,result,count (u64) | 6 = SINGLE_INVOCATION |

**grid/block 约定**（依据 `kernels/manifest.json` 的 `flags` 字段推断）：

- **SPMD**（vector_add / axpy，flags bit0）：一线程一元素，grid 覆盖 `count`。
  vector_add 由调用方给 grid/block（保留原样）；axpy 内部用 `block=(256,1,1)`、
  `grid=(ceil(count/256),1,1)`。
- **SINGLE_INVOCATION**（gemm / dot / nrm2，flags=6）：单次调用完成全部工作，
  统一用 `grid=(1,1,1)`、`block=(1,1,1)`。

> 该 grid/block 约定是从 image flags 推断的（源码内标注 `// TODO(Good)`），
> 需在 WSL/Linux 用真实设备回归确认。

GEMM 的 10 种 dtype 共用 `submit_gemm()`，只有 `dtype` 常量不同；运行时始终发射
**naive（variant 1）** image。整数 GEMM 输出 INT32（设备负责饱和），运行时不感知输出宽度。

---

## 4. Requirement 清单与完成状态

图例：✅ 已实现并对照公开契约、🟡 已接线但正确性需 WSL 回归、⛔ 未做（超出本文件范围）。

| ID | 分值 | Gate | 内容 | 状态 | 说明 / TODO |
|---|---:|---|---|:--:|---|
| R101 | 4 | **Basic** | Device/ISA query、错误名、TLS last error | ✅ | 保留起步包实现 |
| R102 | 6 | **Basic** | allocation/free、OOM、reuse、非法 free | ✅ | 转发 `aecDeviceAlloc(bytes,64,..)` / `aecDeviceFree` |
| R103 | 6 | **Basic** | 同步 H2D/D2H + allocation-relative bounds | ✅ | null/0 字节运行时拦截；越界由设备判定 |
| R104 | 4 | **Basic** | Vector Add image 与 launch mapping | ✅ | resolve(1,FP32,0) + 32B 参数块；校验 block≤1024、args_size |
| R201 | 10 | **Basic** | FP32 / INT32 GEMM | ✅ | `submit_gemm` 走 naive image；grid=block=(1,1,1) |
| R105 | 5 | Good | Stream FIFO 与异步操作 | 🟡 | 同步确定性 Stream 模型；FIFO 平凡满足 |
| R106 | 5 | Good | Event generation、cycles、异步错误 | 🟡 | Event 快照 Stream 累计周期；故障暂存→sync 上报 |
| R202 | 10 | Good | FP4/FP8/FP16/BF16/FP64 GEMM | 🟡 | 已接线（同一 `submit_gemm`），数值待 WSL 回归 |
| R203 | 4 | Good | INT4/INT8 与 INT32 饱和输出 | 🟡 | 已接线；打包/饱和由设备处理，待回归 |
| R204 | 6 | Good | FP32 AXPY / DOT / NRM2 | 🟡 | 已接线；AXPY=SPMD，DOT/NRM2=单次调用 |
| R301 | 6 | Good | ABI sequence、resolve、completion、stats | 🟡 | 全局递增 sequence；stats 逐字段镜像设备 |
| R302 | 6 | Good | 双 DMA 通道、异步边界与恢复 | 🟡 | Stream 按 `(id-1)%2` 轮转通道 0/1 |
| R303 | 4 | Good | host registration 与 zero-copy | 🟡 | 注册区间表；命中则打 REGISTERED\|ZERO_COPY |
| R304 | 4 | Good | DMA/ISA fault propagation 与恢复 | 🟡 | 故障暂存到 Stream；下一条合法命令可恢复 |
| R401 | 10 | Excellent | DMA policy Agent | ⛔ | `agents/dma_agent.py` 仍为合法 baseline stub |
| R402 | 10 | Excellent | Kernel-image policy Agent | ⛔ | `agents/kernel_agent.py` 仍为合法 baseline stub |

**等级 gate（`docs/05` 第 2 节）**：

- **Basic**：总分 ≥ 30 且 R101–R104、R201 全通过 → 本实现的**主要目标**。
- **Good**：总分 ≥ 75，Basic 通过，且 R105/R106/R202–R204/R301–R304 全通过。
- **Excellent**：总分 ≥ 90，Good 通过，两个 Agent correctness 通过且有正的隐藏加速。

---

## 5. 强制执行路径（不得绕过 AEC image）

`docs/01` 第 5 节 / `docs/02` 第 7 节要求，成功的 `aecLaunch` 与计算接口必须：

1. 用 `(kernel_id, dtype, variant)` 经 `aecDeviceResolveKernel` 解析冻结 image handle；
2. 按规范生成 little-endian 参数块；
3. 保留（launch）或按 image flags 选择（内部计算）合法 grid/block；
4. 提交 `AEC_DEVICE_OP_ISA_LAUNCH`；
5. 由设备解释并退休 AEC 指令。

**禁止**：在 Host 侧直接算出结果后绕过 image；自定义 code loader / ISA / trace；
自造 kernel image。评分会核对 image handle、retired count、trace digest、trap 与虚拟周期
（`grader/public_grade.py` 中 `_run_float_gemm` / `test_r104` / `test_r301` 均有校验）。
本实现严格遵守：所有计算路径都是 resolve + submit，无任何主机侧算子。

---

## 6. Agent 输入/输出 JSON 契约

两个 Agent 均为独立进程：从 **stdin 读一个 JSON**，向 **stdout 写一个 JSON**
（无多余字段/日志），单次超时 1 秒，stdout+stderr ≤ 64 KiB，不得联网/读评分器/跨 case 存状态。
结构以 `schemas/` 为准：

- `schemas/dma_agent_input.schema.json` / `schemas/dma_agent_output.schema.json`
- `schemas/kernel_agent_input.schema.json` / `schemas/kernel_agent_output.schema.json`

**DMA Agent**（`agents/dma_agent.py`）

- 输入：`{case_id, direction("h2d"/"d2h"), bytes, alignment, registered(bool), concurrency}`
- 输出（**只含**这四个键）：`{"channel":0/1, "chunk_bytes":4096/65536/1048576, "queue_depth":1/2/4/8, "use_zero_copy":bool}`
- 约束：`use_zero_copy` 仅在 `registered=true` 时可为 true。
- 目标周期公式（`docs/05` 第 3 节，越小越优）：
  ```
  setup + ceil(ceil(bytes/32)/parallelism) + 24*(ceil(bytes/chunk_bytes)-1) + alignment_penalty
  setup = 45 (registered zero-copy) 否则 100
  parallelism = min(queue_depth, concurrency, 2)
  alignment_penalty = 13 (alignment < 64) 否则 0
  ```

**Kernel Agent**（`agents/kernel_agent.py`）

- 输入：`{case_id, dtype, m, n, k, alignment, workspace, candidates:[{id,semantic_kernel_id,image_id,variant,workspace,alignment,divisibility},...]}`
- 输出（**只含**一个键）：`{"kernel_id":"<candidate-id>"}`
- 合法性：naive=任意合法 shape；tiled 需 M/N/K 均可整除 4；vectorized 需 M/N/K 均可整除 8
  且 alignment ≥ 16；并满足 candidate 自身 workspace/alignment/divisibility 约束。
- 周期来自设备对 image 的真实解释（`aecDeviceEvaluateKernel`），非 grader 估算公式。

---

## 7. Basic → Good 的后续 TODO 优先级

当前 `src/aec_runtime.cpp` 已把 Basic 全部接线、Good 大部分接线（同步确定性模型）。
建议在 WSL/Linux 上按下列顺序回归与打磨：

1. **回归 R201 GEMM grid/block 约定**：确认 SINGLE_INVOCATION 用 `(1,1,1)` 被设备接受
   且数值正确（这是 Basic gate 的关键假设）。
2. **回归 R105/R106 Stream/Event**：确认同步模型下 FIFO、Event 代际、elapsed>0、
   异步故障经 `aecStreamSync` 上报为 `AEC_ERROR_DEVICE`。
3. **回归 R202/R203 其余 dtype GEMM**：FP4/FP8(E4M3/E5M2)/FP16/BF16/FP64/INT4/INT8
   的存储格式与饱和输出。
4. **回归 R204 AXPY/DOT/NRM2**：确认 AXPY 的 SPMD block 宽度与 DOT/NRM2 单次调用约定。
5. **两个 Agent（Excellent）**：把 baseline stub 换成按上述周期公式/合法性择优的策略，
   争取正的隐藏加速。

> 源码内所有需要设备回归的位置均标注 `// TODO(Good)` / `// TODO(Good/Excellent)`。
> 若某高等级路径在真实设备上被拒绝，最保守的回退是让对应接口返回
> `AEC_ERROR_NOT_SUPPORTED`——这不会影响 Basic gate（Basic 只需 R101–R104、R201）。
