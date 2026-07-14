# AEC C1 编译器（赛道 C-C1）操作手册

## 1. 定位

本项目是赛道 C 子题 **C1：AEC IR 编译器** 的参赛实现骨架。目标是把 **PTX 风格中间表示**
编译为 **AEC ISA 128-bit 定长机器码**（`.aecbin`），并提供配套反汇编器与自动调优 Agent。

- 前端：PTX 词法/语法分析 → PTX AST。
- 中端：AST → 内部 IR（基本块 + CFG）→ 优化 pass 流水线。
- 后端：谓词化 if-conversion → 循环展开 → 列表调度(pre-RA) → 线性扫描分配 → 降级 → 128-bit 编码 → 裸 `.aecbin` 指令流。
- 工具：`aec-cc`（编译器，提交入口 `compiler/aec-cc`）、`aec-objdump`（配套反汇编器）。

具体完成度、验证数字、风险、TODO 详见 [`../docs/C1-完成度审计.md`](../docs/C1-完成度审计.md)（工程状态唯一事实源）。

> 重要：本仓库只在 `C1/` 目录下工作，**不修改** `public/` 下的任何官方素材。

---

## 2. 目录结构

```
C1/
├── Makefile                     构建脚本（固定 -std=c++11，POSIX recipe）
├── README.md                    本手册
├── include/aec/                 公共头文件
│   ├── isa.h                    AEC ISA 常量 + 128-bit 编码器/解码器声明
│   ├── ir.h                     内部 IR：Function/BasicBlock/Inst/Operand
│   ├── ptx_ast.h                PTX 抽象语法树
│   ├── binfmt.h                 .aecbin 容器格式（Header/Section/Reloc/Symbol）
│   ├── target.h                 机器限制 + 编译选项 Options（-O 级别、pass 开关）
│   ├── passes.h                 各阶段函数声明（buildIR/buildCFG/passes/regalloc/…）
│   └── driver.h                 前端 parse + 完整 compile 流水线 + 反汇编 API
├── src/
│   ├── isa/encoder.cpp          逐位编码器 + golden 自检
│   ├── ptx/lexer.cpp            PTX 分词器
│   ├── ptx/ptx_lexer.h          词法私有接口
│   ├── ptx/parser.cpp           tokens → PTX AST（递归下降）
│   ├── ir/ir_builder.cpp        PTX AST → IR + 指令选择 + 基本块切分
│   ├── ir/cfg.cpp               CFG 构建（succ/pred）
│   ├── passes/const_prop.cpp    常量传播 / 折叠     （T2）
│   ├── passes/copy_prop.cpp     复制传播           （T2）
│   ├── passes/cse.cpp           公共子表达式消除 + 冗余 load 消除（T2/T3）
│   ├── passes/mad_contract.cpp  MUL;ADD → MAD 合并（非融合，省指令）（T2-T5）
│   ├── passes/dce.cpp           死代码消除         （T2）
│   ├── passes/licm.cpp          循环不变量外提 + 循环不变 load 外提（T2/T3）
│   ├── passes/pred_opt.cpp      边界 guard if-conversion（T2/正确性）
│   ├── passes/loop_rotate.cpp   while → do-while 规范化（使能展开，T5）
│   ├── passes/strength_reduce.cpp 地址归纳变量强度削减（→加法递推，T5）
│   ├── passes/unroll.cpp        循环展开（省循环控制指令）（T5）
│   ├── regalloc/linear_scan.cpp 256-GPR 线性扫描分配（T4）
│   ├── sched/list_sched.cpp     DDG + 列表调度 + 双发射配对（T4）
│   ├── codegen/lower.cpp        末端合法化 + 展平 + 分支目标解析（T1/T5）
│   ├── binfmt/writer.cpp        .aecbin 序列化
│   ├── binfmt/reader.cpp        .aecbin 解析
│   └── driver.cpp               流水线编排 + 编码 + 反汇编 + 周期估算
├── tools/
│   ├── aec-cc.cpp               编译器 CLI 入口
│   └── aec-objdump.cpp          反汇编器 CLI 入口
├── sim/                         自建 AEC 功能模拟器 + 官方 CModel 验证 harness
└── tests/run_public.sh          编译 + 反汇编 + 校验全部 5 个公开用例
```

---

## 3. 构建

```bash
cd C1
make               # 生成 bin/aec-cc 与 bin/aec-objdump
make selftest      # 构建并运行编码器 golden 自检
make test          # 聚合：test-public + test-extreme（public 门禁 + 本地 fast contract 套件）
make test-public   # 构建并跑 tests/run_public.sh（编译/反汇编/校验全部 5 个公开用例）
make test-extreme  # 本地 fast contract 套件（自建模拟器，非官方 CModel）
make test-frontier # 本地 fast frontier  套件（自建模拟器，非官方 CModel）
make clean         # 清理 obj/ bin/
```

### 关于编译器标准（重要）

- **固定 `-std=c++11`**：源码全部按 C++11 编写并在 g++ 4.9.2 上测试；评测镜像的
  **GCC 13.3 完全支持 c++11**，固定它（而非启用 c++17）让构建停在代码实际编译验证过的
  标准上，避免依赖任何从未测过的 C++17 行为。
- 不使用 C++14/17 库特性（`std::optional`/`std::filesystem`/结构化绑定/`make_unique`），
  也不使用 C++17 移除的特性（`register`/`std::auto_ptr`/`std::random_shuffle`/动态异常规格）。
- `-finput-charset=UTF-8` 保证源码里的 UTF-8 中文注释跨 locale 可移植。
- **提交入口 `compiler/aec-cc`**：是一个**入库的 wrapper 脚本**（不是预编译二进制——编译器
  平台相关，必须在评测机上 build）。它首次调用时自动 `make build` 出 native `bin/aec-cc` 再
  exec，因此无论评测机是否预先构建，`compiler/aec-cc kernel.ptx -O2 -o out --report r.json`
  都能直接工作。本机（MinGW）另在 `compiler/aec-cc.exe` 放一份 native 供本地测试。
- **提交前建议**：在一台带 GCC 的 Linux 机器上跑一次 `make build submit`，确认评测环境能
   从源码构建出 `compiler/aec-cc`（本开发机 g++ 4.9.2 只能验证 c++11 编译，未在 GCC 13/ARM 上实测）。

### 测试目标详解

```bash
# make test 聚合了以下两条：
make test-public   # ① 公开门禁：编译/反汇编/校验全部 5 个公开用例
make test-extreme  # ② 本地 fast contract 套件（自建模拟器，非官方 CModel）

# 单独跑 frontier 套件（不在 make test 内，按需使用）：
make test-frontier
```

每个 `test-*` 目标的等价直接命令（C1 为工作目录，模块导入基于包路径）：

```bash
# 等价于 make test-public（bash 脚本包装的编译+反汇编+校验）
bash tests/run_public.sh

# 等价于 make test-extreme
python3 -m tests.extreme.run_extreme --suite contract --backend local --profile fast --opt all

# 等价于 make test-frontier
python3 -m tests.extreme.run_extreme --suite frontier --backend local --profile fast --opt O2
```

> **`--opt` 是必需的**：`run_extreme` 要求显式选择 `O0`、`O2`、`O3` 或 `all`；
> 不传 `--opt` 会报错，`make test-*` 已包含该参数。

> **Windows 注意**：若系统没有 `python3` 这个命令（常见于原生 Windows Python 安装），
> 请显式替换为 `py -3.13`（或你的实际 Python 版本），不要依赖隐式解释器回退：
> `py -3.13 -m tests.extreme.run_extreme --suite contract --backend local --profile fast --opt all`。

### 官方 CModel 验证（Linux 服务器）

本地 `--backend local` 使用自建 AEC 功能模拟器（`sim/aec_sim.py`）作 oracle，
**PASS 不代表官方 CModel 认可**。正式验证须在 Linux x86-64 服务器上通过官方 golden model
运行，由仓库根目录的 `scripts/remote_exec.py` 提交：

```bash
# 从 repo 根目录运行；将 <isolated-checkout> 替换为服务器上的隔离快照路径。
python scripts/remote_exec.py "cd <isolated-checkout>/C1 && python3 -m tests.extreme.run_extreme \
  --suite contract --backend cmodel --profile strict --opt all"

python scripts/remote_exec.py "cd <isolated-checkout>/C1 && python3 -m tests.extreme.run_extreme \
  --suite frontier --backend cmodel --profile strict --opt O2"
```

上述命令在 Linux 服务器上以 `--backend cmodel --profile strict` 运行，对比官方 AEC
golden model 输出。**不需要本地 WSL 环境**。

### 测试语义说明

| 结果 | 含义 |
|------|------|
| **PASS** | 当前所选 backend 的执行与 oracle 完整匹配；只有 `--backend cmodel` 的 PASS 才是官方 CModel 证据。 |
| **XFAIL** | **预期失败**——代码已执行、输出已比对、结果已注册到用例清单的**已知编译器缺陷**。XFAIL 不是跳过，也不是未执行；它是有跟踪记录的、可复现的被接受偏离。 |
| **FAIL** | 实现与 oracle 不一致，退出码非零，需调查。 |
| **XPASS** | 标记为 XFAIL 的用例意外 PASS——同样退出码非零，需调查（可能是缺陷已被修复但未更新清单）。 |

---

## 4. 用法示例

```bash
# 基本编译（默认 -O2）
bin/aec-cc input.ptx -o output.aecbin

# 指定优化级别（-O0 跳过所有优化 pass；-O2/-O3 按顺序执行）
bin/aec-cc input.ptx -O0 -o out.aecbin
bin/aec-cc input.ptx -O3 -o out.aecbin

# 输出性能报告（供 Agent 读取）
bin/aec-cc input.ptx -O2 -o out.aecbin --report perf.json

# 单独关闭某个 pass（A/B 对比调试）
bin/aec-cc input.ptx -O2 --no-cse --no-mad-contract -o out.aecbin

# 反汇编为可读 AEC 汇编
bin/aec-objdump out.aecbin

# 编码器 golden 自检
bin/aec-cc --selftest
```

`aec-cc` 支持的开关：`-O0/-O2/-O3`、`-o`、`--report`、`--sched-window N`、
`--no-const-prop|--no-copy-prop|--no-dce|--no-cse|--no-licm|--no-mad-contract|--no-pred-opt|--no-dual-issue`、
`--selftest`、`-v/--verbose`、`-h/--help`。

---

## 5. 架构与数据流

```
   input.ptx
      │  lexer.cpp（分词）
      ▼
   Token 流
      │  parser.cpp（递归下降）
      ▼
   ptx::Module (AST)                       ← ptx_ast.h
      │  ir_builder.cpp（指令选择 + 基本块切分）
      ▼
   ir::Program / Function / BasicBlock      ← ir.h
      │  cfg.cpp（succ/pred 边）
      ▼
   带 CFG 的 IR
      │  passes/*.cpp（-O2/-O3 按序执行；-O0 跳过）
      │    const_prop → copy_prop → cse → mad_contract → licm → dce →（迭代）→ pred_opt
      ▼
   优化后 IR
      │  loop_rotate → strength_reduce → unroll（-O2；while→do-while→地址递推→展开）
      │  list_sched.cpp（DDG + 列表调度 + 双发射配对，pre-RA）
      │  linear_scan.cpp（虚拟寄存器 → 物理 R1..R255）
      │  lower.cpp（展平 + 分支标签 → 绝对 PC）
      ▼
   线性 ir::Inst 流
      │  driver.cpp::toFields + isa::encode（逐条 128-bit 编码）
      ▼
   isa::Word128[] + Reloc + Symbol
      │  binfmt/writer.cpp
      ▼
   output.aecbin  ──(reader.cpp + driver.cpp::disassemble)──►  aec-objdump 可读汇编
```

各文件职责一句话概括见第 2 节目录树内注释；关键约定：

- **虚拟寄存器**：`ir_builder` 为每个 PTX 寄存器分配一个虚拟寄存器；`linear_scan` 才改写为
  `Operand::Phys`（物理 R 号，R0 保留作 scratch/zero）。
- **谓词**：`%pN` 直接映射为谓词 id N（0..7）；`setp` 写 `CMPP` 的 dst，`@%pN bra` 生成
  `BRX`，二者共用同一 id。
- **立即数**：算术指令的立即数操作数由 `ir_builder` 通过前置 `LOADI` 物化成寄存器；只有
  `LOADI/BR/BRX/LD.pmem` 把立即数放进 word0。
- **参数块**：`ld.param.*` 降级为 `LD.pmem`，其 word0 为参数块内字节偏移，并在
  Relocation 段登记 `RELOC_PARAM_ADDR`。

---

## 6. AEC ISA 128-bit 编码速查表

一条指令 = 4 个 little-endian `uint32`：

```
word3 = Opcode:16 | Pred/Ctrl:16
word2 = Dest:16   | Src1:16
word1 = Src2 / 指令专用字段
word0 = Imm32 / Src3
```

Pred/Ctrl 位域（C1 spec §5.2）：

| 位 | 含义 |
|---:|---|
| 15    | predication enable（`BRX` 不置位，谓词直接放 2:0） |
| 14    | pred_neg |
| 13:11 | LD/ST/ATOM memory space（gmem=0,smem=1,cmem=2,lmem=3,pmem=4）/ MBAR scope；CVT* 的源类型占 [13:10] |
| 10:8  | 指令族 subop（`CMP`/`CMPP` 比较码 eq..ge） |
| 7     | 保留（必须 0） |
| 6:3   | 数据类型 selector（`.none` 指令为 0xf） |
| 2:0   | 谓词 P0–P7 |

类型 selector（C1 spec §5.3）：`b32=0x0 b64=0x1 u32=0x2 s32=0x3 f32=0x8 none=0xf`（PTX 子集用到的六种；`u8=0x4 s8=0x5 f64=0x9 f16=0xa bf16=0xb` 属 AEC 扩展 ISA，仅 dev harness 用）。
比较 selector：`eq=0 ne=1 lt=2 le=3 gt=4 ge=5`。
特殊寄存器 selector（放 Src1）：`tid.x=0x100 ntid.x=0x101 ctaid.x=0x102 nctaid.x=0x103 laneid=0x104`，y/z 分量为 0x110.. / 0x120..。
CVT*（AEC 扩展 ISA，dev harness 用）：目标类型 [6:3]，源类型 [13:10]，[9:7]=0。

常用指令形态：

| 指令 | 编码要点 |
|---|---|
| `LOADI.type Rd,#imm`   | Dest=Rd，word0=imm32 |
| `CPY.type Rd,%special` | special selector 放 Src1（普通寄存器则为寄存器拷贝） |
| `MAD/FMA Rd,Ra,Rb,Rc`  | Src2=Rb 在 word1，Src3=Rc 在 word0 |
| `CMPP.cmp.type Pd,Ra,Rb` | Dest=谓词号，cmp 在 Pred/Ctrl[10:8] |
| `BRX Pn,target`        | 谓词在 [2:0]，word0=目标绝对指令下标 |
| `LD.gmem.type Rd,[Ra]` | Src1=地址寄存器，space 在 [13:11]；`.f64` 载入用 `LD.b64` 到寄存器对 |
| `ST.gmem.type [Ra],Rs` | Dest=0，Src1=地址，Src2=源；ST 只有 32 位，`.f64` 存储拆两条 `ST.b32` |

> golden 验证：`src/isa/encoder.cpp::selfTest()` 内置 8 条向量，取自一份参考 AEC
> `program.bin`（cvtff 用例），逐位一致，可用 `bin/aec-cc --selftest` 运行；
> 解码器 `sim/aec_decode.py --selftest` 同样对这 8 条逐位校验。

---

## 7. `.aecbin` 格式

裸 AEC 128-bit 指令流（C1 spec §10）：无 header、无 data / relocation / symbol 段，
`entry_pc=0`，所有 label 在编译期解析为绝对指令下标。

- 每条指令 16 字节 = `word0..word3`，小端；文件写入顺序 `w0, w1, w2, w3`（`writer.cpp`）。
- 文件大小是 16 的非零倍数，至少一条指令。
- 参数按 ABI 布局进 `.pmem`，运行时以 `LOADI 偏移; LD.pmem [Rtmp]` 寄存器寻址（C1 spec §7）。

> 反汇编器（`aec-objdump`）在内存 `Image` 里另保留符号/重定位表，仅用于带标签的
> dump，不写入 `.aecbin`。

---

## 8. 后续工作与风险

所有实现状态、验证数字、风险、TODO 的完整清单见
[`../docs/C1-完成度审计.md`](../docs/C1-完成度审计.md)：

| 审计章节 | 内容 |
|-----------|------|
| §三–§五 | 实现矩阵、完成度估计、优先级与 TODO |
| §七 | 完整风险清单（自建 oracle 边界、工程风险） |
| §六 | Performance Model 参数表与缺失假设 |

### T4 — 寄存器与调度（正确性 12、性能 10）
- **[P1]** `src/regalloc/linear_scan.cpp`：实现真正的 **spill**（选牺牲区间、分配 LMEM 槽、
  在 def/use 处插入 ST/LD、改写操作数）。当前超 255 寄存器时仅计数并钳位到 R255（桩）。
- **[P1]** `src/sched/list_sched.cpp`：构建完整 DDG（RAW/WAR/WAW + AEC 延迟），按关键路径
  高度做 ready-list 列表调度，交织 LD/计算隐藏访存延迟并最大化双发射配对。当前仅统计相邻
  可配对数、**不重排**。

### T3 — 内存优化（性能 10）
- ✅ 已实现：`cse.cpp` 直线冗余 load 消除（T3 的 `[%rd6]` 两次载入 → 复用寄存器，
  gmem load 4→3）；`licm.cpp` 循环不变 load 外提；list 调度器提前发射 load 隐藏延迟。
- **[P3]** 未来：128-byte 宽事务合并 / 把跨迭代复用的 load 提升到 SMEM（跨线程、复杂、低 ROI）。

### T2 — 控制与标量优化（性能 8）
- ✅ 已实现：`const_prop`（LOADI 常量折叠 + 级联传播）、`cse`（块内值编号）、
  `licm`（自然循环不变量外提）、`dce`（迭代 use-count 删死码）、`pred_opt`（边界 guard if-conversion）。
- **[P3]** 未来：基本块合并（公开用例热点在单块内，ROI 低）。

### T1 — 基础 Lowering（正确性 4，门禁基础）
- ✅ 已实现：special register、`mad.lo`/`mul.wide`/`add.u64` 地址计算（32-bit 折叠，
  高 32 位恒 0，见 C1 spec §8.2）、`@%pN`/`@!%pN`（含谓词取反）分支、`mov` 各类型、
  `ret→HALT`。5/5 公开用例 + 132 变体通过。
- **[P3]** `src/driver.cpp`：支持多 kernel 输出（当前只发第一个 kernel）。

---

## 9. 风险提示

- **周期模型**：评测平台不暴露固定 cycle model（组委会已明确）。C1 资料包随附 ARM
  golden model 作正确性 oracle；本仓另有 `sim/aec_sim.py` 自建功能 oracle 交叉验证。
  - Agent 的 `est_cycles` 是编译器自带的**启发式估算**（`driver.cpp::estimateCycles`），
    仅用于在候选配置间择优，不是官方周期数。
- **仅覆盖真实 PTX 子集**：解析器按公开用例（真实 NVIDIA PTX：`.version 7.0`/`sm_70`/
  `%tid.x`/`mad.lo.u32`/`setp`/`bra` 等）实现；未识别的指令会打 `UNHANDLED:` 标记而非报错
  （利于鲁棒性变异测试），但语义未覆盖，需按 T1 TODO 扩展。
- **优化 pass 为 identity 桩**：当前 `-O0` 与 `-O2/-O3` 生成的代码相同，性能分尚未启动；
  正确性门禁依赖 lowering + 编码（已通路），性能收益需按第 8 节填充。
- **64-bit 地址近似**：`add.u64`/`mul.wide` 目前按 32 位近似处理，未实现寄存器对进位，
  是执行正确性的已知缺口（见 T1 TODO）。
