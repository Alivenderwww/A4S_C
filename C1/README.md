# AEC C1 编译器（赛道 C-C1）操作手册

## 1. 定位

本项目是赛道 C 子题 **C1：AEC IR 编译器** 的参赛实现骨架。目标是把 **PTX 风格中间表示**
编译为 **AEC ISA 128-bit 定长机器码**（`.aecbin`），并提供配套反汇编器与自动调优 Agent。

- 前端：PTX 词法/语法分析 → PTX AST。
- 中端：AST → 内部 IR（基本块 + CFG）→ 优化 pass 流水线。
- 后端：GEMM→TMUL 降级 → 循环展开 → 列表调度(pre-RA) → 线性扫描分配 → 降级 → 128-bit 编码 → `.aecbin` 容器。
- 工具：`aec-cc`（编译器）、`aec-objdump`（反汇编器）、`agent/run_agent.py`（自动调优）。

具体完成度、验证数字、风险、TODO 详见 [`../docs/C1-完成度审计.md`](../docs/C1-完成度审计.md)（工程状态唯一事实源）。

> 重要：本仓库只在 `C1/` 目录下工作，**不修改** `public/` 下的任何官方素材。

---

## 2. 目录结构

```
C1/
├── Makefile                     构建脚本（自动探测 -std=c++17/ c++11）
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
│   ├── passes/const_prop.cpp    常量传播          （T2）
│   ├── passes/dce.cpp           死代码消除        （T2）
│   ├── passes/cse.cpp           公共子表达式消除  （T2）
│   ├── passes/licm.cpp          循环不变量外提    （T2）
│   ├── passes/mem_coalesce.cpp  内存合并/复用     （T3）
│   ├── passes/pred_opt.cpp      谓词执行优化      （T2）
│   ├── regalloc/linear_scan.cpp 256-GPR 线性扫描分配（T4）
│   ├── sched/list_sched.cpp     DDG + 列表调度 + 双发射配对（T4）
│   ├── codegen/gemm_tmul.cpp    GEMM 识别 + TMUL 降级（T5）
│   ├── codegen/lower.cpp        末端合法化 + 展平 + 分支目标解析（T1）
│   ├── binfmt/writer.cpp        .aecbin 序列化
│   ├── binfmt/reader.cpp        .aecbin 解析
│   └── driver.cpp               流水线编排 + 编码 + 反汇编 + 周期估算
├── tools/
│   ├── aec-cc.cpp               编译器 CLI 入口
│   └── aec-objdump.cpp          反汇编器 CLI 入口
├── agent/run_agent.py           自动调优循环（多配置扫描 → 选优 → 重编 → 报告）
└── tests/run_public.sh          编译 + 反汇编 + 校验全部 5 个公开用例
```

---

## 3. 构建

```bash
cd C1
make            # 生成 bin/aec-cc 与 bin/aec-objdump
make selftest   # 构建并运行编码器 golden 自检
make test       # 构建并跑 tests/run_public.sh
make clean      # 清理 obj/ bin/
```

### 关于编译器版本（重要）

- **评测镜像使用 GCC 13.3（支持 C++17）**；本开发机装的是 **g++ 4.9.2**，它**不认识
  `-std=c++17`**（会报 `unrecognized command line option`）。
- Makefile 因此**自动探测**：先试 `-std=c++17`，不支持则回退 `-std=c++11`：
  ```make
  CXXSTD := $(shell echo 'int main(){return 0;}' | $(CXX) -std=c++17 -fsyntax-only -x c++ - 2>/dev/null && echo c++17 || echo c++11)
  ```
- **源码全部按 C++11 编写**（不使用 `std::optional`/`std::filesystem`/结构化绑定/
  `make_unique` 等 C++14/17 库特性），因此在 g++ 4.9.2 与 GCC 13.3 上都能编译。
- **Windows 提示**：请在 **Git Bash / MSYS2 / WSL** 下执行 `make`（recipe 里用了
  `mkdir -p`、`rm -rf` 等 POSIX 命令）。MinGW 下产物会带 `.exe` 后缀，Makefile 已用
  `$(EXE)` 处理；Git Bash 下 `./bin/aec-cc` 能自动匹配 `aec-cc.exe`。若用原生 GCC 建议
  直接在 WSL/新版 g++ 下构建，与评测环境一致。

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

# 单独关闭某个 pass（供 Agent 探索配置空间）
bin/aec-cc input.ptx -O2 --no-cse --no-dual-issue -o out.aecbin

# 反汇编为可读 AEC 汇编
bin/aec-objdump out.aecbin

# 编码器 golden 自检
bin/aec-cc --selftest

# 自动调优（扫描多配置、选周期最少者、重编并生成最终报告）
python3 agent/run_agent.py input.ptx -o out.aecbin --report agent_report.json
```

`aec-cc` 支持的开关：`-O0/-O2/-O3`、`-o`、`--report`、`--sched-window N`、
`--no-const-prop|--no-dce|--no-cse|--no-licm|--no-mem-coalesce|--no-pred-opt|--no-dual-issue|--no-gemm`、
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
      │    const_prop → cse → licm → dce →（迭代）→ mem_coalesce → pred_opt
      ▼
   优化后 IR
      │  gemm_tmul.cpp（GEMM 识别 + TMUL 降级）
      │  unroll.cpp（循环展开，-O3 按选项）
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

Pred/Ctrl 位域（Track-B spec.md §3.2）：

| 位 | 含义 |
|---:|---|
| 15    | predication enable（`BRX` 不置位，谓词直接放 2:0） |
| 14    | pred_neg |
| 13:11 | LD/ST/ATOM memory space（gmem=0,smem=1,cmem=2,lmem=3,pmem=4）/ MBAR scope；CVT* 的源类型占 [13:10] |
| 10:8  | 指令族 subop（CMP 比较码 / SHUF/VOTE mode / TMUL mode） |
| 7     | 保留（必须 0） |
| 6:3   | 数据类型 selector（`.none` 指令为 0xf） |
| 2:0   | 谓词 P0–P7 |

类型 selector（Track-B §4）：`b32=0x0 b64=0x1 u32=0x2 s32=0x3 u8=0x4 s8=0x5 f32=0x8 f64=0x9 f16=0xa bf16=0xb none=0xf`。`0x6/0x7/0xc–0xe` 保留 —— AEC 不存在 fp8/fp4/int4 标量类型。
比较 selector：`eq=0 ne=1 lt=2 le=3 gt=4 ge=5`。
特殊寄存器 selector（放 Src1）：`tid.x=0x100 ntid.x=0x101 ctaid.x=0x102 nctaid.x=0x103 laneid=0x104`，y/z 分量为 0x110.. / 0x120..。
CVT*：目标类型 [6:3]，源类型 [13:10]，[9:7]=0（Track-B §5.3）。

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

> golden 验证：`src/isa/encoder.cpp::selfTest()` 内置 8 条向量，取自
> `Track-B/testcases/tests/aec_cases/cvtff/program.bin`（Track-B §A.1 编码），
> 逐位一致，可用 `bin/aec-cc --selftest` 运行。此外 12 个自包含 Track-B
> `aec_cases` 在 `sim/` 上执行结果与其 `expected/gmem` 逐字节一致。

---

## 7. `.aecbin` 容器格式

自定义、显式、小端、与主机字节序无关（见 `include/aec/binfmt.h`）：

```
[ FileHeader 32B ]
[ SectionEntry × sectionCount，每个 16B ]
[ CODE   段 ]  16 × instructionCount 字节（每条 = word0..word3 小端）
[ DATA   段 ]  参数块 / 常量原始字节
[ RELOC  段 ]  u32 count + RelocEntry[]（每个 16B：instrIndex,kind,addend,reserved）
[ SYMBOL 段 ]  u32 count +（u32 nameLen + name 字节 + u32 value + u32 kind）×
```

- FileHeader：`magic='AEC1'(0x31434541) version headerBytes sectionCount entryPC
  instructionCount paramBytes flags`。
- 本骨架恒定输出 4 个段：`CODE / DATA / RELOC / SYMBOL`，满足 spec.md 对必备段的要求。
- Relocation：`RELOC_PARAM_ADDR` 表示该指令的立即数是参数块字节偏移。
- Symbol：kind=0 为 kernel 入口，kind=1 为标签（值为绝对指令下标）。
- ⚠️ 与 spec.md 的容器要求相对，Track-B 实际用的是 **`aecbin-raw`**（裸 128-bit 指令流，无头无段；见 `Track-B/testcases/.../build.json`），设备把裸字装 IMEM、参数走 PMEM。最终提交格式（裸流 vs 本容器）待组委会确认（P0.5-A）。

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

### T3 — 内存优化（正确性 10、性能 9）
- **[P2]** `src/passes/mem_coalesce.cpp`：合并同基址+仿射偏移的相邻访问为宽事务；把循环不变
  的重复 load 提升到 SMEM（`TLDA`）/寄存器。PTX-03 的 `[%rd5]` 每次迭代重载是典型目标。

### T2 — 控制与标量优化（正确性 8、性能 5）
- **[P2]** `src/passes/cse.cpp`：对纯指令做值编号并复用（PTX-02 有两条相同 `add.f32` +
  冗余 `mul.f32`）。
- **[P2]** `src/passes/licm.cpp`：从 CFG back-edge 识别循环，把不变量提升到 preheader
  （PTX-02 的 `%f1+%f2` 在 LOOP 内却是不变量）。
- **[P3]** `src/passes/const_prop.cpp`：LOADI 常量折叠 + 传播。
- **[P3]** `src/passes/dce.cpp`：按 use-count 删除无副作用死指令（迭代到不动点）。
- **[P3]** `src/passes/pred_opt.cpp`：小分支 if-conversion 成谓词直线代码（各用例尾部的
  `@%pN bra DONE`）。

### T1 — 基础 Lowering（正确性 4，门禁基础）
- **[P1]** `src/ir/ir_builder.cpp`：完善 lowering 语义正确性——
  `mul.wide` 的 64 位结果、`add.u64` 的寄存器对（R[n]:R[n+1]）进位、`cvt` 各精度规则、
  `@!%pN`（谓词取反）分支、`mov` 各类型；补充未覆盖 PTX 指令。
- **[P3]** `src/driver.cpp`：支持多 kernel 输出（当前只发第一个 kernel）。

---

## 9. 风险提示

- **无独立 golden model / cycle model**：C1 未随包提供专用参考执行/周期模型。但：
  - Track-B `testcases/tests/aec_cases/` 提供 36 个 `program.bin` + 35 个 `expected/gmem` 执行 golden，既验编码又验执行语义（12/12 自包含用例逐字节一致）；
  - Agent 的 `est_cycles` 是编译器自带的**启发式估算**（`driver.cpp::estimateCycles`），
    不是官方周期数。真实周期模型可用后，应替换 `agent/run_agent.py::read_cycles` 的来源。
- **仅覆盖真实 PTX 子集**：解析器按公开用例（真实 NVIDIA PTX：`.version 7.0`/`sm_70`/
  `%tid.x`/`mad.lo.u32`/`setp`/`bra` 等）实现；未识别的指令会打 `UNHANDLED:` 标记而非报错
  （利于鲁棒性变异测试），但语义未覆盖，需按 T1 TODO 扩展。
- **优化 pass 为 identity 桩**：当前 `-O0` 与 `-O2/-O3` 生成的代码相同，性能分尚未启动；
  正确性门禁依赖 lowering + 编码（已通路），性能收益需按第 8 节填充。
- **64-bit 地址近似**：`add.u64`/`mul.wide` 目前按 32 位近似处理，未实现寄存器对进位，
  是执行正确性的已知缺口（见 T1 TODO）。
