# AEC 功能模拟器 + C1 正确性对拍（dogfooding oracle）

> **为什么存在**：C1 官方**不发 golden model / cycle model**（那是赛道 B 的交付物）。
> 这套工具是我们**自建的本地正确性 oracle**：把编译出的 `.aecbin` 在一个独立实现的 AEC
> 功能模拟器上执行，和 numpy 独立参考结果对拍，从而在没有官方 golden 的情况下自测 C1 的
> 执行正确性。详见 [../C1_实现流程分析.md](../C1_实现流程分析.md) §5。

## 组成

| 文件 | 作用 |
|------|------|
| `aec_decode.py` | `.aecbin` 解析 + 128-bit 指令解码（`encoder.cpp` 的逆）。`--selftest` 对 8 条 golden 向量验证解码 |
| `aec_sim.py` | AEC 功能模拟器：32-lane warp、lockstep、numpy 向量化。256 GPR/lane + 8 谓词 + gmem/pmem/smem/lmem |
| `cases.py` | 5 个公开 kernel 的**参数化** launch 配置 + numpy 独立参考 |
| `dogfood.py` | 对拍驱动（默认 5 个 case）：编译 → 模拟 → 比对，输出 PASS/FAIL + max_abs_diff + cycles |
| `mutate.py` | 语义保持的 PTX 变异（寄存器重命名、插入死代码），对标赛题鲁棒性测法 |
| `bench.py` | **广泛 bench**：尺寸扫描（含非 blockDim 整数倍→测发散）+ 变异变体，全部 oracle 自动对拍 |

## 用法

```bash
# 0) 先建编译器
cd .. && make && cd sim

# 1) 解码器自检（应输出 "all 8 golden vectors decode correctly"）
py -3.13 aec_decode.py --selftest

# 2) 全部对拍（或选子集：vadd poly reuse reg gemm）
py -3.13 dogfood.py
py -3.13 dogfood.py vadd reuse --opt O2
py -3.13 dogfood.py --strict          # 把 ISA 非法类型(如 ADD.b64)标成错误

# 3) 广泛 bench（尺寸扫描 + 变异变体，覆盖远超 5 个固定 case）
py -3.13 bench.py                 # 全套
py -3.13 bench.py --mutants 10    # 每个 kernel 更多变异
py -3.13 bench.py gemm vadd       # 选子集

# 4) 反汇编任意 .aecbin（排错用）
py -3.13 aec_decode.py build/PTX-03_repeated_reuse.aecbin
```

### 广泛 bench 现状（113 变体，覆盖赛题全部 8 类变异）

| 赛题变异类型 | 实现 | 结果 |
|---|---|---|
| ① 参数/矩阵尺寸 | 尺寸扫描 | PASS（含大尺寸） |
| ② 寄存器重命名 | `mutate.rename_registers` | 全 PASS |
| ③ 基本块重排 | `mutate.reorder_blocks`（显式化 fallthrough + 洗牌） | 全 PASS |
| ④ 循环次数变更 | `mutate.set_loop_count` + 重算参考（poly/reuse） | 全 PASS |
| ⑤ 死代码插入 | `mutate.insert_dead_code` | 全 PASS |
| ⑥ 寄存器压力增加 | `mutate.increase_register_pressure`（-O0） | 全 PASS（**曾暴露 b64 pair clobber bug，已修**） |
| ⑦ 数据类型变更 | `mutate.gemm_to_{bf16,f32}` + 重算参考 | bf16/f32 PASS；fp8/fp4/int 待 T5 TMUL |
| ⑧ 内存复用模式 | reuse block 扫描（col-vs-idx 复用度） | 全 PASS |

**汇总：113 PASS / 0 DIVERGE / 0 FAIL** —— pred_opt（越界 guard 谓词化）已实现，非整数倍 N 的发散情况全部转绿。

> bench 至今抓出的 bug：① reuse 的 R5 跨循环 clobber（regalloc 活跃度）；② CVT 源类型未编码；③ 跨块 CSE；④ **b64 寄存器对 clobber**（寄存器压力变异暴露 —— 脚手架在发非法 `ADD.b64`/`LD.pmem.b64`，且 allocator 不为 pair 保留 Rd+1；已按 §1.2 把地址运算降为合法 32-bit）。这四个都是 oracle+bench 抓出来的真实 bug。

`py -3.13` 是本机装了 numpy 的解释器（PATH 上的 `python`=3.12 无 numpy）。退出码非 0 表示有 case 失败，可进 CI。

## 当前基线结果：**5/5 正确性通过（T1-T5 全绿）**

| case | 结果 | 说明 |
|------|------|------|
| vadd (T1) | ✅ bit-exact | 基础 lowering |
| poly (T2) | ✅ bit-exact | 循环 + 标量算术 |
| reuse (T3) | ✅ bit-exact | **已修**：改成活跃度感知的线性扫描寄存器分配（跨循环回边正确延伸区间），不再复用循环活跃的 R5 |
| reg (T4) | ✅ bit-exact | 多路独立算术 |
| gemm (T5) | ✅ | **已修**：(1) CVT 现在把源类型编码进 [13:10]（之前只编 dst → golden 会当 f32→f32 空拷）；(2) `mad.f32`→`FMA`（sm_70 融合语义）。**残留风险**：f16 用 `LD.b32` 读低 16 位，若输入缓冲恰在 gmem 末尾，末元素会多读 2 字节（flat gmem 内一般无害；边界情况需 aligned 2-byte load 或走 TMUL）；TMUL 下降（T5 的 11 性能分）仍 TODO |

> 修复经过（本轮）：`regalloc/linear_scan.cpp` 重写为真·liveness 线性扫描；`isa/encoder.cpp` + `ir/ir_builder.cpp` 补 CVT 源类型编码 + `mad.f32`→FMA；`sim` 侧 CVT 改为忠实读取编码的源类型（不再猜 f16），使 oracle 不再掩盖 bug。

## ⚠️ 这套 oracle 的边界（别过度信任）

- 它验证的是"**编译器是否产出了 PTX 源意图的计算 + 合法控制流**"，**不是**官方隐藏 golden 的
  bit-exact 替身。sim 语义与真 golden 若有出入，sim 通过≠官方通过。
- FP：FMA 单次舍入（float64 中间量），MAD 两次舍入 → 若 `mad.f32` 误映射成 MAD 会被抓出。
- 整数 ADD/SUB/MUL 对任意整型/位型都按 32-bit 回绕（让现脚手架的 `ADD.b64` 指针运算能跑）；
  `--strict` 会把 ISA 非法类型标红。
- **`BRX` 要求 active lane 条件一致**：发散分支（未谓词化的越界 guard 在部分块）会 raise
  ExecError —— 这正是用来抓 §1.3"越界 guard 必须 if-conversion 谓词化"的。当前 case 的 N 都是
  blockDim 整数倍（不发散）所以能跑；**修完谓词化后把 N 改成非整数倍即可测发散正确性**。
- 单 warp 顺序执行、无 atomics/barrier 建模（公开 kernel 用不到）。

## 怎么加新 case / 用于回归

在 `cases.py` 加一个函数返回 `dict(ptx=, grid=, block=, param=, gmem=, out=(off,count,dtype), ref=)`，
注册到 `ALL`。改任何 lowering / pass 后跑 `py -3.13 dogfood.py` 即可回归。建议每实现一个
Phase（见 [../C1_实现流程分析.md](../C1_实现流程分析.md) §4）就把对应 case 从 FAIL 推到 PASS。
