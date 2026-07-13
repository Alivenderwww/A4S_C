# AEC 功能模拟器 + C1 正确性对拍（dogfooding oracle）

> **为什么存在**：C1 官方**不发 golden model / cycle model**（那是赛道 B 的交付物）。
> 这套工具是我们**自建的本地正确性 oracle**：把编译出的 `.aecbin` 在一个独立实现的 AEC
> 功能模拟器上执行，和 numpy 独立参考结果对拍，从而在没有官方 golden 的情况下自测 C1 的
> 执行正确性。详见 [../C1_实现流程分析.md](../C1_实现流程分析.md) §5 及
> [`../../docs/C1-完成度审计.md`](../../docs/C1-完成度审计.md)（工程状态唯一事实源）。

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

# uv 环境（无全局 numpy 时）
uv run --python 3.13 --with numpy python bench.py

# 4) 反汇编任意 .aecbin（排错用）
py -3.13 aec_decode.py build/PTX-03_repeated_reuse.aecbin
```

### bench 输出解读

`bench.py` 运行后输出每类变异的结果（PASS/DIVERGE/FAIL/SIM-ERR/COMPILE-ERR），每类最后一行显示汇总计数。所有五类非 PASS 计数均为 0 代表全部通过。

dogfood.py 额外输出 cycles 字段，该值为**自建 sim scoreboard 代理的 cycle 计数**（顺序执行 + stall 累加），非官方隐藏评测或官方 cycle 模型的结果。

当前回归结果（PASS/DIVERGE/FAIL/SIM-ERR/COMPILE-ERR 计数、8 类变异覆盖矩阵、至今抓出的 bug 清单）详见 [`../../docs/C1-完成度审计.md`](../../docs/C1-完成度审计.md) §二。

`py -3.13` 是本机装了 numpy 的解释器（PATH 上的 `python`=3.12 无 numpy）。退出码非 0 表示有 case 失败，可进 CI。

## ⚠️ 这套 oracle 的边界（别过度信任）

- 它验证的是"**编译器是否产出了 PTX 源意图的计算 + 合法控制流**"，**不是**官方隐藏 golden 的
  bit-exact 替身。sim 语义与真 golden 若有出入，sim 通过≠官方通过。
- FP：FMA 单次舍入（float64 中间量），MAD 两次舍入 → 若 `mad.f32` 误映射成 MAD 会被抓出。
- 整数 ADD/SUB/MUL 对任意整型/位型都按 32-bit 回绕（让现脚手架的 `ADD.b64` 指针运算能跑）；
  `--strict` 会把 ISA 非法类型标红。
- **`BRX` 要求 active lane 条件一致**：发散分支（未谓词化的越界 guard）会 raise
  ExecError —— 这是用来抓"越界 guard 必须 if-conversion 谓词化"的。DIVERGE 计数见
  [`../../docs/C1-完成度审计.md`](../../docs/C1-完成度审计.md) §二。
- 单 warp 顺序执行、无 atomics/barrier 建模（公开 kernel 用不到）。

## 怎么加新 case / 用于回归

在 `cases.py` 加一个函数返回 `dict(ptx=, grid=, block=, param=, gmem=, out=(off,count,dtype), ref=)`，
注册到 `ALL`。改任何 lowering / pass 后跑 `py -3.13 dogfood.py` 即可回归。建议每实现一个
Phase（见 [../C1_实现流程分析.md](../C1_实现流程分析.md) §4）就把对应 case 从 FAIL 推到 PASS。
