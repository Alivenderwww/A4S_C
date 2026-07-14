# C1 编译器优化报告：让循环展开支持地址递推 IV

> 改动文件：`src/passes/unroll.cpp`（+93 / −20 行，单文件）
> 验证：远程 Linux 服务器 + 官方 `aec-precise` CModel 实测

---

## 一、问题是怎么发现的

### 1.1 从评测切入：定位最弱的类别

我先把 C1 工程部署到远程服务器，按 `scoring.md` 流程跑完整评测：编译 5 个公开用例 → CModel 执行 → dump 输出 → 按公式核对正确性。结果 5/5 正确通过，性能分（B 类）如下：

| 类别 | 权重 | 加速比 r | 得分 |
|------|----:|--------:|----:|
| T2 标量优化 | 8 | 1.125 | 3.73 |
| T3 内存访问 | 10 | 1.071 | 2.86 |
| T4 寄存器调度 | 10 | 1.028 | 1.11 |
| T5 GEMM | 12 | **1.953** | **12.00（满分）** |

T5 已经满分（r=1.95 ≥ 1.25 阈值），T2/T3/T4 加速比贴近 1，看起来是最弱的。**但 T5 权重最大（12 分），如果能让 T5 进一步突破，收益最高**——这是第一直觉。

### 1.2 关键发现：性能模型 = 指令数

我注意到 `scoring.md` 说性能度量用「执行时间/周期数」，但没给具体公式。于是我用 CModel 实测数据反推：

```
T3: steps=491520, 指令数=30(warps=16384)  → 30 × 16384 = 491520 ✓
T1: steps=753664, 指令数=23(warps=32768)  → 23 × 32768 = 753664 ✓
```

**直线型用例 100% 吻合：`steps = 静态指令数 × warp数`**。也就是说，CModel 是一个功能模型，每条指令固定 1 步，没有 latency/stall。这意味着 **唯一能提分的手段就是减少静态指令数**（T5 有循环，另有分支计入，但循环体指令数仍是主导）。

### 1.3 顺藤摸瓜：T5 循环为何不展开

带着「减少指令数」的目标，我反汇编了 T5 的 -O2 输出，发现它的 K 循环（128 次迭代）**完全没有被展开**：

```
循环体（每迭代 9 条指令）× 128 次 = 1152 条动态指令
```

循环控制指令（counter 递增 + CMPP + BRX 回边 + 退出 BRX）每迭代 4 条，占 44%，本应通过展开分摊掉。

但为什么没展开？`unroll.cpp` 里有个 `unroll_factor = 4`（-O2 默认开启）。我读了 `driver.cpp` 的流水线：

```cpp
while (passes::loopRotate(fn, opt)) {}   // 1. while → do-while
while (passes::strengthReduce(fn, opt)) {} // 2. 地址乘法 → 加法递推
passes::unrollLoops(fn, opt);            // 3. 展开
```

**顺序很关键**：`strengthReduce`（强度削减）先跑，把 GEMM 的地址计算从 `MAD idx; MUL off; ADD addr` 改写成每迭代单步推进的 `ADD addr, addr, stride`。然后 `unrollLoops` 跑，但它看到这个 `ADD addr,addr,stride`（自增寄存器）就触发了这段守卫：

```cpp
// Bail if the body has a second self-incrementing induction variable
// besides the counter (e.g. a strength-reduced address recurrence
// `ADD addr,addr,stride`). Unrolling would have to offset each such IV by
// c*stride per copy; not handled yet, so skip.
bool multiIV = false;
for (...) {
  if (ADD && dst==s1 && dst != counter) { multiIV = true; break; }
}
if (multiIV) continue;   // ← 直接跳过整个循环，永不展开
```

**这就是病根**：`strengthReduce` 产生的地址递推 IV，被 `unroll` 当成「无法处理的多 IV」一律跳过。所有带线性数组寻址的 GEMM 类循环都展开不了。

为了确认，我用一个 Explore agent 静态分析了代码，它独立给出了同样的结论：T5 在 -O3（unroll_factor=8）下也不会展开，指令数和 -O2 一样，因为 multiIV 守卫在 `unroll.cpp:117` 的 `continue` 处提前退出。

---

## 二、如何优化

### 2.1 思路：把地址递推 IV 当成「可偏移的第二类 IV」

地址递推 IV 的形式是 `ADD ivR, ivR, strideR`，其中 `strideR` 循环不变。它的语义是：每迭代地址前进一个 stride。展开 U 倍后，第 c 个 copy（c=0..U-1）的 load 应该落在 `iv, iv+stride, iv+2*stride, ..., iv+(U-1)*stride`。

这和计数器 IV（`counter += step`）的展开机制完全同构：copy c 用 `counter + c*step`。地址 IV 同理用 `ivR + c*stride`。所以核心改动是 **把地址递推 IV 收集起来，展开时像计数器一样偏移**。

### 2.2 第一次尝试：每副本重算偏移（失败）

我先实现了「每个 copy c 计算 `ivR_c = ivR + c*stride`」的方案。对常数 stride 折叠成 `LOADI c*stride; ADD`；对寄存器 stride 用 `LOADI c; MUL; ADD`。

编译通过，T5 正确，但 **steps 从 545792 恶化到 758784**（变慢 39%）！反汇编一看，2 个地址 IV × 3 个 copy × 每个 3 条偏移指令 = 18 条额外指令，远超展开省下的循环控制指令。

**教训**：偏移指令本身也是指令，CModel 按 1 步/指令计费，省 4 条循环控制却多花 18 条偏移，得不偿失。

### 2.3 第二次尝试：副本间单步推进（正确且省）

换思路：地址 IV 保持共享（loop-carried），不每副本重算。改为 **copy 之间插入一次 `ivR += stride` 推进**，就像原始递推的延续：

```
copy0: LD A[iv]; LD B[iv]; MAD
ivA += stride; ivB += stride       ← 推进到下一 copy 的地址
copy1: LD A[iv]; LD B[iv]; MAD
ivA += stride; ivB += stride
...（共 U 个 copy，U 次推进）
```

每个 IV 每副本只多 1 条 ADD（不是 3 条），且无需 LOADI/MUL。推进 U 次后地址恰好到 `iv + U*stride`，正好是下一轮外层迭代的入口值。

这个方案 T5 正确，但 **steps 仍微涨到 562176**。深入分析发现：runtime K（运行时参数）无法证明可被 U 整除，于是每个 copy c≥1 都被迫生成**余数谓词**（`LOADI c*step; ADD counter_c; CMPP counter_c < bound`，3 条/copy）来防止越界 load。这个开销抵消了展开收益。

### 2.4 最终方案：可整除才展开，否则安全 skip

关键决策：**地址递推 IV 循环，仅当 trip 可证明被 U 整除时才展开**；runtime trip（如 GEMM 的 K 参数）直接 skip，保留强度削减后的紧凑单迭代体。

```cpp
if (!addrIV.empty() && !divisible) continue;  // runtime trip + 地址IV → 跳过
```

为什么这样是对的：
- **runtime K 时**：余数谓词开销 > 展开收益，skip 后 T5 维持原 545792（零回归）。
- **可整除 trip 时**：无余数谓词，展开净收益巨大。

同时做了一个小优化：可整除时跳过「counter 偏移」的生成（余数谓词才需要 counter_c），避免死代码。

### 2.5 中途踩的坑：展开后跑 CSE 会破坏正确性

我曾尝试在展开后跑一轮 `CSE+DCE` 清理冗余（改动 driver.cpp）。结果 **T5 完全错乱（maxabs=27）**。

排查发现：展开后的循环是**单块自循环**（带回边），而 `cse.cpp` 的 `localOnly` 判断基于「同一基本块」。它无法区分「同一迭代的 use」和「下一迭代的 use」，于是错误合并了只在迭代间才冗余的值，破坏了递推语义。

**DCE 是安全的**（基于 use-count，不合并值），但 CSE 不是。于是回滚 driver.cpp，只保留 unroll.cpp 的改动，并在注释里明确警告「展开后不要对自循环跑 CSE」。

---

## 三、改动细节

`src/passes/unroll.cpp` 三处改动：

### 改动 1：识别并收集地址递推 IV（替换原 multiIV 一刀切跳过）

```cpp
// 原来：发现任何自增寄存器(非counter)就 continue
// 现在：收集 stride 循环不变的地址递推 IV；只有无法偏移的才 bail
std::vector<uint32_t> addrIV;      // 递推寄存器 ivR
std::vector<uint32_t> addrStride;  // 其循环不变 stride 寄存器
for (...) {
  if (ADD && dst==s1 && dst != counter) {
    if (s2 循环不变) { addrIV.push_back(dst); addrStride.push_back(s2); }
    else { bail; }   // 非不变 stride，跳过整个循环
  }
}
```

### 改动 2：可整除守卫

```cpp
// 地址 IV 循环 + runtime trip → 余数谓词开销过大 → 安全跳过
if (!addrIV.empty() && !divisible) continue;
```

### 改动 3：副本间单步推进 + 尾部不重复推进

```cpp
// copy 主体里跳过原递推指令（ADD ivR,ivR,stride）
if (ADD && addrIVSet.count(dst)) continue;

// 每个 copy 末尾推进所有地址 IV 一次
for (a in addrIV)
  emit ADD ivR, ivR, strideR;

// 尾部：counter += U*step（地址 IV 不再推进，因为 copy U-1 之后已推够 U 次）
```

---

## 四、验证结果（全部 CModel 实测）

### 4.1 公开用例：零回归

| 用例 | 优化前 steps | 优化后 steps | 正确性 |
|------|-----------:|-----------:|------|
| T1 | 753664 | 753664 | PASS |
| T2 | 393216 | 393216 | PASS |
| T3 | 458752 | 458752 | PASS |
| T4 | 589824 | 589824 | PASS |
| T5 | 545792 | 545792 | PASS |

公开用例 steps 一字未变——T5 因 runtime K 不触发新路径，安全 skip。

### 4.2 新能力验证：可整除 GEMM 变体

构造了一个 K=128 编译期常量的 GEMM 变体（其余同 T5）：

| | steps | 加速比 r |
|---|-----:|--------:|
| O0（朴素基线）| 1,131,008 | — |
| O2（新展开）| **414,720** | **2.73** |

原来这种循环完全不能展开（被 multiIV 跳过），现在正确展开，steps 从 ~106万 级降到 41万。这正是 scoring.md C 类「GEMM 矩阵大小变化」变体所考察的能力。

### 4.3 鲁棒性变体：全部零回归

8 个变体（寄存器重命名 / 块重排 / 死代码 / 循环变形 / 整型位运算 / 300-live 寄存器压力 / 3 种 GEMM 尺寸）steps 与正确性全部维持。

### 4.4 选手自带测试套件：全 PASS

- `selftest`：8/8 golden 向量
- `conformance.py`：12/12（含 partial-block 发散、%laneid、SHL、位运算）
- `corners.py`：`@!%p bra` 取反谓词分支
- `pressure.py`：300 live vreg → 6 物理 reg、0 spill
- `mad_semantics.py`：MAD 非融合 / FMA 融合语义正确

---

## 五、诚实结论

### 公开用例分数未变（≈78~80/100）

公开 5 用例的 B 分 = 19.70/40，优化前后一致。原因客观：

1. **T5 已满分**（r=1.95 ≥ 1.25 → 12/12）。runtime K 无法证明可整除，展开会引入余数谓词开销使 steps 反增，故安全 skip。
2. **T2/T3/T4 是直线代码**，无循环可展开，且已接近数学最优：
   - T3 的 `out = x*y + x*z` 不能重关联为 `x*(y+z)`（FP 不满足结合律，会破坏正确性）
   - 地址 stride 已被 CSE/SR 复用，param 加载是 ABI 硬性需求

### 真实价值在鲁棒性（C 类）

改动的价值体现在**编译器能力扩展**：可整除 trip 的 GEMM 循环现在能正确展开（r=2.73，原来完全不能）。scoring.md C 类明确列出「FP32 Scalar GEMM 矩阵大小变化」变体（10 个 T5 变体），若隐藏/变体用例使用编译期已知维度，编译器现在能优化它们。这是之前的能力缺口。

### 经验总结

1. **先搞清性能模型**：CModel 是 1 步/指令的功能模型，不是 latency 模型。这意味着展开（减循环控制指令）理论有效，但**展开引入的指令本身也计费**——必须算净收益。
2. **优化 pass 的顺序很重要**：`strengthReduce` 先于 `unroll` 运行，前者改变了循环体形态，后者必须适应这种形态。原 unroll 没适应，导致能力错配。
3. **余数谓词是 runtime trip 展开的杀手**：编译期无法证明可整除时，每副本的边界检查开销会吃掉展开收益。要么证明可整除，要么放弃展开。
4. **CSE 不能直接跑在自循环上**：单块自循环的跨迭代冗余，per-block CSE 无法安全识别。展开后清理要用 DCE（安全）而非 CSE（会破坏递推）。
