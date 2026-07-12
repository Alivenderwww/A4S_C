# C2 固定 kernel image 研究（反汇编全部 34 个）

> 把 C2 starter-kit 的 34 个固定 kernel image 全部反汇编研究。完整反汇编见
> [C2_images_disasm.txt](C2_images_disasm.txt)。对 **C2**（kernel_agent 选型）和
> **C1**（T5 TMUL 参照 + 约定校验）都有直接价值。

## 0. 编目

| semantic_kernel_id | kernel | 数量 | 说明 |
|---|---|---:|---|
| 1 | vector_add_f32 | 1 | 标量 |
| 10 | gemm_naive_dtype_1..10 | 10 | 每 dtype 一个 |
| 11 | gemm_tiled_dtype_1..10 | 10 | |
| 12 | gemm_vectorized_dtype_1..10 | 10 | |
| 20/21/22 | axpy / dot / nrm2 (f32) | 3 | 向量运算 |

**共 34 = 1 vadd + 30 GEMM(3 变体 × 10 dtype)+ 3 向量。**

## 1. ✅ 没有任何未知代码

全部 34 个 image 的每一条指令,opcode 都在我们已知的表内(`aec_isa.h` 编号)。**我们的解码器 100% 覆盖,ISA 理解无盲区。** 用到的 opcode:
`CPY LD LOADI ADD CMPP BRX TLDA HALT TMUL TSTA MUL ST BR SQRT`。
(CPY 大量出现主要是 `CPY R255,R255` 的 NOP 延迟填充。)

## 2. GEMM 核心结构(三变体完全一致)

所有 GEMM(naive/tiled/vectorized、所有 dtype)**核心序列相同**:

```text
; 参数(LD.pmem,自然偏移)
 0-2: LD.pmem.b64 R0=A_ptr, R2=B_ptr, R4=C_ptr      ; b64 pair (R0:R1 等)
 3-6: LD.pmem.u32 R6=M@0x18, R7=N@0x1c, R8=K@0x20, R9=?@0x24
 7-10: LOADI R10=0(m), R11=0(n), R12=0(k), R13=16   ; tile step = 16
LOOP(=11):
  11: TLDA.<ty> R32, [R0]        ; 载入 A tile 到 R32..
  12: TLDA.<ty> R48, [R2]        ; 载入 B tile 到 R48..
  13: TMUL.<ty> R64, R32, R48, R64  ; C_acc(R64) += A_tile × B_tile
      (... NOP 延迟填充 ...)
  +0: ADD R12 += 16;  CMPP.lt R12<K;  BRX -> 11   ; K 维 tile 循环
  +3: TSTA.<ty> [R4], R64        ; 存 C tile
  +4: R12=0; ADD R11 += 16; CMPP.lt R11<N; BRX -> 11   ; N 维循环
  +8: R11=0; ADD R10 += 16; CMPP.lt R10<M; BRX -> 11   ; M 维循环
  HALT
```

**寄存器约定**:R0/R2/R4=A/B/C 基址(b64 对);R6/R7/R8=M/N/K;R9=第 4 个参数;
R10/R11/R12=M/N/K 循环计数器(步长 16);R13=16;**R32=A tile / R48=B tile / R64=C 累加器**。
**tile 尺寸 = 16×16。dtype → `TMUL.type` 直接对应**(f4e2m1/f16/f32/bf16/s8/…)。

## 3. 🔑 变体差异 = 只有 NOP 数量(对 C2 是实锤)

| 变体 | 指令数 | NOP(CPY R255) | 核心计算 |
|---|---:|---:|---|
| naive | 91 | 64 | 相同 |
| tiled | 43 | 16 | 相同 |
| **vectorized** | **27** | **0** | 相同 |

三个变体**算出的结果完全一样**,区别只是 NOP 延迟填充的多少(建模不同的执行延迟)。

**→ C2 行动项:`kernel_agent` 在候选 image 里应优先选 `vectorized`(0 NOP = 虚拟周期最少)。** 这直接影响 C2 的 Agent 性能分。选型逻辑:同 (dtype, shape) 下,variant=3(vectorized)最快。

## 4. ⚠️ TMUL 的 tile 地址语义仍不透明(T5 关键)

**地址寄存器 R0/R2/R4 全程不推进** —— K/N/M 三个循环只推进计数器 R10/R11/R12,`TLDA/TMUL/TSTA` 永远引用同一个基址 R0/R2/R4。所以:
- **指令里看不到"载入哪一块 tile"** —— 设备一定是用**循环计数器 R10/R11/R12 当作 tile 坐标**,通过某个**内部约定**去算 `base + offset`。这个约定(哪几个寄存器是坐标、tile 在内存里的排布、fragment 到 lane 的映射)**没有编码在指令里,也没写在公开文档里**。
- **结论**:光靠反汇编**无法完全还原** TMUL 语义。要彻底搞定,只能在 **WSL 上用 C2 设备黑盒实跑**这些 image(给已知 A/B,看输出 C),反推 tile→内存映射;或直接问组委会。
- **但收获巨大**:我们现在有了**精确的指令骨架 + 寄存器约定 + tile 尺寸(16)**。若 C1 golden 与 C2 设备同源(都来自 simple-gpgpu),照这个骨架发码 + 用 R10/R11/R12 当坐标,**很可能能跑对** —— 只是没设备就没法本地验证,仍不建议盲赌 16 正确性分。

## 5. 对 C1 的交叉验证(顺带确认)

- ✅ 所有 image 以 **HALT** 结尾 —— 印证我们把 kernel 出口从 RET 改成 HALT 是对的。
- ✅ 参数用 **LD.pmem 自然偏移**(a@0, b@8, c@16, 标量依次往后)—— 和我们 C1 编译器一致。
- ✅ 地址运算用 **ADD.b64 + LD.pmem.b64 pair**(我们收窄成 32 位,算出的地址相同,也正确)。
- ⚠️ GEMM 有**第 4 个 u32 参数**(@0x24,超出 M/N/K)—— 疑似 leading dimension/stride 或标志位,属 C2 kernel 的 ABI;C1 的 PTX-05 只有 M/N/K 三个。

## 6. 小结

| 问题 | 答案 |
|---|---|
| 有未知代码吗? | **没有**,解码器全覆盖 |
| TMUL 怎么用? | TLDA(A)+TLDA(B)+TMUL(累加) 循环 K,TSTA 存,循环 N/M;tile 16×16 |
| 变体区别? | **只有 NOP 数量**(vectorized 最快)→ C2 kernel_agent 选 vectorized |
| TMUL tile 语义能反汇编出来吗? | **不能完全** —— 地址靠设备内部约定 + 循环计数器,需 WSL 实跑或问组委会 |
| 对 C1 T5 的帮助 | 有了精确骨架 + 寄存器约定;但没设备仍无法验证,不建议盲赌 |
