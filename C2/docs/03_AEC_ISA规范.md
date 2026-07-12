# AEC-C2 ISA 规范

本赛道使用 Track B 的 AEC ISA 编码。基线为 `simple-gpgpu` 仓库 `main` 分支，冻结提交见
`golden/b_isa_public.json`。C2 不重新编号 Track B 指令。

学生实现 Runtime 和 Agent，不需要生成或修改 kernel image。组织方提供的 device 会执行固定
image；本文件用于说明 Runtime 与 device 之间的二进制边界。

## 1. 指令格式

每条指令 128 bit，按 16 字节对齐，保存为四个 little-endian `uint32_t`：

```text
bits 127:112  Opcode
bits 111:96   Pred/Ctrl
bits 95:80    Dest
bits 79:64    Src1
bits 63:32    Src2 或指令专用字段
bits 31:0     Imm32 或 Src3
```

对应 `aecIsaInstruction`：

```text
word3 = Opcode:u16 | Pred/Ctrl:u16
word2 = Dest:u16   | Src1:u16
word1 = Src2/专用字段
word0 = Imm32/Src3
```

当前 register file 为 256×32 bit。64-bit 值使用 `R[n]:R[n+1]`，低 32 bit 在前。

## 2. Pred/Ctrl

| Bit | 含义 |
|---:|---|
| 15 | 普通指令的 predication enable |
| 14:11 | TMUL mode 7 的扩展精度 selector |
| 12:11 | LD/ST memory space |
| 10:8 | CMP operation、TMUL mode 或 tensor layout |
| 6:3 | data type |
| 2:0 | predicate P0–P7 |

`BRX` 总是在 bit 2:0 指定分支 predicate，不设置 bit 15。未分配位必须为 0。

比较 selector：`eq=0, ne=1, lt=2, le=3, gt=4, ge=5`。

memory space：`gmem=0, smem=1, cmem=2, lmem=3`。C2 另外使用 `pmem=4` 读取只读
parameter block；这是 C2 扩展，不改变 Track B 已分配值。

## 3. Opcode

| 范围 | 指令 |
|---|---|
| `0x0001–0x000a` | ADD–MAX |
| `0x0010–0x0019` | AND–FLO |
| `0x0020–0x0023` | CMP–PICK |
| `0x0030–0x0033` | LD、ST、LDC、ATOM |
| `0x0040–0x0049` | BR–MBAR |
| `0x0050–0x0059` | LOADI–MTCH |
| `0x0060–0x0065` | TMUL–TDUP |
| `0x0070–0x0076` | RCP–SQRT |
| `0x0080–0x0081` | RDTSC、RDPMC |

完整编号见 `include/aec_isa.h`。`0x0000` 不是 C2 NOP。固定 image 需要延迟填充时使用
`CPY.u32 R255,R255`。

## 4. Type selector

```text
0 f32      1 f64      2 f16      3 bf16
4 f8e4m3   5 f8e5m2   6 f4e2m1   7 s32
8 u32      9 s8      10 u8      11 s4
12 u4     13 b32     14 b64
```

注意：`aecDataType` 是 Runtime API enum，数值不等于 ISA type selector。转换规则由
`aecIsaTypeForDtype()` 给出。

## 5. 固定 image 使用的指令形式

| 指令 | 操作数 |
|---|---|
| `LOADI` | Dest=Rd，word0=imm32 |
| `CPY Rd,%special` | special selector 放在 Src1 |
| `LD.gmem Rd,[Ra]` | Src1=64-bit address pair |
| `ST.gmem [Ra],Rs` | Dest=0，Src1=address pair，Src2=source |
| `CMP/CMPP` | operation 在 Pred/Ctrl[10:8] |
| `BR/BRX` | word0=absolute instruction-index target |
| `MAD/FMA/TMUL` | Src2 在 word1，Src3 在 word0 |

special selector：

```text
0x0100 tid.x    0x0101 ntid.x    0x0102 ctaid.x    0x0103 nctaid.x
0x0104 laneid   0x0105 warpid
0x0110..0x0113  对应 y 分量
0x0120..0x0123  对应 z 分量
```

Vector Add 和 AXPY 使用 `%tid.x`、`%ctaid.x`、`%ntid.x` 计算线性 x 索引，不存在
自定义 `GLOBAL_TID` selector。

## 6. Tensor 形式

固定 GEMM image 使用：

```text
TLDA.type R32,[R0]
TLDA.type R48,[R2]
TMUL.type R64,R32,R48,R64
TSTA.type [R4],R64
```

编码要求：

- TLDA：Dest=tile destination，Src1=address register，layout=0。
- TMUL：Dest/Src1/Src2/Src3 分别放入四个 operand 字段；精度 mode 在 bit 10:8。
- TSTA：Dest=0，Src1=address register，Src2=tile source，layout=0。
- naive/tiled/vectorized variant 位于 image header，不写入指令字段。

TMUL mode：`f32=0, f16=1, bf16=2, s8=3, s4=4, f8e4m3=5, f4e2m1=6`。
mode 7 使用扩展 selector：`f64=0, s32=1`；C2 为 `f8e5m2` 使用保留值 2。

## 7. Golden encoding

`ADD.f32 @P3 R1,R2,R3,R4`：

```text
word0 = 0x00000004
word1 = 0x00000003
word2 = 0x00010002
word3 = 0x00018003
```

`CPY.u32 R1,%tid.x`：

```text
word0 = 0x00000000
word1 = 0x00000000
word2 = 0x00010100
word3 = 0x00510040
```

更多固定向量见 `golden/b_isa_public.json`，可运行 `examples/02_isa_encoding.c` 检查本地
头文件和编译器环境。
