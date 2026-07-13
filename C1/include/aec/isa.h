// isa.h - AEC ISA constants + 128-bit instruction encoder/decoder.
//
// Encodes the AEC machine code defined by the C1 spec: opcode table (§4),
// 128-bit layout + Pred/Ctrl fields (§5.1/§5.2), type selectors (§5.3),
// memory space (§5.4) and special-register selectors (§5.5). The numbering is
// independently cross-checked bit-exact against a reference AEC program.bin by
// aec::isa::selfTest().
#ifndef AEC_ISA_H
#define AEC_ISA_H

#include <cstdint>
#include <string>

namespace aec {
namespace isa {

// --- Opcodes (bits 127:112). C1 spec §4 defines the 18 emitted for the PTX
//     subset; the rest model the broader AEC ISA for the local dev harness. --
enum class Op : uint16_t {
  ADD = 0x0001, SUB = 0x0002, MUL = 0x0003, MAD = 0x0004, FMA = 0x0005,
  DIV = 0x0006, NEG = 0x0007, ABS = 0x0008, MIN = 0x0009, MAX = 0x000a,

  AND = 0x0010, OR = 0x0011, XOR = 0x0012, NOT = 0x0013, SHL = 0x0014,
  SHR = 0x0015, BFX = 0x0016, BINS = 0x0017, POPC = 0x0018, FLO = 0x0019,

  CMP = 0x0020, CMPP = 0x0021, SEL = 0x0022, PICK = 0x0023,

  LD = 0x0030, ST = 0x0031, LDC = 0x0032, ATOM = 0x0033,

  BR = 0x0040, BRX = 0x0041, JMP = 0x0042, CALL = 0x0043, RET = 0x0044,
  HALT = 0x0045, SSYNC = 0x0046, SYNC_CT = 0x0047, SYNC_WG = 0x0048,
  MBAR = 0x0049,

  CVTFF = 0x0050, CVTFI = 0x0051, CVTIF = 0x0052, CVTII = 0x0053,
  CPY = 0x0054, LOADI = 0x0055, LOADI64 = 0x0056, SHUF = 0x0057,
  VOTE = 0x0058, MTCH = 0x0059,

  TMUL = 0x0060, TMUL_S = 0x0061, TLDA = 0x0062, TSTA = 0x0063,
  TMOV = 0x0064, TDUP = 0x0065,

  RCP = 0x0070, RSQ = 0x0071, SIN = 0x0072, COS = 0x0073, EXP = 0x0074,
  LOG = 0x0075, SQRT = 0x0076,

  RDTSC = 0x0080, RDPMC = 0x0081
};

// --- Type selectors (Pred/Ctrl bits 6:3), C1 spec §5.3. -------------------
enum class Type : uint8_t {
  B32 = 0x0, B64 = 0x1, U32 = 0x2, S32 = 0x3, U8 = 0x4, S8 = 0x5,
  F32 = 0x8, F64 = 0x9, F16 = 0xa, BF16 = 0xb, NONE = 0xf
};

// --- Memory space (Pred/Ctrl bits 13:11), C1 spec §5.4. -------------------
enum class Space : uint8_t {
  GMEM = 0, SMEM = 1, CMEM = 2, LMEM = 3, PMEM = 4
};

// --- Compare operation (Pred/Ctrl bits 10:8 for CMP/CMPP). ----------------
enum class Cmp : uint8_t {
  EQ = 0, NE = 1, LT = 2, LE = 3, GT = 4, GE = 5
};

// --- Special-register selectors (placed in the Src1 field). ---------------
enum SpecialReg : uint16_t {
  TID_X = 0x0100, NTID_X = 0x0101, CTAID_X = 0x0102, NCTAID_X = 0x0103,
  LANEID = 0x0104, WARPID = 0x0105,
  TID_Y = 0x0110, NTID_Y = 0x0111, CTAID_Y = 0x0112, NCTAID_Y = 0x0113,
  TID_Z = 0x0120, NTID_Z = 0x0121, CTAID_Z = 0x0122, NCTAID_Z = 0x0123
};

static const uint8_t  kPredicateNone = 15;
static const uint16_t kPredEnable    = 0x8000u;

// A single encoded 128-bit instruction: four little-endian words.
//   word3 = Opcode:16 | Pred/Ctrl:16
//   word2 = Dest:16   | Src1:16
//   word1 = Src2 or instruction-specific field
//   word0 = Imm32 or Src3
struct Word128 {
  uint32_t word0 = 0;
  uint32_t word1 = 0;
  uint32_t word2 = 0;
  uint32_t word3 = 0;
  bool operator==(const Word128 &o) const {
    return word0 == o.word0 && word1 == o.word1 &&
           word2 == o.word2 && word3 == o.word3;
  }
};

// Flat field bundle handed to encode(). Mirrors aecIsaEncode() arguments.
struct Fields {
  Op       op       = Op::RET;
  Type     type     = Type::NONE;
  uint8_t  predicate = kPredicateNone; // guarding predicate (or BRX branch pred)
  bool     pred_neg  = false;          // negate the predicate (Pred/Ctrl [14]).
  uint16_t dst      = 0;
  uint16_t src1     = 0;
  uint16_t src2     = 0;
  uint16_t src3     = 0;
  uint32_t imm      = 0;
  uint32_t modifier = 0; // cmp op / mem space / cvt source type, per opcode.
};

// True if word0 carries an imm32 instead of Src3 for this opcode/space.
bool usesImmediate(Op op, uint32_t memory_space);

// Encode one instruction. Bit-exact with the C1 spec §5 layout.
Word128 encode(const Fields &f);

// Human-readable mnemonic for an opcode (used by aec-objdump).
const char *opName(Op op);
const char *typeName(Type t);

// Decode just the opcode from an encoded word (for the disassembler).
inline Op decodeOp(const Word128 &w) {
  return static_cast<Op>(static_cast<uint16_t>(w.word3 >> 16));
}

// Verify the encoder against the 8 public golden vectors. Returns true on a
// bit-exact match for every vector; on failure prints a diff to stderr.
bool selfTest();

} // namespace isa
} // namespace aec

#endif // AEC_ISA_H
