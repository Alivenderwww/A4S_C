// encoder.cpp - Bit-exact AEC instruction encoder + golden self-test.
//
// Implements the Track-B AEC Precise ISA (spec.md §A.1 opcodes, §3 layout /
// Pred/Ctrl, §4 types, §5 operand placement). Verified against the Track-B
// aec_cases program.bin vectors by selfTest().
#include "aec/isa.h"

#include <cstdint>
#include <cstdio>

namespace aec {
namespace isa {

static const uint16_t kTypeShift   = 3u;
static const uint16_t kFamilyShift = 8u;
static const uint16_t kSpaceShift  = 11u;
static const uint16_t kTypeMask    = 0x000fu;
static const uint16_t kFamilyMask  = 0x0007u;
static const uint16_t kSpaceMask   = 0x0007u;

// Track-B §4.1: these opcodes take type .none; their [6:3] field is 0xf.
static bool isTypeless(Op op) {
  switch (op) {
    case Op::LOADI: case Op::LOADI64: case Op::BR: case Op::BRX:
    case Op::JMP: case Op::CALL: case Op::RET: case Op::HALT:
    case Op::SSYNC: case Op::SYNC_CT: case Op::SYNC_WG: case Op::MBAR:
    case Op::VOTE: case Op::MTCH: case Op::RDTSC: case Op::RDPMC:
      return true;
    default:
      return false;
  }
}

bool usesImmediate(Op op, uint32_t /*memory_space*/) {
  return op == Op::LOADI || op == Op::LOADI64 || op == Op::BR ||
         op == Op::BRX || op == Op::CALL || op == Op::SSYNC ||
         op == Op::RDPMC;
}

// Tensor precision mode (Pred/Ctrl [10:8]); mode 7 uses the extended selector.
uint8_t tensorModeForType(Type t) {
  switch (t) {
    case Type::F32:  return 0;
    case Type::F16:  return 1;
    case Type::BF16: return 2;
    case Type::S8:   return 3;
    case Type::F64:
    case Type::S32:  return 7;
    default:         return 0xffu;
  }
}

uint8_t tensorExtendedModeForType(Type t) {
  if (t == Type::F64) return 0;
  if (t == Type::S32) return 1;
  return 0;
}

Word128 encode(const Fields &f) {
  Word128 w;
  uint16_t pred_ctrl = 0;

  // Type field [6:3]: the value for typed ops, 0xf (.none) for typeless ones.
  const Type ty = isTypeless(f.op) ? Type::NONE : f.type;
  pred_ctrl |= static_cast<uint16_t>(
      (static_cast<uint16_t>(ty) & kTypeMask) << kTypeShift);

  if (f.op == Op::BRX) {
    // BRX always names its branch predicate in bits [2:0], no enable bit.
    pred_ctrl |= static_cast<uint16_t>(f.predicate & 0x7u);
  } else if (f.predicate != kPredicateNone) {
    pred_ctrl |= kPredEnable | static_cast<uint16_t>(f.predicate & 0x7u);
  }

  if (f.op == Op::CMP || f.op == Op::CMPP) {
    pred_ctrl |= static_cast<uint16_t>((f.modifier & kFamilyMask) << kFamilyShift);
  } else if (f.op == Op::LD || f.op == Op::ST) {
    pred_ctrl |= static_cast<uint16_t>((f.modifier & kSpaceMask) << kSpaceShift);
  } else if (f.op == Op::TMUL || f.op == Op::TMUL_S) {
    const uint8_t mode = tensorModeForType(f.type);
    pred_ctrl |= static_cast<uint16_t>((mode & kFamilyMask) << kFamilyShift);
    if (mode == 7) {
      pred_ctrl |= static_cast<uint16_t>(tensorExtendedModeForType(f.type) << 11);
    }
  } else if (f.op == Op::TLDA || f.op == Op::TSTA) {
    pred_ctrl |= static_cast<uint16_t>((f.modifier & kFamilyMask) << kFamilyShift);
  } else if (f.op == Op::CVTFF || f.op == Op::CVTFI ||
             f.op == Op::CVTIF || f.op == Op::CVTII) {
    // Track-B §5.3: destination type in [6:3] (f.type), source type in [13:10]
    // (f.modifier), [9:7]=0.
    pred_ctrl |= static_cast<uint16_t>((f.modifier & 0xfu) << 10);
  } else if (f.op == Op::MBAR) {
    pred_ctrl |= static_cast<uint16_t>((f.modifier & 0x3u) << kFamilyShift);
  }

  const bool imm = usesImmediate(f.op, f.modifier);
  w.word0 = imm ? f.imm : static_cast<uint32_t>(f.src3);
  w.word1 = static_cast<uint32_t>(f.src2);
  w.word2 = (static_cast<uint32_t>(f.dst) << 16) | static_cast<uint32_t>(f.src1);
  w.word3 = (static_cast<uint32_t>(f.op) << 16) | pred_ctrl;
  return w;
}

const char *opName(Op op) {
  switch (op) {
    case Op::ADD: return "ADD"; case Op::SUB: return "SUB";
    case Op::MUL: return "MUL"; case Op::MAD: return "MAD";
    case Op::FMA: return "FMA"; case Op::DIV: return "DIV";
    case Op::NEG: return "NEG"; case Op::ABS: return "ABS";
    case Op::MIN: return "MIN"; case Op::MAX: return "MAX";
    case Op::AND: return "AND"; case Op::OR:  return "OR";
    case Op::XOR: return "XOR"; case Op::NOT: return "NOT";
    case Op::SHL: return "SHL"; case Op::SHR: return "SHR";
    case Op::BFX: return "BFX"; case Op::BINS: return "BINS";
    case Op::POPC: return "POPC"; case Op::FLO: return "FLO";
    case Op::CMP: return "CMP"; case Op::CMPP: return "CMPP";
    case Op::SEL: return "SEL"; case Op::PICK: return "PICK";
    case Op::LD:  return "LD";  case Op::ST:  return "ST";
    case Op::LDC: return "LDC"; case Op::ATOM: return "ATOM";
    case Op::BR:  return "BR";  case Op::BRX: return "BRX";
    case Op::JMP: return "JMP"; case Op::CALL: return "CALL";
    case Op::RET: return "RET"; case Op::HALT: return "HALT";
    case Op::SSYNC: return "SSYNC"; case Op::SYNC_CT: return "SYNC_CT";
    case Op::SYNC_WG: return "SYNC_WG"; case Op::MBAR: return "MBAR";
    case Op::LOADI: return "LOADI"; case Op::CPY: return "CPY";
    case Op::LOADI64: return "LOADI64"; case Op::CVTFF: return "CVTFF";
    case Op::CVTFI: return "CVTFI"; case Op::CVTIF: return "CVTIF";
    case Op::CVTII: return "CVTII"; case Op::SHUF: return "SHUF";
    case Op::VOTE: return "VOTE"; case Op::MTCH: return "MTCH";
    case Op::TMUL: return "TMUL"; case Op::TMUL_S: return "TMUL_S";
    case Op::TLDA: return "TLDA"; case Op::TSTA: return "TSTA";
    case Op::TMOV: return "TMOV"; case Op::TDUP: return "TDUP";
    case Op::RCP: return "RCP"; case Op::RSQ: return "RSQ";
    case Op::SIN: return "SIN"; case Op::COS: return "COS";
    case Op::EXP: return "EXP"; case Op::LOG: return "LOG";
    case Op::SQRT: return "SQRT"; case Op::RDTSC: return "RDTSC";
    case Op::RDPMC: return "RDPMC";
  }
  return "?";
}

const char *typeName(Type t) {
  switch (t) {
    case Type::B32: return "b32"; case Type::B64: return "b64";
    case Type::U32: return "u32"; case Type::S32: return "s32";
    case Type::U8: return "u8"; case Type::S8: return "s8";
    case Type::F32: return "f32"; case Type::F64: return "f64";
    case Type::F16: return "f16"; case Type::BF16: return "bf16";
    case Type::NONE: return "";
  }
  return "?";
}

// --- Golden self-test -----------------------------------------------------
namespace {
struct Golden {
  const char *name;
  Word128     expect;
  Fields      fields;
};
Word128 W(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  Word128 w; w.word0 = w0; w.word1 = w1; w.word2 = w2; w.word3 = w3; return w;
}
} // namespace

bool selfTest() {
  Golden g[8];

  // Vectors are the encoded instructions of Track-B aec_cases/cvtff, decoded
  // from its program.bin. w = [w0, w1, w2, w3].

  // LOADI.none R10, 0x100
  g[0].name = "LOADI.none R10,0x100";
  g[0].expect = W(256, 0, 655360, 5570680);
  g[0].fields.op = Op::LOADI; g[0].fields.dst = 10; g[0].fields.imm = 0x100;

  // CPY.u32 R1, %laneid
  g[1].name = "CPY.u32 R1,%laneid";
  g[1].expect = W(0, 0, 65796, 5505040);
  g[1].fields.op = Op::CPY; g[1].fields.type = Type::U32;
  g[1].fields.dst = 1; g[1].fields.src1 = LANEID;

  // MUL.u32 R3, R1, R2
  g[2].name = "MUL.u32 R3,R1,R2";
  g[2].expect = W(0, 2, 196609, 196624);
  g[2].fields.op = Op::MUL; g[2].fields.type = Type::U32;
  g[2].fields.dst = 3; g[2].fields.src1 = 1; g[2].fields.src2 = 2;

  // ADD.u32 R10, R10, R3
  g[3].name = "ADD.u32 R10,R10,R3";
  g[3].expect = W(0, 3, 655370, 65552);
  g[3].fields.op = Op::ADD; g[3].fields.type = Type::U32;
  g[3].fields.dst = 10; g[3].fields.src1 = 10; g[3].fields.src2 = 3;

  // CVTIF.f32.u32 R4, R1  (dst type f32 in [6:3], src type u32 in [13:10])
  g[4].name = "CVTIF.f32.u32 R4,R1";
  g[4].expect = W(0, 0, 262145, 5376064);
  g[4].fields.op = Op::CVTIF; g[4].fields.type = Type::F32;
  g[4].fields.dst = 4; g[4].fields.src1 = 1;
  g[4].fields.modifier = static_cast<uint32_t>(Type::U32);

  // CVTFF.f16.f32 R8, R4
  g[5].name = "CVTFF.f16.f32 R8,R4";
  g[5].expect = W(0, 0, 524292, 5251152);
  g[5].fields.op = Op::CVTFF; g[5].fields.type = Type::F16;
  g[5].fields.dst = 8; g[5].fields.src1 = 4;
  g[5].fields.modifier = static_cast<uint32_t>(Type::F32);

  // ST.gmem.u32 [R10], R8
  g[6].name = "ST.gmem.u32 [R10],R8";
  g[6].expect = W(0, 8, 10, 3211280);
  g[6].fields.op = Op::ST; g[6].fields.type = Type::U32;
  g[6].fields.src1 = 10; g[6].fields.src2 = 8;
  g[6].fields.modifier = static_cast<uint32_t>(Space::GMEM);

  // HALT
  g[7].name = "HALT";
  g[7].expect = W(0, 0, 0, 4522104);
  g[7].fields.op = Op::HALT;

  bool ok = true;
  for (int i = 0; i < 8; ++i) {
    Word128 got = encode(g[i].fields);
    if (!(got == g[i].expect)) {
      ok = false;
      std::fprintf(stderr,
          "[selftest] MISMATCH %s\n  expect w=[%u,%u,%u,%u]\n  got    w=[%u,%u,%u,%u]\n",
          g[i].name, g[i].expect.word0, g[i].expect.word1,
          g[i].expect.word2, g[i].expect.word3, got.word0, got.word1,
          got.word2, got.word3);
    }
  }
  if (ok) std::fprintf(stderr, "[selftest] all 8 golden vectors match\n");
  return ok;
}

} // namespace isa
} // namespace aec
