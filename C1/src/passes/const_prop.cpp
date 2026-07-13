// const_prop.cpp - Constant propagation / folding.  Scoring category: T2.
//
// Per block, tracks which registers hold a known constant (defined by an
// unpredicated LOADI). When both source operands of an integer/bitwise op are
// known constants, the op is folded to a LOADI of the computed value (AEC §6
// semantics: 32-bit modular ADD/SUB/MUL, bitwise AND/OR/XOR, masked SHL/SHR).
// The folded LOADI is itself a new constant, so folding cascades; the now-unused
// source LOADIs become dead and are removed by DCE. Float folding is skipped
// (kept bit-exact / rounding-safe by construction). This is the "propagation"
// the T2 category asks for and, with DCE, cleans up the dead/constant
// computation the robustness variants insert.
#include "aec/passes.h"

#include <cstdint>
#include <map>

namespace aec {
namespace passes {

namespace {

// A 32-bit integer/bit type whose constant value we can fold exactly.
bool isFoldableIntType(ir::Type t) {
  return t == ir::Type::U32 || t == ir::Type::S32 || t == ir::Type::B32;
}

// Fold a two-operand integer/bit op. Returns false if the op is not foldable.
bool foldBinary(ir::Op op, uint32_t a, uint32_t b, uint32_t &out) {
  uint64_t r;
  switch (op) {
    case ir::Op::ADD: r = (uint64_t)a + b; break;
    case ir::Op::SUB: r = (uint64_t)a - b; break;
    case ir::Op::MUL: r = (uint64_t)a * b; break;
    case ir::Op::AND: r = (uint64_t)(a & b); break;
    case ir::Op::OR:  r = (uint64_t)(a | b); break;
    case ir::Op::XOR: r = (uint64_t)(a ^ b); break;
    case ir::Op::SHL: r = (uint64_t)(a << (b & 31)); break;
    case ir::Op::SHR: r = (uint64_t)(a >> (b & 31)); break;  // logical (u32).
    default: return false;
  }
  out = (uint32_t)(r & 0xffffffffu);
  return true;
}

} // namespace

bool constProp(ir::Function &fn, const Options &opt) {
  if (!opt.const_prop) return false;
  bool changed = false;

  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];

    // Registers known constant in this block: vreg -> immediate value.
    std::map<uint32_t, uint32_t> constOf;

    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst &in = b.insts[ii];

      // Try to fold a two-operand integer op whose sources are both constants.
      if (isFoldableIntType(in.type) && !in.hasImm &&
          in.dst.kind == ir::Operand::Reg &&
          in.s1.kind == ir::Operand::Reg && in.s2.kind == ir::Operand::Reg) {
        std::map<uint32_t, uint32_t>::iterator c1 = constOf.find(in.s1.value);
        std::map<uint32_t, uint32_t>::iterator c2 = constOf.find(in.s2.value);
        uint32_t folded;
        if (c1 != constOf.end() && c2 != constOf.end() &&
            foldBinary(in.op, c1->second, c2->second, folded)) {
          in.op = ir::Op::LOADI;
          in.hasImm = true; in.imm = folded;
          in.s1 = ir::Operand(); in.s2 = ir::Operand(); in.s3 = ir::Operand();
          changed = true;
        }
      }

      // Update the known-constant map for this definition.
      if (in.dst.kind == ir::Operand::Reg) {
        uint32_t D = in.dst.value;
        bool pair = (in.type == ir::Type::B64 || in.type == ir::Type::F64);
        // A LOADI with no guard defines a stable constant; anything else (or a
        // predicated def) makes D non-constant from here on.
        if (in.op == ir::Op::LOADI && in.hasImm && in.guard < 0)
          constOf[D] = in.imm;
        else
          constOf.erase(D);
        if (pair) constOf.erase((D + 1) & 0xff);
      }
    }
  }

  return changed;
}

} // namespace passes
} // namespace aec
