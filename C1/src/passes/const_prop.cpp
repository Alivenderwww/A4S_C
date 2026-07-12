// const_prop.cpp - Constant propagation / folding.  Scoring category: T2.
//
// STATUS: wired identity stub. The scaffolding walks the function and builds
// the LOADI-defined constant map that a real implementation would consult;
// it deliberately makes no rewrites yet (returns false = no change).
#include "aec/passes.h"

#include <map>

namespace aec {
namespace passes {

bool constProp(ir::Function &fn, const Options &opt) {
  if (!opt.const_prop) return false;
  bool changed = false;

  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];

    // Constants currently known in this block: vreg -> immediate value.
    std::map<uint32_t, uint32_t> constOf;

    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst &in = b.insts[ii];

      // Record LOADI destinations as known constants (block-local).
      if (in.op == ir::Op::LOADI && in.hasImm && in.dst.kind == ir::Operand::Reg) {
        constOf[in.dst.value] = in.imm;
      }

      // TODO(T2): when both sources of ADD/SUB/MUL/AND/... are in `constOf`,
      // fold the instruction to a LOADI of the computed value and set changed.
      // TODO(T2): substitute constant operands into arithmetic to expose
      // further DCE/CSE. Invalidate constOf entries on redefinition.
      (void)in;
    }
  }

  return changed;
}

} // namespace passes
} // namespace aec
