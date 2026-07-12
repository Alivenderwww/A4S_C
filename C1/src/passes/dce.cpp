// dce.cpp - Dead code elimination.  Scoring category: T2.
//
// STATUS: wired identity stub. The scaffolding computes a use-count over
// virtual registers (what a real DCE consults) but performs no removal yet.
#include "aec/passes.h"

#include <map>

namespace aec {
namespace passes {

namespace {
bool hasSideEffect(const ir::Inst &in) {
  // Stores, branches, barriers, tensor stores and returns must never be
  // removed regardless of whether their (absent) dst is used.
  if (in.isTerminator()) return true;
  switch (in.op) {
    case ir::Op::ST: case ir::Op::TSTA: case ir::Op::ATOM:
    case ir::Op::SYNC_WG: case ir::Op::SYNC_CT: case ir::Op::SSYNC:
    case ir::Op::MBAR:
      return true;
    default:
      return false;
  }
}
} // namespace

bool dce(ir::Function &fn, const Options &opt) {
  if (!opt.dce) return false;
  bool changed = false;

  // Count uses of every virtual register across the whole function.
  std::map<uint32_t, int> useCount;
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst &in = b.insts[ii];
      const ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 3; ++k)
        if (srcs[k]->kind == ir::Operand::Reg)
          useCount[srcs[k]->value]++;
    }
  }

  // TODO(T2): iterate to a fixpoint removing instructions whose dst is a Reg
  // with zero uses and !hasSideEffect(in). Removing an instruction drops the
  // use counts of its sources, which may expose further dead defs.
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    for (unsigned ii = 0; ii < fn.blocks[bi].insts.size(); ++ii) {
      const ir::Inst &in = fn.blocks[bi].insts[ii];
      (void)in; (void)hasSideEffect; (void)useCount;
    }
  }

  return changed;
}

} // namespace passes
} // namespace aec
