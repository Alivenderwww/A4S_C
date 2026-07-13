// dce.cpp - Dead code elimination.  Scoring category: T2.
//
// Counts uses of every virtual register across the function and drops any
// instruction whose register result is never used and has no side effect,
// iterating to a fixpoint (removing a dead def can make its sources dead).
// Loads/stores/barriers are kept regardless (a load may fault). This also
// cleans up the definitions that CSE/const-prop leave unreferenced.
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
  bool changedAny = false;

  // Iterate to a fixpoint: removing a dead def can make its sources dead too.
  bool changed = true;
  while (changed) {
    changed = false;

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

    // Drop any instruction whose Reg destination is never used, has no side
    // effect, and is not a memory read (loads may fault -> keep conservatively).
    for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
      ir::BasicBlock &b = fn.blocks[bi];
      std::vector<ir::Inst> kept;
      kept.reserve(b.insts.size());
      for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
        const ir::Inst &in = b.insts[ii];
        std::map<uint32_t, int>::iterator it =
            (in.dst.kind == ir::Operand::Reg) ? useCount.find(in.dst.value)
                                              : useCount.end();
        int uc = (it != useCount.end()) ? it->second : 0;
        bool removable = in.dst.kind == ir::Operand::Reg && uc == 0 &&
                         !hasSideEffect(in) && in.op != ir::Op::LD &&
                         in.op != ir::Op::LDC;
        if (removable) { changed = true; changedAny = true; }
        else kept.push_back(in);
      }
      b.insts.swap(kept);
    }
  }

  return changedAny;
}

} // namespace passes
} // namespace aec
