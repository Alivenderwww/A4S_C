// mem_coalesce.cpp - GPGPU memory coalescing / load reuse.  Category: T3.
//
// STATUS: wired identity stub. Scans loads/stores and tags candidate reuse
// groups (consecutive loads from the same base register) but rewrites nothing.
//
// T3 (repeated_global_load) loads `[%rd6]` twice into %f1 and %f3 though the
// address and value are identical — the load-reuse / coalescing win lives there.
#include "aec/passes.h"

namespace aec {
namespace passes {

bool memCoalesce(ir::Function &fn, const Options &opt) {
  if (!opt.mem_coalesce) return false;
  bool changed = false;

  unsigned loads = 0, stores = 0;
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst &in = b.insts[ii];
      if (in.op == ir::Op::LD) ++loads;
      if (in.op == ir::Op::ST) ++stores;
      // TODO(T3): group loads that share a base address register + affine
      // offset into a single coalesced transaction; keep invariant reused
      // loads in registers; record memory_transactions.
      (void)in;
    }
  }
  (void)loads; (void)stores;

  return changed;
}

} // namespace passes
} // namespace aec
