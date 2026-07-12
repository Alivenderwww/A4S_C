// mem_coalesce.cpp - GPGPU memory coalescing / load reuse.  Category: T3.
//
// STATUS: wired identity stub. Scans loads/stores and tags candidate reuse
// groups (consecutive loads from the same base register) but rewrites nothing.
//
// PTX-03 (repeated_reuse) reloads `[%rd5]` every loop iteration though the
// address is loop-invariant, and streams `[%rd7]` — the coalescing / shared-
// memory-cache win lives there (see spec.md 4.2 "内存合并访问").
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
      // offset into a single wide/coalesced transaction; promote invariant
      // reused loads into SMEM via TLDA/registers; record memory_transactions.
      (void)in;
    }
  }
  (void)loads; (void)stores;

  return changed;
}

} // namespace passes
} // namespace aec
