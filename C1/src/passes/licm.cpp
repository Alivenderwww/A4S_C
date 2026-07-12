// licm.cpp - Loop-invariant code motion.  Scoring category: T2.
//
// STATUS: wired identity stub. Detects natural loops from the CFG back-edges
// (a real LICM's prerequisite) but hoists nothing yet.
//
// PTX-02 (invariant_poly) computes `add.f32 %f5,%f1,%f2` etc. inside its LOOP
// although %f1/%f2 are loop-invariant param values — that is the hoist target.
#include "aec/passes.h"

#include <vector>

namespace aec {
namespace passes {

bool licm(ir::Function &fn, const Options &opt) {
  if (!opt.licm) return false;
  bool changed = false;

  // Identify back-edges i -> h where h <= i (a header dominating the latch).
  // buildCFG must have run so succ/pred are populated.
  for (unsigned i = 0; i < fn.blocks.size(); ++i) {
    const ir::BasicBlock &b = fn.blocks[i];
    for (unsigned s = 0; s < b.succ.size(); ++s) {
      int h = b.succ[s];
      if (h <= (int)i) {
        // (i is a latch, h is a loop header) -> the loop body is [h..i].
        // TODO(T2): compute the invariant set (instructions whose operands are
        // all defined outside [h..i] and are pure), create/insert a preheader
        // block before h, move invariant instructions there, set changed=true.
        (void)h;
      }
    }
  }

  return changed;
}

} // namespace passes
} // namespace aec
