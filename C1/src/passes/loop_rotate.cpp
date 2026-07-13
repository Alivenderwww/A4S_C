// loop_rotate.cpp - Rotate guard-at-top `while` loops into do-while form.
// Scoring categories: T5 (and any counted loop). Enabling transform for unroll.
//
// Mirrors LLVM's LoopRotate: it is not itself an optimization but a
// CANONICALIZATION that lets the downstream unroller (and LICM) handle one loop
// shape. A PTX `while (k<K) {..}` lowers to a guard-at-top loop:
//     H:  CMPP.ge Pg, iv, bound ; BRX Pg -> EXIT      (test at the TOP)
//     L:  ...body...; ADD iv,iv,step ; BR -> H         (unconditional back-edge)
// Rotation moves the test to the bottom (a do-while / single-latch loop) and
// puts a copy of the guard before the loop so a zero-trip loop is still skipped:
//     preheader: ... ; CMPP.ge Pg, iv, bound ; BRX Pg -> EXIT     (guard)
//     L:  ...body...; ADD iv,iv,step ; CMPP.lt Pg, iv, bound ; BRX Pg -> L
// The rotated latch ends in exactly `ADD ; CMPP.lt ; BRX self`, which is the
// shape unroll.cpp already matches -- so unrolling the K-loop needs no change to
// the unroller. The transform is value-preserving: the pre-guard makes the
// do-while run iff the original while would (k<bound on entry), and each
// iteration is identical.
#include "aec/passes.h"

#include <string>
#include <vector>

namespace aec {
namespace passes {

namespace {
// Logical negation of a compare op: the exit test `iv >= bound` becomes the
// continue test `iv < bound` at the latch.
uint32_t invertCmp(uint32_t c) {
  switch (c) {
    case 0: return 1;  // eq -> ne
    case 1: return 0;  // ne -> eq
    case 2: return 5;  // lt -> ge
    case 3: return 4;  // le -> gt
    case 4: return 3;  // gt -> le
    case 5: return 2;  // ge -> lt
    default: return c;
  }
}
} // namespace

bool loopRotate(ir::Function &fn, const Options &opt) {
  if (!opt.unroll) return false;      // only useful ahead of the unroller (-O2+)
  buildCFG(fn);

  for (int h = 1; h + 1 < (int)fn.blocks.size(); ++h) {
    // Header H must be a PURE guard block: `CMPP.rel Pg, iv, bound ; BRX Pg`.
    ir::BasicBlock &H = fn.blocks[h];
    if (H.insts.size() != 2) continue;
    const ir::Inst cmp = H.insts[0];
    const ir::Inst brx = H.insts[1];
    if (cmp.op != ir::Op::CMPP || cmp.dst.kind != ir::Operand::Pred) continue;
    if (brx.op != ir::Op::BRX || brx.guardNeg) continue;
    if ((cmp.dst.value & 7) != (uint32_t)(brx.guard & 7)) continue;
    if (cmp.s1.kind != ir::Operand::Reg || cmp.s2.kind != ir::Operand::Reg)
      continue;

    // Body L is the fall-through successor (the block right after H) and must be
    // a single-block back-edge: it ends in an unconditional `BR -> H` and its
    // only successor is H.
    ir::BasicBlock &L = fn.blocks[h + 1];
    if (L.insts.empty() || !L.label.empty()) continue;   // unlabeled body only
    const ir::Inst &back = L.insts.back();
    if (back.op != ir::Op::BR || back.target != H.label || H.label.empty())
      continue;
    if (L.succ.size() != 1 || L.succ[0] != h) continue;

    // Single-latch while: H's only predecessors are the preheader (h-1) and the
    // latch (h+1). Anything else (a second back-edge / a jump to the header
    // label) would be broken by erasing H, so bail.
    if (H.pred.size() != 2) continue;
    bool hasPre = false, hasLatch = false;
    for (unsigned p = 0; p < H.pred.size(); ++p) {
      if (H.pred[p] == h - 1) hasPre = true;
      if (H.pred[p] == h + 1) hasLatch = true;
    }
    if (!hasPre || !hasLatch) continue;

    // The BRX exit target must be forward (a genuine loop exit, not the back
    // edge) so we are really looking at `while (cond) body`.
    bool exitForward = false;
    for (int e = h + 2; e < (int)fn.blocks.size(); ++e)
      if (fn.blocks[e].label == brx.target) { exitForward = true; break; }
    if (!exitForward) continue;

    // Preheader is the block immediately before H, entering it by fall-through
    // (no terminator). It dominates the loop and runs once.
    ir::BasicBlock &P = fn.blocks[h - 1];
    if (!P.insts.empty() && P.insts.back().isTerminator()) continue;
    bool preFallsToH = false;
    for (unsigned s = 0; s < P.succ.size(); ++s)
      if (P.succ[s] == h) preFallsToH = true;
    if (!preFallsToH) continue;

    // --- rotate ----------------------------------------------------------
    // 1. Pre-guard at the end of the preheader: skip the whole loop when the
    //    entry induction value already fails the test (zero-trip, e.g. K==0).
    P.insts.push_back(cmp);                 // CMPP.rel Pg, iv, bound
    P.insts.push_back(brx);                 // BRX Pg -> EXIT

    // 2. Turn L into the do-while latch: drop the back-edge BR, then test the
    //    (post-increment) induction value at the bottom and loop on continue.
    L.insts.pop_back();                     // remove `BR -> H`
    ir::Inst latchCmp = cmp;
    latchCmp.modifier = invertCmp(cmp.modifier);   // exit test -> continue test
    L.insts.push_back(latchCmp);
    ir::Inst latchBrx = brx;
    L.label = H.label;                      // reuse the header label for the latch
    latchBrx.target = L.label;              // BRX Pg -> self
    L.insts.push_back(latchBrx);

    // 3. Drop the now-empty header H (its guard moved to the preheader; the back
    //    edge became the latch self-branch). Targets use labels, so erasing the
    //    block is safe; buildCFG rewires succ/pred.
    fn.blocks.erase(fn.blocks.begin() + h);

    buildCFG(fn);
    return true;                            // one rotation per invocation; re-run
  }

  return false;
}

} // namespace passes
} // namespace aec
