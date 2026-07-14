// pred_opt.cpp - Bounds-guard if-conversion (predication).  Category: T2 / robustness.
//
// AEC has no runtime SIMT reconvergence: `BRX` requires a UNIFORM condition
// across active lanes. A PTX bounds guard `setp.ge %p, idx, n; @%p bra DONE;`
// is divergent whenever N is not a multiple of blockDim (the last partial block
// has some lanes with idx>=n) -> the BRX is an execution error. See §1.3.
//
// This pass if-converts such guards: it flips the guard compare into a "keep"
// predicate (idx < n), deletes the branch, and predicates every MEMORY
// operation in the guarded body with that predicate. Non-kept lanes then run
// the same straight-line/loop code (computing unused garbage) but perform no
// loads/stores, so there is no out-of-bounds access and no divergent branch.
// Arithmetic and loop control stay unpredicated, keeping the loop `BRX` uniform
// (its trip count is uniform). Multiple guards (e.g. GEMM's row+col checks) are
// AND-combined by predicating each later guard compare on the previous keep.
#include "aec/passes.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace aec {
namespace passes {

namespace {

bool isMemOp(ir::Op op) {
  switch (op) {
    case ir::Op::LD: case ir::Op::ST: case ir::Op::LDC: case ir::Op::ATOM:
    case ir::Op::TLDA: case ir::Op::TSTA: case ir::Op::TMOV:
      return true;
    default:
      return false;
  }
}

// Logical negation of a compare op (eq<->ne, lt<->ge, le<->gt): keep = !skip.
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

bool predOpt(ir::Function &fn, const Options &opt) {
  if (!opt.pred_opt) return false;
  buildCFG(fn);
  const int n = (int)fn.blocks.size();

  std::map<std::string, int> labelIdx;
  for (int i = 0; i < n; ++i)
    if (!fn.blocks[i].label.empty()) labelIdx[fn.blocks[i].label] = i;

  // 1. Collect forward-branch bounds guards: a block ending in `BRX Pg -> tgt`
  //    with tgt > bi (a skip), whose Pg is produced by a CMPP in the block.
  std::vector<int> guardBlk, cmpAt;
  std::vector<uint32_t> guardPred;
  std::vector<char> guardNegVec;
  int exitIdx = -1;
  for (int bi = 0; bi < n; ++bi) {
    if (fn.blocks[bi].insts.empty()) continue;
    const ir::Inst &last = fn.blocks[bi].insts.back();
    if (last.op != ir::Op::BRX || last.guard < 0) continue;
    std::map<std::string, int>::iterator it = labelIdx.find(last.target);
    if (it == labelIdx.end() || it->second <= bi) continue;   // must be forward.
    // Skip a LOOP-EXIT guard: if any later block branches back into this block,
    // it is a loop header and the forward `BRX` is the loop exit, not a bounds
    // guard. If-converting it would delete the exit and leave the back-edge
    // unconditional -> infinite loop (e.g. a `while (k<K)` GEMM K-loop).
    bool loopHeader = false;
    for (int bb = bi + 1; bb < n && !loopHeader; ++bb)
      for (unsigned s = 0; s < fn.blocks[bb].succ.size(); ++s)
        if (fn.blocks[bb].succ[s] == bi) { loopHeader = true; break; }
    if (loopHeader) continue;
    uint32_t pg = (uint32_t)(last.guard & 0x7);
    int cidx = -1;
    for (int ii = (int)fn.blocks[bi].insts.size() - 2; ii >= 0; --ii) {
      const ir::Inst &in = fn.blocks[bi].insts[ii];
      if (in.op == ir::Op::CMPP && in.dst.kind == ir::Operand::Pred &&
          (in.dst.value & 0x7) == pg) { cidx = ii; break; }
    }
    if (cidx < 0) continue;
    guardBlk.push_back(bi); cmpAt.push_back(cidx); guardPred.push_back(pg);
    guardNegVec.push_back(last.guardNeg ? 1 : 0);
    exitIdx = it->second;
  }
  if (guardBlk.empty()) return false;

  // 2. Flip each guard compare into a keep predicate, chain later guards onto
  //    the previous keep (so the final predicate is the AND of all keeps), and
  //    delete the guard branch.
  int keep = -1;
  for (size_t k = 0; k < guardBlk.size(); ++k) {
    ir::BasicBlock &b = fn.blocks[guardBlk[k]];
    if (k > 0) {
      // Chained guard: this compare runs under @keep, so a lane masked off by an
      // earlier guard does NOT write this predicate and would read a stale
      // value. Clear it to false first (x != x) so the AND is correct without
      // relying on any predicate-register reset convention.
      ir::Operand src = b.insts[cmpAt[k]].s1;
      ir::Type    cty = b.insts[cmpAt[k]].type;
      ir::Inst clr; clr.op = ir::Op::CMPP; clr.type = cty;
      clr.dst = ir::Operand::pred(guardPred[k]);
      clr.s1 = src; clr.s2 = src; clr.modifier = 1u;   // .ne : x != x -> false
      b.insts.insert(b.insts.begin() + cmpAt[k], clr);
      ++cmpAt[k];
    }
    ir::Inst &cmp = b.insts[cmpAt[k]];
    // Flip the branch compare into a keep (execute) predicate. A plain
    // `@%p bra SKIP` branches when P, so keep = !P -> invert the compare; a
    // `@!%p bra SKIP` branches when !P, so keep = P -> leave the compare as is.
    if (!guardNegVec[k]) cmp.modifier = invertCmp(cmp.modifier);
    if (k > 0) cmp.guard = keep;               // P_k = keep_(k-1) && (this cmp).
    keep = (int)guardPred[k];
    b.insts.pop_back();                        // remove the BRX.
  }
  const int lastGuard = guardBlk.back();

  // 3. Identify the guarded body: blocks after the last guard, excluding the
  //    exit/merge block (where lanes reconverge) and the guard blocks. Lanes
  //    that fail the guard fall straight through this region.
  std::vector<char> inBody(n, 0);
  for (int bi = 0; bi < n; ++bi) {
    bool isGuard = false;
    for (size_t k = 0; k < guardBlk.size(); ++k) isGuard |= (guardBlk[k] == bi);
    if (!(isGuard || bi == exitIdx || bi < lastGuard)) inBody[bi] = 1;
  }

  // A register defined in the body but READ outside it (at/after the merge point,
  // or before the guard) holds a value the reconverged code depends on, so a
  // masked-off lane must NOT overwrite it -- its defining instruction has to be
  // predicated. This is the `cond ? a : b` case: the body's write to the result
  // register escapes to the merge block. A value consumed only inside the body
  // feeds a (predicated) memory op, so it can stay unpredicated -- which is what
  // keeps loop counters (dead after the loop, hence not escaping) uniform and
  // the loop's own BRX condition warp-uniform.
  std::set<uint32_t> usedOutside;
  for (int bi = 0; bi < n; ++bi) {
    if (inBody[bi]) continue;
    for (size_t ii = 0; ii < fn.blocks[bi].insts.size(); ++ii) {
      const ir::Inst &in = fn.blocks[bi].insts[ii];
      const ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
      for (int u = 0; u < 3; ++u)
        if (srcs[u]->kind == ir::Operand::Reg) usedOutside.insert(srcs[u]->value);
    }
  }

  // 4. Predicate, within the body, every memory op (out-of-bounds safety) and
  //    every non-terminator whose result escapes the body (correctness of the
  //    reconverged value), all under the combined keep predicate.
  for (int bi = 0; bi < n; ++bi) {
    if (!inBody[bi]) continue;
    for (size_t ii = 0; ii < fn.blocks[bi].insts.size(); ++ii) {
      ir::Inst &in = fn.blocks[bi].insts[ii];
      if (in.guard >= 0 || in.isTerminator()) continue;
      const bool escapes = in.dst.kind == ir::Operand::Reg &&
                           usedOutside.count(in.dst.value) != 0;
      if (isMemOp(in.op) || escapes) in.guard = keep;
    }
  }

  buildCFG(fn);
  return true;
}

} // namespace passes
} // namespace aec
