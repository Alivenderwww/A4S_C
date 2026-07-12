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
  int exitIdx = -1;
  for (int bi = 0; bi < n; ++bi) {
    if (fn.blocks[bi].insts.empty()) continue;
    const ir::Inst &last = fn.blocks[bi].insts.back();
    if (last.op != ir::Op::BRX || last.guard < 0) continue;
    std::map<std::string, int>::iterator it = labelIdx.find(last.target);
    if (it == labelIdx.end() || it->second <= bi) continue;   // must be forward.
    uint32_t pg = (uint32_t)(last.guard & 0x7);
    int cidx = -1;
    for (int ii = (int)fn.blocks[bi].insts.size() - 2; ii >= 0; --ii) {
      const ir::Inst &in = fn.blocks[bi].insts[ii];
      if (in.op == ir::Op::CMPP && in.dst.kind == ir::Operand::Pred &&
          (in.dst.value & 0x7) == pg) { cidx = ii; break; }
    }
    if (cidx < 0) continue;
    guardBlk.push_back(bi); cmpAt.push_back(cidx); guardPred.push_back(pg);
    exitIdx = it->second;
  }
  if (guardBlk.empty()) return false;

  // 2. Flip each guard compare into a keep predicate, chain later guards onto
  //    the previous keep (so the final predicate is the AND of all keeps), and
  //    delete the guard branch.
  int keep = -1;
  for (size_t k = 0; k < guardBlk.size(); ++k) {
    ir::BasicBlock &b = fn.blocks[guardBlk[k]];
    ir::Inst &cmp = b.insts[cmpAt[k]];
    cmp.modifier = invertCmp(cmp.modifier);
    if (k > 0) cmp.guard = keep;               // P_k = keep_(k-1) && (this cmp).
    keep = (int)guardPred[k];
    b.insts.pop_back();                        // remove the BRX.
  }
  const int lastGuard = guardBlk.back();

  // 3. Predicate every memory op in the guarded body (blocks after the last
  //    guard, excluding the exit block) with the combined keep predicate.
  for (int bi = 0; bi < n; ++bi) {
    bool isGuard = false;
    for (size_t k = 0; k < guardBlk.size(); ++k) isGuard |= (guardBlk[k] == bi);
    if (isGuard || bi == exitIdx || bi < lastGuard) continue;
    for (size_t ii = 0; ii < fn.blocks[bi].insts.size(); ++ii) {
      ir::Inst &in = fn.blocks[bi].insts[ii];
      if (isMemOp(in.op) && in.guard < 0) in.guard = keep;
    }
  }

  buildCFG(fn);
  return true;
}

} // namespace passes
} // namespace aec
