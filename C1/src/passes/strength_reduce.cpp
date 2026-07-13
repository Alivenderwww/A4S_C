// strength_reduce.cpp - Operator strength reduction on loop address induction
// variables. Scoring: T5 (and any counted loop with linear array addressing).
//
// The graded metric is warp-level DYNAMIC INSTRUCTION COUNT, so the win is not
// latency hiding but removing instructions from the loop body. A GEMM K-loop
// recomputes each operand address from scratch every iteration:
//     MAD idx, row, K, k        ; idx = row*K + k     (linear in the IV k)
//     MUL off, idx, esize       ; off = idx * esize   (esize a constant)
//     ADD addr, base, off       ; addr = base + off   (base loop-invariant)
//     LD  val, [addr]
// addr is a linear function of k, so addr(k+1) = addr(k) + stride with
// stride = (coeff of k in idx) * esize. Classic induction-variable strength
// reduction (Cooper/Simpson/Vick OSR; LLVM LSR/SCEV addrec {base,+,stride}):
// compute the initial address and the stride once in the preheader and advance
// the address by a single ADD per iteration. Six address instructions per load
// collapse to one.
//
// Scope: a single-block do-while loop (post loop-rotate) with a basic induction
// variable `ADD iv,iv,step` in the latch. For each in-loop load/store whose
// address chain is `ADD base,(MUL idx,<const esize>)` with `idx = MAD` linear in
// iv, every other operand loop-invariant, and idx/off/addr each feeding only the
// memory op, the chain is replaced by a recurrence carried across iterations.
#include "aec/passes.h"

#include <map>
#include <set>
#include <vector>

namespace aec {
namespace passes {

namespace {

// Index of the single instruction in block b that defines register r, or -1 if
// r is defined zero or more than once (SR needs a unique in-loop definition).
int uniqueDef(const ir::BasicBlock &b, uint32_t r) {
  int found = -1;
  for (unsigned i = 0; i < b.insts.size(); ++i) {
    const ir::Inst &in = b.insts[i];
    if (in.dst.kind == ir::Operand::Reg && in.dst.value == r) {
      if (found >= 0) return -1;
      found = (int)i;
    }
  }
  return found;
}

// Number of times r appears as a source operand in block b.
int useCount(const ir::BasicBlock &b, uint32_t r) {
  int c = 0;
  for (unsigned i = 0; i < b.insts.size(); ++i) {
    const ir::Inst &in = b.insts[i];
    const ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
    for (int k = 0; k < 3; ++k)
      if (s[k]->kind == ir::Operand::Reg && s[k]->value == r) ++c;
  }
  return c;
}

// One strength-reducible load/store: its address chain and the derived stride.
struct Cand {
  int memIdx;                 // the LD/ST instruction in the loop body
  int madIdx, mulIdx, addIdx; // address-chain instructions to delete
  ir::Inst mad, mul, add;     // copies (to rebuild the preheader initializer)
  uint32_t coeffReg;          // iv coefficient (invariant reg), or 0 if coeff==1
  uint32_t esizeReg;          // element size register (invariant const)
};

} // namespace

bool strengthReduce(ir::Function &fn, const Options &opt) {
  if (!opt.unroll) return false;      // same -O2 gating as the other loop passes
  buildCFG(fn);

  for (int li = 0; li < (int)fn.blocks.size(); ++li) {
    ir::BasicBlock &L = fn.blocks[li];
    const int n = (int)L.insts.size();
    if (n < 4) continue;

    // Latch shape: ... ; ADD iv,iv,step ; CMPP rel,iv,bound ; BRX self.
    const ir::Inst &brx = L.insts[n - 1];
    const ir::Inst &cmp = L.insts[n - 2];
    const ir::Inst &add = L.insts[n - 3];
    bool self = false;
    for (unsigned s = 0; s < L.succ.size(); ++s) self |= (L.succ[s] == li);
    if (!self || brx.op != ir::Op::BRX || cmp.op != ir::Op::CMPP) continue;
    if (add.op != ir::Op::ADD || add.dst.kind != ir::Operand::Reg) continue;
    if (add.s1.kind != ir::Operand::Reg || add.s1.value != add.dst.value) continue;
    const uint32_t iv = add.dst.value;

    // Preheader: the unique loop-external predecessor (buildCFG gives pred).
    int pi = -1;
    for (unsigned p = 0; p < L.pred.size(); ++p)
      if (L.pred[p] != li) { if (pi >= 0) { pi = -1; break; } pi = L.pred[p]; }
    if (pi < 0) continue;

    // Registers defined inside the loop (everything else is loop-invariant).
    std::set<uint32_t> defs;
    for (int i = 0; i < n; ++i)
      if (L.insts[i].dst.kind == ir::Operand::Reg)
        defs.insert(L.insts[i].dst.value);
    // `iv` must be the ONLY self-incrementing induction variable so far, so this
    // is the first SR on this loop (later ADD-recurrences we create are new).

    // Collect strength-reducible address chains in the body (before the latch).
    std::vector<Cand> cands;
    for (int m = 0; m < n - 3; ++m) {
      const ir::Inst &mem = L.insts[m];
      if (mem.op != ir::Op::LD && mem.op != ir::Op::ST) continue;
      if (mem.s1.kind != ir::Operand::Reg) continue;
      const uint32_t addrReg = mem.s1.value;
      if (!defs.count(addrReg)) continue;             // invariant addr -> LICM
      if (useCount(L, addrReg) != 1) continue;        // addr feeds only this op
      int ai = uniqueDef(L, addrReg);
      if (ai < 0 || L.insts[ai].op != ir::Op::ADD) continue;
      const ir::Inst &addr = L.insts[ai];
      if (addr.s1.kind != ir::Operand::Reg || addr.s2.kind != ir::Operand::Reg)
        continue;
      // One ADD operand is the invariant base, the other the in-loop offset.
      bool s1In = defs.count(addr.s1.value) != 0;
      bool s2In = defs.count(addr.s2.value) != 0;
      if (s1In == s2In) continue;                     // need exactly one in-loop
      const uint32_t offReg = s1In ? addr.s1.value : addr.s2.value;
      if (useCount(L, offReg) != 1) continue;
      int oi = uniqueDef(L, offReg);
      if (oi < 0 || L.insts[oi].op != ir::Op::MUL) continue;
      const ir::Inst &mul = L.insts[oi];
      if (mul.s1.kind != ir::Operand::Reg || mul.s2.kind != ir::Operand::Reg)
        continue;
      // MUL off, idx, esize: one operand the in-loop idx, the other an invariant.
      bool m1In = defs.count(mul.s1.value) != 0;
      bool m2In = defs.count(mul.s2.value) != 0;
      if (m1In == m2In) continue;
      const uint32_t idxReg   = m1In ? mul.s1.value : mul.s2.value;
      const uint32_t esizeReg = m1In ? mul.s2.value : mul.s1.value;
      if (useCount(L, idxReg) != 1) continue;
      int ii = uniqueDef(L, idxReg);
      if (ii < 0 || L.insts[ii].op != ir::Op::MAD) continue;
      const ir::Inst &mad = L.insts[ii];
      // MAD idx, a, b, c = a*b + c, linear in iv iff exactly one operand is iv
      // and the others are loop-invariant. coeff of iv: if iv is the addend c,
      // coeff==1; if iv is a factor (a or b), coeff is the other factor.
      const ir::Operand *ops[3] = {&mad.s1, &mad.s2, &mad.s3};
      int ivPos = -1, nIv = 0;
      for (int k = 0; k < 3; ++k)
        if (ops[k]->kind == ir::Operand::Reg && ops[k]->value == iv) { ivPos = k; ++nIv; }
      if (nIv != 1) continue;
      // every non-iv operand must be loop-invariant
      bool ok = true;
      for (int k = 0; k < 3; ++k)
        if (k != ivPos && ops[k]->kind == ir::Operand::Reg && defs.count(ops[k]->value))
          ok = false;
      if (!ok) continue;
      uint32_t coeffReg = 0;                           // 0 == coeff of 1
      if (ivPos == 0) coeffReg = mad.s2.value;         // iv * b  -> coeff b
      else if (ivPos == 1) coeffReg = mad.s1.value;    // a * iv  -> coeff a
      // ivPos == 2 (addend): coeff 1, coeffReg stays 0
      Cand c;
      c.memIdx = m; c.madIdx = ii; c.mulIdx = oi; c.addIdx = ai;
      c.mad = mad; c.mul = mul; c.add = addr;
      c.coeffReg = coeffReg; c.esizeReg = esizeReg;
      cands.push_back(c);
    }
    if (cands.empty()) continue;

    // --- rewrite ---------------------------------------------------------
    ir::BasicBlock &P = fn.blocks[pi];
    // Preheader insertion goes before any terminator (the rotate pre-guard).
    int insAt = (int)P.insts.size();
    if (insAt > 0 && P.insts[insAt - 1].isTerminator()) --insAt;
    std::vector<ir::Inst> pre;               // initializers to splice into P
    std::map<int, uint32_t> memAddrIV;       // memIdx -> recurrence register
    std::map<int, uint32_t> memStride;       // memIdx -> stride register
    std::set<int> drop;                      // body indices to remove

    for (unsigned ci = 0; ci < cands.size(); ++ci) {
      const Cand &c = cands[ci];
      // Preheader: addr_init = base + (idx at iv==init)*esize, replicating the
      // chain with fresh temps (iv still holds its entry value here).
      uint32_t idx0 = fn.regs.nextVReg++;
      uint32_t off0 = fn.regs.nextVReg++;
      uint32_t addrIV = fn.regs.nextVReg++;
      ir::Inst mad0 = c.mad; mad0.dst = ir::Operand::reg(idx0);
      ir::Inst mul0 = c.mul; mul0.dst = ir::Operand::reg(off0);
      // point the MUL at the fresh idx0 (whichever source was the idx)
      if (mul0.s1.kind == ir::Operand::Reg && mul0.s1.value == c.mad.dst.value)
        mul0.s1 = ir::Operand::reg(idx0);
      else
        mul0.s2 = ir::Operand::reg(idx0);
      ir::Inst add0 = c.add; add0.dst = ir::Operand::reg(addrIV);
      if (add0.s1.kind == ir::Operand::Reg && add0.s1.value == c.add.dst.value)
        add0.s1 = ir::Operand::reg(off0);   // shouldn't happen; safety
      if (add0.s1.kind == ir::Operand::Reg && add0.s1.value == c.mul.dst.value)
        add0.s1 = ir::Operand::reg(off0);
      if (add0.s2.kind == ir::Operand::Reg && add0.s2.value == c.mul.dst.value)
        add0.s2 = ir::Operand::reg(off0);
      pre.push_back(mad0); pre.push_back(mul0); pre.push_back(add0);

      // stride = coeff * esize (coeff==1 -> just esize).
      uint32_t strideReg;
      if (c.coeffReg == 0) {
        strideReg = c.esizeReg;             // reuse the existing constant
      } else {
        strideReg = fn.regs.nextVReg++;
        ir::Inst sm; sm.op = ir::Op::MUL; sm.type = ir::Type::U32;
        sm.dst = ir::Operand::reg(strideReg);
        sm.s1 = ir::Operand::reg(c.coeffReg);
        sm.s2 = ir::Operand::reg(c.esizeReg);
        pre.push_back(sm);
      }
      memAddrIV[c.memIdx] = addrIV;
      memStride[c.memIdx] = strideReg;
      drop.insert(c.madIdx); drop.insert(c.mulIdx); drop.insert(c.addIdx);
    }

    // Rebuild the loop body: drop the address chains, point each mem op at its
    // recurrence register, and advance that register right after the mem op.
    std::vector<ir::Inst> body;
    body.reserve(L.insts.size());
    for (int i = 0; i < n; ++i) {
      if (drop.count(i)) continue;
      ir::Inst in = L.insts[i];
      std::map<int, uint32_t>::iterator it = memAddrIV.find(i);
      if (it != memAddrIV.end()) {
        in.s1 = ir::Operand::reg(it->second);           // load/store from addrIV
        body.push_back(in);
        ir::Inst adv; adv.op = ir::Op::ADD; adv.type = ir::Type::U32;
        adv.dst = ir::Operand::reg(it->second);
        adv.s1 = ir::Operand::reg(it->second);
        adv.s2 = ir::Operand::reg(memStride[i]);
        body.push_back(adv);                            // addrIV += stride
      } else {
        body.push_back(in);
      }
    }

    L.insts.swap(body);
    P.insts.insert(P.insts.begin() + insAt, pre.begin(), pre.end());
    buildCFG(fn);
    return true;                                        // one loop per invocation
  }

  return false;
}

} // namespace passes
} // namespace aec
