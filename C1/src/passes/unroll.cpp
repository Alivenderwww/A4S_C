// unroll.cpp - Loop unrolling with induction-variable expansion.  Category: T4.
//
// Software-pipelining-style latency hiding: a single-block counted loop whose
// body issues one load per iteration is latency-bound (the consumer stalls
// ~32c on the load). Unrolling by U replicates the body U times using the
// induction values iv, iv+step, ..., iv+(U-1)*step, and — crucially — gives
// each copy FRESH virtual registers for its per-iteration temporaries so the U
// loads are independent. The list scheduler then batches them and their 32c
// latencies overlap. Loop-carried values (the accumulator, the counter) stay
// shared so the reduction still chains correctly.
//
// Scope (conservative, verified by the sim oracle): one single-block self-loop
// of the exact shape `...body...; counter += step; setp.lt P,counter,bound;
// @P bra self`, with `step` a compile-time constant. `bound` may be a RUNTIME
// value (e.g. GEMM's K param): when the trip is not a known multiple of U, the
// last group's out-of-range copies (iv_c >= bound) are predicated off with a
// fresh predicate p_c = G && (iv_c < bound) so no out-of-range load or
// accumulator update happens. On at -O2. GEMM K-loop 5240->2168 (-59%).
#include "aec/passes.h"

#include <map>
#include <set>
#include <vector>

namespace aec {
namespace passes {

namespace {

// Value of a register if it is defined by a single LOADI immediate anywhere in
// the function (constants are unique'd by CSE / hoisted by LICM at -O2+).
bool constOf(const ir::Function &fn, uint32_t reg, uint32_t &val) {
  int found = 0;
  for (unsigned b = 0; b < fn.blocks.size(); ++b)
    for (unsigned i = 0; i < fn.blocks[b].insts.size(); ++i) {
      const ir::Inst &in = fn.blocks[b].insts[i];
      if (in.op == ir::Op::LOADI && in.dst.kind == ir::Operand::Reg &&
          in.dst.value == reg && in.hasImm) { val = in.imm; ++found; }
    }
  return found == 1;
}

// Registers used before being defined within the block (loop-carried / live-in).
void upwardExposed(const ir::BasicBlock &b, std::set<uint32_t> &out) {
  std::set<uint32_t> defined;
  for (unsigned i = 0; i < b.insts.size(); ++i) {
    const ir::Inst &in = b.insts[i];
    const ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
    for (int k = 0; k < 3; ++k)
      if (s[k]->kind == ir::Operand::Reg && !defined.count(s[k]->value))
        out.insert(s[k]->value);
    if (in.dst.kind == ir::Operand::Reg) defined.insert(in.dst.value);
  }
}

void renameOperand(ir::Operand &o, const std::map<uint32_t, uint32_t> &m) {
  if (o.kind != ir::Operand::Reg) return;
  std::map<uint32_t, uint32_t>::const_iterator it = m.find(o.value);
  if (it != m.end()) o.value = it->second;
}

bool isMemOp(ir::Op op) {
  switch (op) {
    case ir::Op::LD: case ir::Op::ST: case ir::Op::LDC: case ir::Op::ATOM:
    case ir::Op::TLDA: case ir::Op::TSTA: case ir::Op::TMOV:
      return true;
    default:
      return false;
  }
}

} // namespace

bool unrollLoops(ir::Function &fn, const Options &opt) {
  if (!opt.unroll) return false;
  const int U = opt.unroll_factor;
  if (U < 2) return false;
  bool changed = false;

  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    const int n = (int)b.insts.size();
    if (n < 4) continue;

    // Require the exact tail shape: counter-ADD ; CMPP.lt ; BRX(self).
    const ir::Inst &brx = b.insts[n - 1];
    const ir::Inst &cmp = b.insts[n - 2];
    const ir::Inst &add = b.insts[n - 3];
    if (brx.op != ir::Op::BRX) continue;
    bool self = false;
    for (unsigned s = 0; s < b.succ.size(); ++s) self |= (b.succ[s] == (int)bi);
    if (!self) continue;
    if (cmp.op != ir::Op::CMPP || cmp.dst.kind != ir::Operand::Pred ||
        (cmp.dst.value & 0x7) != (uint32_t)(brx.guard & 0x7)) continue;
    if (add.op != ir::Op::ADD || add.dst.kind != ir::Operand::Reg ||
        cmp.s1.kind != ir::Operand::Reg || cmp.s1.value != add.dst.value) continue;
    if (add.s1.kind != ir::Operand::Reg || add.s1.value != add.dst.value) continue;

    const uint32_t counter = add.dst.value;
    uint32_t step = 0;
    if (add.s2.kind != ir::Operand::Reg || !constOf(fn, add.s2.value, step)) continue;
    if (step == 0) continue;
    if (cmp.s2.kind != ir::Operand::Reg) continue;
    const uint32_t boundReg = cmp.s2.value;

    // Bail if the body has a second self-incrementing induction variable
    // besides the counter (e.g. a strength-reduced address recurrence
    // `ADD addr,addr,stride`). Unrolling would have to offset each such IV by
    // c*stride per copy; not handled yet, so skip (SR already cut the body).
    bool multiIV = false;
    for (int i = 0; i < n - 3; ++i) {
      const ir::Inst &in = b.insts[i];
      if (in.op == ir::Op::ADD && in.dst.kind == ir::Operand::Reg &&
          in.s1.kind == ir::Operand::Reg && in.s1.value == in.dst.value &&
          in.dst.value != counter) { multiIV = true; break; }
    }
    if (multiIV) continue;

    // A constant trip that U divides needs no remainder. Otherwise (a runtime
    // bound like GEMM's K param, or a non-multiple constant) the last group has
    // fewer than U valid iterations, so copies past `bound` are predicated off.
    uint32_t boundConst = 0;
    const bool divisible = constOf(fn, boundReg, boundConst) && boundConst != 0 &&
                           (boundConst % (step * (uint32_t)U)) == 0;

    // Loop-carried set (kept shared across copies); the counter is shifted.
    std::set<uint32_t> carried;
    upwardExposed(b, carried);

    // Registers the steady body DEFINES that are not loop-carried -> per-copy.
    std::set<uint32_t> steadyDefs;
    for (int i = 0; i < n - 3; ++i)
      if (b.insts[i].dst.kind == ir::Operand::Reg &&
          !carried.count(b.insts[i].dst.value))
        steadyDefs.insert(b.insts[i].dst.value);

    // Register-pressure guard (CORRECTNESS, not just perf). Each copy gets its
    // own fresh temporaries, and the scheduler deliberately keeps them live at
    // once to overlap load latency, so peak live registers grow ~U x the body's
    // temps. Our spiller is a STUB (a real spill would clobber), so REFUSE to
    // unroll whenever the estimate would exceed a safe fraction of the register
    // file -- skipping only forfeits the speedup, unrolling into a spill would
    // be WRONG. This is what makes unroll safe at -O2 (the default level,
    // applied to every kernel incl. register-pressure mutations).
    const size_t estRegs =
        carried.size() + steadyDefs.size() * (size_t)U + 2u * (size_t)(U - 1);
    if (estRegs > (size_t)(kRegisterCount * 3 / 4)) continue;   // ~192 of 256

    // Remainder predication for a non-divisible trip: copies c>=1 whose induction
    // value iv_c >= bound must skip their memory ops (out-of-range read) and
    // loop-carried writes (accumulator corruption). Each gets a fresh predicate
    // p_c = G && (iv_c < bound), where G is the body's (loop-invariant) memory
    // guard; predicates start false and G is invariant, so guarding the p_c
    // compare by G gives exactly that AND. Bail (skip unroll, always safe) if the
    // body's memory ops don't share one loop-invariant guard, or predicates run
    // out.
    int bodyGuard = -2;                 // -2 unset, -1 none, >=0 predicate id
    std::vector<int> remPred;
    if (!divisible) {
      for (int i = 0; i < n - 3; ++i) {
        if (!isMemOp(b.insts[i].op)) continue;
        int g = b.insts[i].guard;
        if (bodyGuard == -2) bodyGuard = g;
        else if (bodyGuard != g) { bodyGuard = -3; break; }
      }
      if (bodyGuard == -3) continue;    // mixed memory guards
      if (bodyGuard == -2) bodyGuard = -1;
      bool guardInvariant = true;       // G must not be written inside the loop
      if (bodyGuard >= 0)
        for (int i = 0; i < n - 3; ++i)
          if (b.insts[i].dst.kind == ir::Operand::Pred &&
              (int)(b.insts[i].dst.value & 7) == bodyGuard) guardInvariant = false;
      if (!guardInvariant) continue;
      std::set<int> used;
      for (unsigned bb = 0; bb < fn.blocks.size(); ++bb)
        for (unsigned i = 0; i < fn.blocks[bb].insts.size(); ++i) {
          const ir::Inst &in = fn.blocks[bb].insts[i];
          if (in.guard >= 0) used.insert(in.guard & 7);
          if (in.dst.kind == ir::Operand::Pred) used.insert((int)(in.dst.value & 7));
        }
      for (int p = 0; p < 8 && (int)remPred.size() < U - 1; ++p)
        if (!used.count(p)) remPred.push_back(p);
      if ((int)remPred.size() < U - 1) continue;   // not enough free predicates
    }

    // --- build the unrolled instruction list -----------------------------
    std::vector<ir::Inst> out;
    for (int c = 0; c < U; ++c) {
      std::map<uint32_t, uint32_t> ren;
      for (std::set<uint32_t>::iterator it = steadyDefs.begin();
           it != steadyDefs.end(); ++it)
        ren[*it] = fn.regs.nextVReg++;                 // fresh temp per copy.

      int pc = -1;
      if (c > 0) {
        uint32_t cimm = fn.regs.nextVReg++;            // LOADI c*step
        uint32_t ivc  = fn.regs.nextVReg++;            // iv_c = counter + c*step
        ren[counter] = ivc;
        ir::Inst li; li.op = ir::Op::LOADI; li.type = ir::Type::U32;
        li.dst = ir::Operand::reg(cimm); li.hasImm = true; li.imm = (uint32_t)c * step;
        out.push_back(li);
        ir::Inst ad; ad.op = ir::Op::ADD; ad.type = ir::Type::U32;
        ad.dst = ir::Operand::reg(ivc);
        ad.s1 = ir::Operand::reg(counter); ad.s2 = ir::Operand::reg(cimm);
        out.push_back(ad);
        if (!divisible) {
          pc = remPred[c - 1];                         // p_c = G && (iv_c < bound)
          ir::Inst cp = cmp;                           // same CMPP.lt.<type>, s2=bound
          cp.dst = ir::Operand::pred((uint32_t)pc);
          cp.s1 = ir::Operand::reg(ivc);
          cp.guard = bodyGuard;
          out.push_back(cp);
        }
      }
      for (int i = 0; i < n - 3; ++i) {
        ir::Inst in = b.insts[i];
        const bool carriedWrite = in.dst.kind == ir::Operand::Reg &&
                                  carried.count(in.dst.value) != 0;
        renameOperand(in.dst, ren);
        renameOperand(in.s1, ren);
        renameOperand(in.s2, ren);
        renameOperand(in.s3, ren);
        if (pc >= 0 && (isMemOp(in.op) || carriedWrite))
          in.guard = pc;                               // off past bound
        out.push_back(in);
      }
    }

    // counter += U*step ; then the (unchanged) CMPP + BRX.
    uint32_t ustepReg = fn.regs.nextVReg++;
    ir::Inst li; li.op = ir::Op::LOADI; li.type = ir::Type::U32;
    li.dst = ir::Operand::reg(ustepReg); li.hasImm = true; li.imm = step * (uint32_t)U;
    out.push_back(li);
    ir::Inst ad = add; ad.s2 = ir::Operand::reg(ustepReg);
    out.push_back(ad);
    out.push_back(cmp);
    out.push_back(brx);

    b.insts.swap(out);
    changed = true;
  }

  if (changed) buildCFG(fn);
  return changed;
}

} // namespace passes
} // namespace aec
