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
// @P bra self`, with `step` and `bound` compile-time constants and the trip
// count bound/step divisible by U. Opt-in at -O3 only (keeps -O2 untouched).
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
    uint32_t step = 0, bound = 0;
    if (add.s2.kind != ir::Operand::Reg || !constOf(fn, add.s2.value, step)) continue;
    if (cmp.s2.kind != ir::Operand::Reg || !constOf(fn, cmp.s2.value, bound)) continue;
    if (step == 0 || bound == 0 || (bound % (step * (uint32_t)U)) != 0) continue;

    // Loop-carried set (kept shared across copies); the counter is shifted.
    std::set<uint32_t> carried;
    upwardExposed(b, carried);

    // Registers the steady body DEFINES that are not loop-carried -> per-copy.
    std::set<uint32_t> steadyDefs;
    for (int i = 0; i < n - 3; ++i)
      if (b.insts[i].dst.kind == ir::Operand::Reg &&
          !carried.count(b.insts[i].dst.value))
        steadyDefs.insert(b.insts[i].dst.value);

    // --- build the unrolled instruction list -----------------------------
    std::vector<ir::Inst> out;
    for (int c = 0; c < U; ++c) {
      std::map<uint32_t, uint32_t> ren;
      for (std::set<uint32_t>::iterator it = steadyDefs.begin();
           it != steadyDefs.end(); ++it)
        ren[*it] = fn.regs.nextVReg++;                 // fresh temp per copy.

      if (c == 0) {
        // copy 0 uses the counter directly.
      } else {
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
      }
      for (int i = 0; i < n - 3; ++i) {
        ir::Inst in = b.insts[i];
        renameOperand(in.dst, ren);
        renameOperand(in.s1, ren);
        renameOperand(in.s2, ren);
        renameOperand(in.s3, ren);
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
