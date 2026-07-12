// linear_scan.cpp - 256-GPR linear-scan register allocation.  Category: T4.
//
// A liveness-correct linear-scan allocator. The earlier scaffold built live
// intervals as [firstTextualUse, lastTextualUse], which UNDER-approximates
// liveness across loop back-edges: a value defined before a loop and re-used
// every iteration has its textual last-use at the loop top, so its interval
// stopped early and a later loop-local temporary got the same physical
// register -- clobbering the loop-carried value (the `reuse` dogfood bug).
//
// This version computes real per-block liveness (backward dataflow to a
// fixpoint over the CFG, so back-edges extend ranges over the whole loop),
// derives an interval per virtual register from live-in/live-out + def/use
// positions, then does the standard linear scan (assign R1..R255, expire on
// interval end). Spilling is still a STUB (clamp to the top register + count).
#include "aec/passes.h"
#include "aec/target.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace aec {
namespace regalloc {

namespace {

typedef std::set<uint32_t> RegSet;

struct Interval {
  uint32_t vreg;
  int start;
  int end;
  int phys;
};

bool byStart(const Interval &a, const Interval &b) {
  if (a.start != b.start) return a.start < b.start;
  return a.vreg < b.vreg;
}

// Virtual registers read (upward-exposed uses) / written by one instruction.
void instRegs(const ir::Inst &in, std::vector<uint32_t> &uses, int &defReg) {
  uses.clear();
  defReg = -1;
  const ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
  for (int k = 0; k < 3; ++k)
    if (srcs[k]->kind == ir::Operand::Reg) uses.push_back(srcs[k]->value);
  if (in.dst.kind == ir::Operand::Reg) defReg = (int)in.dst.value;
}

} // namespace

void linearScan(ir::Function &fn, const Options & /*opt*/) {
  const unsigned nb = fn.blocks.size();
  if (nb == 0) return;

  // --- 1. Global instruction positions + per-block [lo,hi]. ---------------
  std::vector<int> blockLo(nb, -1), blockHi(nb, -1);
  int pos = 0;
  for (unsigned b = 0; b < nb; ++b) {
    if (fn.blocks[b].insts.empty()) { blockLo[b] = blockHi[b] = pos; continue; }
    blockLo[b] = pos;
    pos += (int)fn.blocks[b].insts.size();
    blockHi[b] = pos - 1;
  }

  // --- 2. Local use/def sets (block granularity, upward-exposed uses). -----
  std::vector<RegSet> useB(nb), defB(nb), liveIn(nb), liveOut(nb);
  for (unsigned b = 0; b < nb; ++b) {
    RegSet defined;
    const ir::BasicBlock &blk = fn.blocks[b];
    for (unsigned i = 0; i < blk.insts.size(); ++i) {
      std::vector<uint32_t> uses; int def;
      instRegs(blk.insts[i], uses, def);
      for (unsigned k = 0; k < uses.size(); ++k)
        if (!defined.count(uses[k])) useB[b].insert(uses[k]); // upward exposed
      if (def >= 0) { defined.insert((uint32_t)def); defB[b].insert((uint32_t)def); }
    }
  }

  // --- 3. Backward liveness dataflow to a fixpoint (handles loops). --------
  bool changed = true;
  while (changed) {
    changed = false;
    for (int b = (int)nb - 1; b >= 0; --b) {
      RegSet out;
      for (unsigned s = 0; s < fn.blocks[b].succ.size(); ++s) {
        int sc = fn.blocks[b].succ[s];
        if (sc >= 0 && sc < (int)nb)
          out.insert(liveIn[sc].begin(), liveIn[sc].end());
      }
      // in = use ∪ (out − def)
      RegSet in = useB[b];
      for (RegSet::iterator it = out.begin(); it != out.end(); ++it)
        if (!defB[b].count(*it)) in.insert(*it);
      if (out != liveOut[b]) { liveOut[b] = out; changed = true; }
      if (in != liveIn[b])   { liveIn[b] = in;   changed = true; }
    }
  }

  // --- 4. Build one interval per vreg from liveness + def/use positions. ---
  std::map<uint32_t, Interval> iv;
  // extend(v,p): grow v's interval to include position p.
  // (lambda-free for g++ 4.9 portability)
  struct Ext {
    static void go(std::map<uint32_t, Interval> &m, uint32_t v, int p) {
      std::map<uint32_t, Interval>::iterator it = m.find(v);
      if (it == m.end()) { Interval x; x.vreg=v; x.start=p; x.end=p; x.phys=-1; m[v]=x; }
      else { if (p < it->second.start) it->second.start = p;
             if (p > it->second.end)   it->second.end = p; }
    }
  };
  for (unsigned b = 0; b < nb; ++b) {
    if (fn.blocks[b].insts.empty()) continue;
    // live-in reaches the block start; live-out reaches the block end.
    for (RegSet::iterator it = liveIn[b].begin(); it != liveIn[b].end(); ++it)
      Ext::go(iv, *it, blockLo[b]);
    for (RegSet::iterator it = liveOut[b].begin(); it != liveOut[b].end(); ++it)
      Ext::go(iv, *it, blockHi[b]);
    int p = blockLo[b];
    for (unsigned i = 0; i < fn.blocks[b].insts.size(); ++i, ++p) {
      std::vector<uint32_t> uses; int def;
      instRegs(fn.blocks[b].insts[i], uses, def);
      for (unsigned k = 0; k < uses.size(); ++k) Ext::go(iv, uses[k], p);
      if (def >= 0) Ext::go(iv, (uint32_t)def, p);
    }
  }

  std::vector<Interval> ivs;
  ivs.reserve(iv.size());
  for (std::map<uint32_t, Interval>::iterator it = iv.begin(); it != iv.end(); ++it)
    ivs.push_back(it->second);
  std::sort(ivs.begin(), ivs.end(), byStart);

  // --- 5. Linear scan. Free pool R1..R(kRegisterCount-1) (R0 = scratch). ---
  std::vector<int> freePool;
  for (int r = (int)kRegisterCount - 1; r >= 1; --r) freePool.push_back(r);
  std::vector<int> active; // indices into ivs, expired lazily.
  std::map<uint32_t, int> physOf;
  uint32_t spillCount = 0, maxPhys = 0;

  for (unsigned i = 0; i < ivs.size(); ++i) {
    std::vector<int> keep;
    for (unsigned a = 0; a < active.size(); ++a) {
      if (ivs[active[a]].end < ivs[i].start) freePool.push_back(ivs[active[a]].phys);
      else keep.push_back(active[a]);
    }
    active.swap(keep);

    int reg;
    if (!freePool.empty()) { reg = freePool.back(); freePool.pop_back(); }
    else {
      // SPILL STUB: real allocator would spill the furthest-end interval to an
      // LMEM slot and emit LD/ST. TODO(T4): materialize spill code.
      reg = (int)kRegisterCount - 1; ++spillCount;
    }
    ivs[i].phys = reg;
    physOf[ivs[i].vreg] = reg;
    if ((uint32_t)reg > maxPhys) maxPhys = (uint32_t)reg;
    active.push_back((int)i);
  }

  // --- 6. Rewrite Reg operands to their physical register. ----------------
  for (unsigned b = 0; b < nb; ++b) {
    for (unsigned i = 0; i < fn.blocks[b].insts.size(); ++i) {
      ir::Inst &in = fn.blocks[b].insts[i];
      ir::Operand *ops[4] = {&in.dst, &in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 4; ++k) {
        if (ops[k]->kind == ir::Operand::Reg) {
          std::map<uint32_t, int>::iterator it = physOf.find(ops[k]->value);
          *ops[k] = ir::Operand::phys((uint32_t)(it != physOf.end() ? it->second : 0));
        }
      }
    }
  }

  fn.regs.maxPhys = maxPhys;
  fn.regs.spillCount = spillCount;
}

} // namespace regalloc
} // namespace aec
