// linear_scan.cpp - 256-GPR linear-scan register allocation.  Category: T4.
//
// A working (not stubbed) linear-scan allocator: it linearizes the blocks,
// computes a live interval [firstUse,lastUse] per virtual register, then walks
// intervals in start order assigning physical registers R1..R255 and freeing
// them as intervals expire (R0 is reserved as scratch/zero per ir.h).
//
// Spilling itself is a STUB: when the 255 GPRs are exhausted the allocator
// records a spill and reuses the highest register instead of materializing
// load/store spill code. Real spill-slot generation is the T4 TODO.
#include "aec/passes.h"
#include "aec/target.h"

#include <algorithm>
#include <map>
#include <vector>

namespace aec {
namespace regalloc {

namespace {

struct Interval {
  uint32_t vreg;
  int start;
  int end;
  int phys; // assigned physical register (filled during scan).
};

bool byStart(const Interval &a, const Interval &b) { return a.start < b.start; }

// Extend interval[v] to include position p.
void touch(std::map<uint32_t, Interval> &iv, uint32_t v, int p) {
  std::map<uint32_t, Interval>::iterator it = iv.find(v);
  if (it == iv.end()) {
    Interval x; x.vreg = v; x.start = p; x.end = p; x.phys = -1;
    iv[v] = x;
  } else {
    if (p < it->second.start) it->second.start = p;
    if (p > it->second.end) it->second.end = p;
  }
}

void collectRegs(const ir::Inst &in, std::vector<const ir::Operand *> &out) {
  out.clear();
  out.push_back(&in.dst);
  out.push_back(&in.s1);
  out.push_back(&in.s2);
  out.push_back(&in.s3);
}

} // namespace

void linearScan(ir::Function &fn, const Options & /*opt*/) {
  // 1. Linearize and build live intervals over virtual registers.
  std::map<uint32_t, Interval> intervals;
  int pos = 0;
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    for (unsigned ii = 0; ii < b.insts.size(); ++ii, ++pos) {
      std::vector<const ir::Operand *> ops;
      collectRegs(b.insts[ii], ops);
      for (unsigned k = 0; k < ops.size(); ++k)
        if (ops[k]->kind == ir::Operand::Reg)
          touch(intervals, ops[k]->value, pos);
    }
  }

  std::vector<Interval> ivs;
  ivs.reserve(intervals.size());
  for (std::map<uint32_t, Interval>::iterator it = intervals.begin();
       it != intervals.end(); ++it)
    ivs.push_back(it->second);
  std::sort(ivs.begin(), ivs.end(), byStart);

  // 2. Linear-scan assignment. Free pool = R1..R(kRegisterCount-1).
  std::vector<int> freePool;
  for (int r = (int)kRegisterCount - 1; r >= 1; --r) freePool.push_back(r);

  std::vector<int> active; // indices into ivs, kept sorted by end ascending.
  std::map<uint32_t, int> physOf;
  uint32_t spillCount = 0;
  uint32_t maxPhys = 0;

  for (unsigned i = 0; i < ivs.size(); ++i) {
    // Expire intervals that end before this one starts.
    std::vector<int> stillActive;
    for (unsigned a = 0; a < active.size(); ++a) {
      if (ivs[active[a]].end < ivs[i].start) {
        freePool.push_back(ivs[active[a]].phys); // return the register.
      } else {
        stillActive.push_back(active[a]);
      }
    }
    active.swap(stillActive);

    int reg;
    if (!freePool.empty()) {
      reg = freePool.back();
      freePool.pop_back();
    } else {
      // SPILL STUB: out of registers. A real allocator would spill the
      // interval with the furthest end to an LMEM slot and insert LD/ST.
      // Here we clamp to the top register and count the spill.
      reg = (int)kRegisterCount - 1;
      ++spillCount;
      // TODO(T4): pick a spill victim, allocate a spill slot in Space::LMEM,
      // emit ST at the def and LD before each use, and rewrite operands.
    }
    ivs[i].phys = reg;
    physOf[ivs[i].vreg] = reg;
    if ((uint32_t)reg > maxPhys) maxPhys = (uint32_t)reg;
    active.push_back((int)i);
  }

  // 3. Rewrite every Reg operand to its assigned physical register.
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst &in = b.insts[ii];
      ir::Operand *ops[4] = {&in.dst, &in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 4; ++k) {
        if (ops[k]->kind == ir::Operand::Reg) {
          std::map<uint32_t, int>::iterator it = physOf.find(ops[k]->value);
          int phys = (it != physOf.end()) ? it->second : 0;
          *ops[k] = ir::Operand::phys((uint32_t)phys);
        }
      }
    }
  }

  fn.regs.maxPhys = maxPhys;
  fn.regs.spillCount = spillCount;
}

} // namespace regalloc
} // namespace aec
