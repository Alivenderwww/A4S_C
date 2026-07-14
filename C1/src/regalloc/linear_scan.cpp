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
// positions, then does the standard linear scan (expire on interval end).
//
// Register pressure that exceeds the file is spilled for real.  The top of the
// register file (R247..R255) is reserved as reload/store scratch
// and over-pressure values are evicted to per-thread .lmem (SpillAtInterval,
// with LD.lmem/ST.lmem materialized around each use/def in section 6). At -O2
// the pre-RA list scheduler sinks each independent value to its use, so peak
// pressure normally stays tiny -- a kernel with 300 simultaneously-defined
// values allocates ~6 physical registers and spills nothing, and the output is
// byte-identical to the old R1..R255 scan. The spiller only engages once >250
// values are genuinely live at one point, which the scalar §3 subset makes rare
// but no longer a silent miscompile. 64-bit values are spilled as two adjacent
// 32-bit LMEM words and reloaded into a scratch register pair.
#include "aec/passes.h"
#include "aec/target.h"

#include <algorithm>
#include <cstdlib>
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

// LMEM spill helpers.  A slot is one 32-bit word at byte offset slot*4 in the
// per-thread .lmem window; the address must be materialized into a register
// because LD/ST take only [Ra] (no immediate offset, spec §5.2).  `addr` is the
// address-scratch register, `val` the value-scratch register holding the datum.
void emitReload(std::vector<ir::Inst> &out, int addr, int val, int slot) {
  ir::Inst li; li.op = ir::Op::LOADI; li.type = ir::Type::NONE;
  li.dst = ir::Operand::phys((uint32_t)addr);
  li.hasImm = true; li.imm = (uint32_t)slot * 4u;
  out.push_back(li);
  ir::Inst ld; ld.op = ir::Op::LD; ld.type = ir::Type::B32;
  ld.modifier = (uint32_t)isa::Space::LMEM;
  ld.dst = ir::Operand::phys((uint32_t)val);
  ld.s1 = ir::Operand::phys((uint32_t)addr);
  ld.note = "reload";
  out.push_back(ld);
}

void emitStore(std::vector<ir::Inst> &out, int addr, int val, int slot) {
  ir::Inst li; li.op = ir::Op::LOADI; li.type = ir::Type::NONE;
  li.dst = ir::Operand::phys((uint32_t)addr);
  li.hasImm = true; li.imm = (uint32_t)slot * 4u;
  out.push_back(li);
  ir::Inst st; st.op = ir::Op::ST; st.type = ir::Type::B32;
  st.modifier = (uint32_t)isa::Space::LMEM;
  st.s1 = ir::Operand::phys((uint32_t)addr);
  st.s2 = ir::Operand::phys((uint32_t)val);
  st.note = "spill";
  out.push_back(st);
}

void emitReloadWide(std::vector<ir::Inst> &out, int addr, int val, int slot) {
  // Two b32 accesses avoid imposing an 8-byte alignment requirement on the
  // LMEM slot after an odd number of narrow spills.
  emitReload(out, addr, val, slot);
  emitReload(out, addr, val + 1, slot + 1);
}

void emitStoreWide(std::vector<ir::Inst> &out, int addr, int val, int slot) {
  emitStore(out, addr, val, slot);
  emitStore(out, addr, val + 1, slot + 1);
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
      if (def >= 0) {
        defined.insert((uint32_t)def); defB[b].insert((uint32_t)def);
        // A coalesced LD.b64 also defines its pinned high half (else that half
        // reads as an undefined, function-wide-live vreg).
        std::map<uint32_t, uint32_t>::const_iterator h =
            fn.regs.loadPairHi.find((uint32_t)def);
        if (h != fn.regs.loadPairHi.end()) {
          defined.insert(h->second); defB[b].insert(h->second);
        }
      }
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
      if (def >= 0) {
        Ext::go(iv, (uint32_t)def, p);
        std::map<uint32_t, uint32_t>::const_iterator h =
            fn.regs.loadPairHi.find((uint32_t)def);
        if (h != fn.regs.loadPairHi.end()) Ext::go(iv, h->second, p);
      }
    }
  }

  // Coalesced-load high halves are not allocated independently: fold each into
  // its low half's interval (so the pair register stays reserved until both are
  // dead) and drop it -- it is pinned to lo's register + 1 after the scan.
  for (std::map<uint32_t, uint32_t>::const_iterator it = fn.regs.loadPairHi.begin();
       it != fn.regs.loadPairHi.end(); ++it) {
    std::map<uint32_t, Interval>::iterator loI = iv.find(it->first);
    std::map<uint32_t, Interval>::iterator hiI = iv.find(it->second);
    if (loI != iv.end() && hiI != iv.end()) {
      if (hiI->second.start < loI->second.start) loI->second.start = hiI->second.start;
      if (hiI->second.end   > loI->second.end)   loI->second.end   = hiI->second.end;
    }
    if (hiI != iv.end()) iv.erase(hiI);
  }

  std::vector<Interval> ivs;
  ivs.reserve(iv.size());
  for (std::map<uint32_t, Interval>::iterator it = iv.begin(); it != iv.end(); ++it)
    ivs.push_back(it->second);
  std::sort(ivs.begin(), ivs.end(), byStart);

  // Wide vregs occupy a REGISTER PAIR {R, R+1} (64-bit b64/f64 values -> the
  // hardware reads {R[d+1],R[d]}). The allocator must reserve BOTH halves, or a
  // later value assigned R+1 clobbers the high word. narrowAddr() already
  // collapses .b64 pointer math to 32-bit, so in practice this covers genuine
  // 64-bit data (f64 / b64 loads).
  std::set<uint32_t> wide;
  for (unsigned b = 0; b < nb; ++b)
    for (unsigned i = 0; i < fn.blocks[b].insts.size(); ++i) {
      const ir::Inst &in = fn.blocks[b].insts[i];
      if (in.type != ir::Type::B64 && in.type != ir::Type::F64) continue;
      // A wide memory access does not make its ADDRESS a pair.  LD writes a
      // pair destination; ST reads a pair value from s2.  Other wide ops use
      // the ordinary rule that all register operands carry the wide value.
      if (in.op == ir::Op::LD || in.op == ir::Op::LDC) {
        if (in.dst.kind == ir::Operand::Reg) wide.insert(in.dst.value);
        continue;
      }
      if (in.op == ir::Op::ST) {
        if (in.s2.kind == ir::Operand::Reg) wide.insert(in.s2.value);
        continue;
      }
      if (in.dst.kind == ir::Operand::Reg) wide.insert(in.dst.value);
      const ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 3; ++k)
        if (s[k]->kind == ir::Operand::Reg) wide.insert(s[k]->value);
    }
  if (std::getenv("AEC_NO_PAIR")) wide.clear();   // A/B knob: disable pair-awareness.

  // --- 5. Pair-aware linear scan with real narrow and wide spill. ----------
  // Reserve the top of the register file as spill scratch so a reload/store
  // always has a physical register to land in: R247 = LMEM byte-offset address
  // and R248..R255 = the reloaded/spilled values.  An instruction reads/writes
  // at most four distinct virtual registers (dst + 3 sources), so eight value-
  // scratch registers cover the worst case where all four are wide pairs.
  // Allocatable range is R1..R246.  Realistic kernels peak at a handful of live values (the pre-RA
  // list scheduler sinks each value to its use), so the reservation is
  // invisible until pressure actually exceeds 246 -- exactly the regime where a
  // spill is required.  When nothing spills, the scratch band remains unused.
  const int kAllocTop    = (int)kRegisterCount - 10;  // R246: last allocatable.
  const int kAddrScratch = (int)kRegisterCount - 9;   // R247: LMEM address.
  const int kValScratch0 = (int)kRegisterCount - 8;   // R248..R255: values.

  std::set<int> freeSet;
  for (int r = 1; r <= kAllocTop; ++r) freeSet.insert(r);
  std::vector<int> active;               // indices into ivs, expired lazily.
  std::map<uint32_t, int> physOf;
  std::set<uint32_t> spilled;            // vregs with no physical register.
  std::map<uint32_t, int> slotOf;        // spilled vreg -> LMEM slot index.
  uint32_t spillCount = 0, maxPhys = 0;
  int nextSlot = 0;

  for (unsigned i = 0; i < ivs.size(); ++i) {
    std::vector<int> keep;
    for (unsigned a = 0; a < active.size(); ++a) {
      const Interval &e = ivs[active[a]];
      if (e.end < ivs[i].start) {
        freeSet.insert(e.phys);
        if (wide.count(e.vreg)) freeSet.insert(e.phys + 1);   // return both halves.
      } else keep.push_back(active[a]);
    }
    active.swap(keep);

    if (wide.count(ivs[i].vreg)) {                            // needs a pair.
      int reg = -1;
      for (std::set<int>::iterator it = freeSet.begin(); it != freeSet.end(); ++it)
        if (*it + 1 <= kAllocTop && freeSet.count(*it + 1)) { reg = *it; break; }
      if (reg >= 0) { freeSet.erase(reg); freeSet.erase(reg + 1); }
      else {
        // No consecutive pair is available.  Never alias this interval onto
        // an active pair: give it two LMEM words and materialize it at uses.
        spilled.insert(ivs[i].vreg);
        slotOf[ivs[i].vreg] = nextSlot; nextSlot += 2;
        ivs[i].phys = -1; physOf[ivs[i].vreg] = -1;
        continue;
      }
      ivs[i].phys = reg;
      physOf[ivs[i].vreg] = reg;
      active.push_back((int)i);
      if ((uint32_t)(reg + 1) > maxPhys) maxPhys = (uint32_t)(reg + 1);
      continue;
    }

    if (!freeSet.empty()) {                                   // free register: assign.
      int reg = *freeSet.begin(); freeSet.erase(reg);
      ivs[i].phys = reg;
      physOf[ivs[i].vreg] = reg;
      active.push_back((int)i);
      if ((uint32_t)reg > maxPhys) maxPhys = (uint32_t)reg;
      continue;
    }

    // No free register -> spill (Poletto & Sarkar SpillAtInterval): evict the
    // longest-lived narrow value.  If some active interval outlives i, take its
    // register for i and spill it; otherwise spill i itself.
    int victimA = -1, victimEnd = ivs[i].end;
    for (unsigned a = 0; a < active.size(); ++a) {
      const Interval &e = ivs[active[a]];
      if (!wide.count(e.vreg) && e.end > victimEnd) { victimEnd = e.end; victimA = (int)a; }
    }
    if (victimA >= 0) {
      Interval &v = ivs[active[victimA]];
      int reg = v.phys;
      spilled.insert(v.vreg); slotOf[v.vreg] = nextSlot++; physOf[v.vreg] = -1;
      v.phys = -1;
      active.erase(active.begin() + victimA);
      ivs[i].phys = reg;
      physOf[ivs[i].vreg] = reg;
      active.push_back((int)i);
      if ((uint32_t)reg > maxPhys) maxPhys = (uint32_t)reg;
    } else {
      spilled.insert(ivs[i].vreg); slotOf[ivs[i].vreg] = nextSlot++;
      ivs[i].phys = -1; physOf[ivs[i].vreg] = -1;
    }
  }
  const uint32_t spilledValues = (uint32_t)spilled.size();

  // Pin each coalesced high half to its low half's pair register + 1: the
  // LD.b64 wrote the low f32 into R and the high f32 into R+1.
  for (std::map<uint32_t, uint32_t>::const_iterator it = fn.regs.loadPairHi.begin();
       it != fn.regs.loadPairHi.end(); ++it) {
    std::map<uint32_t, int>::iterator lp = physOf.find(it->first);
    if (lp != physOf.end() && lp->second >= 0) {
      physOf[it->second] = lp->second + 1;
    } else if (spilled.count(it->first)) {
      // The b64 definition was stored as two words; make the explicit high-half
      // alias reload the second word when consumed independently.
      spilled.insert(it->second);
      slotOf[it->second] = slotOf[it->first] + 1;
      physOf[it->second] = -1;
    }
  }
  spillCount += spilledValues;
  if (!spilled.empty()) maxPhys = kRegisterCount - 1;   // scratch band in use.

  // --- 6. Rewrite Reg operands to physical registers, materializing spills. -
  // Fast path (no spill): rewrite in place, byte-identical to the old scan.
  // Spill path: rebuild each block, reloading every spilled source into a
  // value-scratch register just before the instruction and storing a spilled
  // destination back to its slot just after.
  for (unsigned b = 0; b < nb; ++b) {
    ir::BasicBlock &blk = fn.blocks[b];
    if (spilled.empty()) {
      for (unsigned i = 0; i < blk.insts.size(); ++i) {
        ir::Inst &in = blk.insts[i];
        ir::Operand *ops[4] = {&in.dst, &in.s1, &in.s2, &in.s3};
        for (int k = 0; k < 4; ++k)
          if (ops[k]->kind == ir::Operand::Reg) {
            std::map<uint32_t, int>::iterator it = physOf.find(ops[k]->value);
            *ops[k] = ir::Operand::phys((uint32_t)(it != physOf.end() ? it->second : 0));
          }
      }
      continue;
    }

    std::vector<ir::Inst> out;
    out.reserve(blk.insts.size());
    for (unsigned i = 0; i < blk.insts.size(); ++i) {
      ir::Inst in = blk.insts[i];
      ir::Operand *ops[4] = {&in.dst, &in.s1, &in.s2, &in.s3};

      // Assign a distinct scratch register (or consecutive pair) to every
      // spilled operand.  At most dst + 3 sources => at most eight registers.
      std::map<uint32_t, int> scr;
      int nextScr = kValScratch0;
      for (int k = 0; k < 4; ++k)
        if (ops[k]->kind == ir::Operand::Reg && spilled.count(ops[k]->value)
            && !scr.count(ops[k]->value)) {
          scr[ops[k]->value] = nextScr;
          nextScr += wide.count(ops[k]->value) ? 2 : 1;
        }

      const bool dstSpill = in.dst.kind == ir::Operand::Reg && spilled.count(in.dst.value);
      const uint32_t dstVreg = dstSpill ? in.dst.value : 0;

      // Reload each spilled source (once per distinct vreg) before `in`.
      std::set<uint32_t> reloaded;
      ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 3; ++k)
        if (srcs[k]->kind == ir::Operand::Reg && spilled.count(srcs[k]->value)
            && reloaded.insert(srcs[k]->value).second) {
          if (wide.count(srcs[k]->value))
            emitReloadWide(out, kAddrScratch, scr[srcs[k]->value], slotOf[srcs[k]->value]);
          else
            emitReload(out, kAddrScratch, scr[srcs[k]->value], slotOf[srcs[k]->value]);
        }

      // Rewrite operands: spilled -> its scratch, others -> physical register.
      for (int k = 0; k < 4; ++k)
        if (ops[k]->kind == ir::Operand::Reg) {
          std::map<uint32_t, int>::iterator s = scr.find(ops[k]->value);
          if (s != scr.end()) { *ops[k] = ir::Operand::phys((uint32_t)s->second); continue; }
          std::map<uint32_t, int>::iterator it = physOf.find(ops[k]->value);
          int p = (it != physOf.end() && it->second >= 0) ? it->second : 0;
          *ops[k] = ir::Operand::phys((uint32_t)p);
        }
      out.push_back(in);

      // Store a spilled destination back to its slot after `in`.
      if (dstSpill) {
        if (wide.count(dstVreg))
          emitStoreWide(out, kAddrScratch, scr[dstVreg], slotOf[dstVreg]);
        else
          emitStore(out, kAddrScratch, scr[dstVreg], slotOf[dstVreg]);
      }
    }
    blk.insts.swap(out);
  }

  fn.regs.maxPhys = maxPhys;
  fn.regs.spillCount = spillCount;
}

} // namespace regalloc
} // namespace aec
