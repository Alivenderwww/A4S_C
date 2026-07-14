// pred_alloc.cpp - virtual predicate -> P0..P7 allocation with GPR spill.  T4.
//
// The IR builder numbers every distinct PTX %pN with an incrementing id
// (predId 0,1,2,...).  A function that uses more than the 8 architectural
// predicates therefore emits ids >= 8, which the CModel rejects ("predicate
// field high bits set") on the CMPP destination -- the guard field is masked to
// 3 bits by the encoder, but the destination is not.
//
// This pass renumbers virtual predicates to physical P0..P7 by real liveness
// (backward dataflow over the CFG, so non-overlapping predicates share a slot),
// then, when more than 7 are simultaneously live, spills the surplus to GPRs:
//   * the defining `CMPP.<cmp> Pd, a, b` becomes `CMP.<cmp> Rval, a, b` -- CMP
//     writes the 0/1 boolean into a GPR;
//   * each guarded use re-materializes the predicate into the reserved scratch
//     predicate P7 with `CMPP.ne P7, Rval, Rzero` immediately before the use.
// The introduced GPR values (Rval per spilled predicate, one shared Rzero) flow
// through the subsequent linear-scan GPR allocator and its spiller like any
// other vreg.  Every realistic kernel keeps <=8 predicates, so the pass is a
// no-op unless the distinct-predicate count exceeds 8.
#include "aec/passes.h"
#include "aec/target.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace aec {
namespace regalloc {

namespace {

typedef std::set<uint32_t> PSet;

struct PInterval { uint32_t pv; int start; int end; int phys; };

bool pByStart(const PInterval &a, const PInterval &b) {
  if (a.start != b.start) return a.start < b.start;
  return a.pv < b.pv;
}

// Predicate defined (CMPP writing a predicate) / used (guard) by one inst.
void instPreds(const ir::Inst &in, int &defP, int &useP) {
  defP = (in.dst.kind == ir::Operand::Pred) ? (int)in.dst.value : -1;
  useP = in.guard;   // -1 if none; BRX and @%p guards both live here.
}

// Grow predicate v's interval to include position p (lambda-free, file style).
struct PExt {
  static void go(std::map<uint32_t, PInterval> &m, uint32_t v, int p) {
    std::map<uint32_t, PInterval>::iterator it = m.find(v);
    if (it == m.end()) { PInterval x; x.pv=v; x.start=p; x.end=p; x.phys=-1; m[v]=x; }
    else { if (p < it->second.start) it->second.start = p;
           if (p > it->second.end)   it->second.end = p; }
  }
};

} // namespace

void predAlloc(ir::Function &fn, const Options & /*opt*/) {
  const unsigned nb = fn.blocks.size();
  if (nb == 0) return;
  if (fn.regs.nextPred <= 8) return;   // ids already fit P0..P7 -- nothing to do.

  // --- 1. Global positions + per-block [lo,hi] (linear_scan layout). ------
  std::vector<int> blockLo(nb, -1), blockHi(nb, -1);
  int pos = 0;
  for (unsigned b = 0; b < nb; ++b) {
    if (fn.blocks[b].insts.empty()) { blockLo[b] = blockHi[b] = pos; continue; }
    blockLo[b] = pos; pos += (int)fn.blocks[b].insts.size(); blockHi[b] = pos - 1;
  }

  // --- 2. Local use/def, then backward liveness to a fixpoint. ------------
  std::vector<PSet> useB(nb), defB(nb), liveIn(nb), liveOut(nb);
  for (unsigned b = 0; b < nb; ++b) {
    PSet defined;
    const ir::BasicBlock &blk = fn.blocks[b];
    for (unsigned i = 0; i < blk.insts.size(); ++i) {
      int dp, up; instPreds(blk.insts[i], dp, up);
      if (up >= 0 && !defined.count((uint32_t)up)) useB[b].insert((uint32_t)up);
      if (dp >= 0) { defined.insert((uint32_t)dp); defB[b].insert((uint32_t)dp); }
    }
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (int b = (int)nb - 1; b >= 0; --b) {
      PSet out;
      for (unsigned s = 0; s < fn.blocks[b].succ.size(); ++s) {
        int sc = fn.blocks[b].succ[s];
        if (sc >= 0 && sc < (int)nb) out.insert(liveIn[sc].begin(), liveIn[sc].end());
      }
      PSet in = useB[b];
      for (PSet::iterator it = out.begin(); it != out.end(); ++it)
        if (!defB[b].count(*it)) in.insert(*it);
      if (out != liveOut[b]) { liveOut[b] = out; changed = true; }
      if (in != liveIn[b])   { liveIn[b] = in;   changed = true; }
    }
  }

  // --- 3. One interval per virtual predicate. -----------------------------
  std::map<uint32_t, PInterval> iv;
  for (unsigned b = 0; b < nb; ++b) {
    if (fn.blocks[b].insts.empty()) continue;
    for (PSet::iterator it = liveIn[b].begin(); it != liveIn[b].end(); ++it)
      PExt::go(iv, *it, blockLo[b]);
    for (PSet::iterator it = liveOut[b].begin(); it != liveOut[b].end(); ++it)
      PExt::go(iv, *it, blockHi[b]);
    int p = blockLo[b];
    for (unsigned i = 0; i < fn.blocks[b].insts.size(); ++i, ++p) {
      int dp, up; instPreds(fn.blocks[b].insts[i], dp, up);
      if (up >= 0) PExt::go(iv, (uint32_t)up, p);
      if (dp >= 0) PExt::go(iv, (uint32_t)dp, p);
    }
  }
  std::vector<PInterval> ivs;
  ivs.reserve(iv.size());
  for (std::map<uint32_t, PInterval>::iterator it = iv.begin(); it != iv.end(); ++it)
    ivs.push_back(it->second);
  std::sort(ivs.begin(), ivs.end(), pByStart);

  // --- 4. Linear scan.  P0..P6 allocatable; P7 reserved for spill re-test. -
  const int kScratchPred  = 7;   // physical predicate used to reload a spill.
  const int kAllocTopPred = 6;   // P0..P6 handed out to virtual predicates.
  std::set<int> freeP;
  for (int p = 0; p <= kAllocTopPred; ++p) freeP.insert(p);
  std::vector<int> active;                 // indices into ivs.
  std::map<uint32_t, int> physOf;          // pv -> P0..P6
  std::set<uint32_t> spilled;              // pv held in a GPR instead.

  for (unsigned i = 0; i < ivs.size(); ++i) {
    std::vector<int> keep;
    for (unsigned a = 0; a < active.size(); ++a) {
      const PInterval &e = ivs[active[a]];
      if (e.end < ivs[i].start) { if (e.phys >= 0) freeP.insert(e.phys); }
      else keep.push_back(active[a]);
    }
    active.swap(keep);

    if (!freeP.empty()) {
      int p = *freeP.begin(); freeP.erase(p);
      ivs[i].phys = p; physOf[ivs[i].pv] = p; active.push_back((int)i);
      continue;
    }
    // No free predicate -> spill the longest-lived (SpillAtInterval).
    int victimA = -1, vEnd = ivs[i].end;
    for (unsigned a = 0; a < active.size(); ++a)
      if (ivs[active[a]].phys >= 0 && ivs[active[a]].end > vEnd) {
        vEnd = ivs[active[a]].end; victimA = (int)a;
      }
    if (victimA >= 0) {
      PInterval &v = ivs[active[victimA]];
      int p = v.phys;
      spilled.insert(v.pv); physOf[v.pv] = -1; v.phys = -1;
      active.erase(active.begin() + victimA);
      ivs[i].phys = p; physOf[ivs[i].pv] = p; active.push_back((int)i);
    } else {
      spilled.insert(ivs[i].pv); ivs[i].phys = -1; physOf[ivs[i].pv] = -1;
    }
  }

  // --- 5. GPR backing for spilled predicates + one shared zero constant. --
  std::map<uint32_t, uint32_t> gprOf;
  uint32_t zeroReg = 0;
  if (!spilled.empty()) {
    for (PSet::iterator it = spilled.begin(); it != spilled.end(); ++it)
      gprOf[*it] = fn.regs.nextVReg++;
    zeroReg = fn.regs.nextVReg++;
  }

  // --- 6. Rewrite: remap physical predicates, materialize spills. ---------
  for (unsigned b = 0; b < nb; ++b) {
    ir::BasicBlock &blk = fn.blocks[b];
    std::vector<ir::Inst> out;
    out.reserve(blk.insts.size());
    for (unsigned i = 0; i < blk.insts.size(); ++i) {
      ir::Inst in = blk.insts[i];
      // Guard use: remap to a physical predicate, or re-materialize a spill.
      if (in.guard >= 0) {
        uint32_t gp = (uint32_t)in.guard;
        if (spilled.count(gp)) {
          ir::Inst rt; rt.op = ir::Op::CMPP; rt.type = ir::Type::U32;
          rt.dst = ir::Operand::pred((uint32_t)kScratchPred);
          rt.s1 = ir::Operand::reg(gprOf[gp]);
          rt.s2 = ir::Operand::reg(zeroReg);
          rt.modifier = (uint32_t)isa::Cmp::NE;   // P7 = (Rval != 0)
          rt.note = "pred-reload";
          out.push_back(rt);
          in.guard = kScratchPred;                // guardNeg preserved as-is.
        } else {
          in.guard = physOf[gp];
        }
      }
      // Predicate def: remap, or lower a spilled def to a GPR-writing CMP.
      if (in.dst.kind == ir::Operand::Pred) {
        uint32_t dp = in.dst.value;
        if (spilled.count(dp)) {
          in.op = ir::Op::CMP;                     // writes 0/1 to the GPR
          in.dst = ir::Operand::reg(gprOf[dp]);
          in.note = "pred-spill";
        } else {
          in.dst = ir::Operand::pred((uint32_t)physOf[dp]);
        }
      }
      out.push_back(in);
    }
    blk.insts.swap(out);
  }

  // --- 7. Materialize the zero constant once at the entry block. ----------
  if (!spilled.empty()) {
    ir::Inst li; li.op = ir::Op::LOADI; li.type = ir::Type::U32;
    li.dst = ir::Operand::reg(zeroReg); li.hasImm = true; li.imm = 0;
    fn.blocks[0].insts.insert(fn.blocks[0].insts.begin(), li);
  }

  fn.regs.spillCount += (uint32_t)spilled.size();
}

} // namespace regalloc
} // namespace aec
