// licm.cpp - Loop-invariant code motion.  Scoring category: T2.
//
// STATUS: wired identity stub. Detects natural loops from the CFG back-edges
// (a real LICM's prerequisite) but hoists nothing yet.
//
// PTX-02 (invariant_poly) computes `add.f32 %f5,%f1,%f2` etc. inside its LOOP
// although %f1/%f2 are loop-invariant param values — that is the hoist target.
#include "aec/passes.h"

#include <map>
#include <set>
#include <vector>

namespace aec {
namespace passes {

namespace {
// Pure, non-memory instructions are always hoistable. LD/LDC are hoistable ONLY
// when the loop performs no memory writes (no aliasing store can change the
// loaded value) — that case is redundant-load-elimination / load reuse (T3).
bool hoistable(const ir::Inst &in) {
  if (in.isTerminator()) return false;
  switch (in.op) {
    case ir::Op::LD: case ir::Op::LDC: case ir::Op::ST: case ir::Op::ATOM:
    case ir::Op::CMP: case ir::Op::CMPP:
    case ir::Op::SYNC_WG: case ir::Op::SYNC_CT: case ir::Op::SSYNC:
    case ir::Op::MBAR:
    case ir::Op::TMUL: case ir::Op::TMUL_S: case ir::Op::TLDA:
    case ir::Op::TSTA: case ir::Op::TMOV: case ir::Op::TDUP:
    case ir::Op::RCP: case ir::Op::RSQ: case ir::Op::SIN: case ir::Op::COS:
    case ir::Op::EXP: case ir::Op::LOG: case ir::Op::SQRT: case ir::Op::RDTSC:
      return false;
    default:
      return true;  // ADD/SUB/MUL/MAD/FMA/AND/SHL/CPY/LOADI/CVT... are pure.
  }
}

// A memory write / fence: its presence in a loop means loads can't be treated
// as loop-invariant (an aliasing store could change the value).
bool writesMemory(const ir::Inst &in) {
  switch (in.op) {
    case ir::Op::ST: case ir::Op::ATOM: case ir::Op::TSTA:
    case ir::Op::SYNC_WG: case ir::Op::SYNC_CT: case ir::Op::SSYNC:
    case ir::Op::MBAR:
      return true;
    default:
      return false;
  }
}
} // namespace

// Loop-invariant code motion. Hoists pure instructions whose operands are all
// loop-invariant out of a natural loop into its (fall-through) entry
// predecessor. Conservative: only single-back-edge loops with one fall-through
// entry predecessor immediately before the header are transformed.
bool licm(ir::Function &fn, const Options &opt) {
  if (!opt.licm) return false;
  bool changedAny = false;

  for (unsigned li = 0; li < fn.blocks.size(); ++li) {
    for (unsigned s = 0; s < fn.blocks[li].succ.size(); ++s) {
      int h = fn.blocks[li].succ[s];
      if (h > (int)li) continue;             // not a back-edge.
      const int lo = h, hi = (int)li;        // contiguous loop body [lo..hi].

      // Entry predecessor: a pred of the header outside the loop. Require
      // exactly one, sitting immediately before the header and falling through
      // (no terminator) — then it dominates the loop and runs once.
      int P = -1, entries = 0;
      for (unsigned p = 0; p < fn.blocks[h].pred.size(); ++p) {
        int pb = fn.blocks[h].pred[p];
        if (pb < lo || pb > hi) { P = pb; ++entries; }
      }
      if (entries != 1 || P != h - 1) continue;
      if (!fn.blocks[P].insts.empty() && fn.blocks[P].insts.back().isTerminator())
        continue;

      // Registers defined inside the loop, and how many times.
      std::map<uint32_t, int> defCount;
      for (int b = lo; b <= hi; ++b)
        for (unsigned ii = 0; ii < fn.blocks[b].insts.size(); ++ii)
          if (fn.blocks[b].insts[ii].dst.kind == ir::Operand::Reg)
            defCount[fn.blocks[b].insts[ii].dst.value]++;

      // A store-free loop lets loop-invariant loads be hoisted (load reuse, T3).
      bool storeFree = true;
      for (int b = lo; b <= hi && storeFree; ++b)
        for (unsigned ii = 0; ii < fn.blocks[b].insts.size(); ++ii)
          if (writesMemory(fn.blocks[b].insts[ii])) { storeFree = false; break; }

      // Iteratively mark invariant instructions.
      std::set<uint32_t> invariantReg;      // regs whose sole loop def is invariant.
      std::set<long> marked;                // encoded (block<<20 | inst).
      bool progress = true;
      while (progress) {
        progress = false;
        for (int b = lo; b <= hi; ++b) {
          for (unsigned ii = 0; ii < fn.blocks[b].insts.size(); ++ii) {
            long id = ((long)b << 20) | ii;
            if (marked.count(id)) continue;
            const ir::Inst &in = fn.blocks[b].insts[ii];
            const bool canHoist = hoistable(in) ||
                (storeFree && (in.op == ir::Op::LD || in.op == ir::Op::LDC));
            if (!canHoist || in.dst.kind != ir::Operand::Reg) continue;
            if (defCount[in.dst.value] != 1) continue;   // single loop def only.
            const ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
            bool ok = true;
            for (int k = 0; k < 3; ++k) {
              if (srcs[k]->kind == ir::Operand::Reg &&
                  defCount.count(srcs[k]->value) &&        // defined in loop...
                  !invariantReg.count(srcs[k]->value))     // ...but not (yet) invariant.
                ok = false;
            }
            if (ok) {
              marked.insert(id);
              invariantReg.insert(in.dst.value);
              progress = true;
            }
          }
        }
      }
      if (marked.empty()) continue;

      // Remove marked instructions from the loop (preserving order) and append
      // them, in program order, to the entry predecessor.
      std::vector<ir::Inst> hoisted;
      for (int b = lo; b <= hi; ++b) {
        std::vector<ir::Inst> keep;
        keep.reserve(fn.blocks[b].insts.size());
        for (unsigned ii = 0; ii < fn.blocks[b].insts.size(); ++ii) {
          long id = ((long)b << 20) | ii;
          if (marked.count(id)) hoisted.push_back(fn.blocks[b].insts[ii]);
          else keep.push_back(fn.blocks[b].insts[ii]);
        }
        fn.blocks[b].insts.swap(keep);
      }
      for (unsigned k = 0; k < hoisted.size(); ++k)
        fn.blocks[P].insts.push_back(hoisted[k]);
      changedAny = true;
    }
  }

  if (changedAny) buildCFG(fn);
  return changedAny;
}

} // namespace passes
} // namespace aec
