// cse.cpp - Common-subexpression elimination + local redundant-load elimination.
// Scoring categories: T2 (redundant arithmetic) and T3 (redundant global loads).
//
// Two local (per-block) rewrites driven by value numbering:
//   * Pure-arithmetic CSE: a later `op.type s1,s2,s3` with identical operands
//     reuses the earlier result (e.g. T2 repeated_expression's duplicate
//     `add.f32 %fX,%f1,%f2`).
//   * Redundant-load elimination (RLE): a later `LD.space.type [Raddr]` from an
//     unchanged address reuses the earlier load when no aliasing store or
//     barrier intervened -- "keep the value in a register, don't reload", the
//     first memory optimization in the CUDA C++ Best Practices Guide. This is
//     legal for the C1 PTX subset because it uses only plain `ld.global`
//     (no .volatile/.acquire/.relaxed ordering); per PTX ISA §8.9.1 program
//     order is a per-thread total order, so a same-thread reload of an address
//     with no intervening aliasing store observes the same value (e.g. T3
//     repeated_global_load's `ld [%rd6]` issued twice into %f1 and %f3).
//
// The IR is NOT SSA (a PTX reg can be redefined), so an available value is
// invalidated whenever any operand/def register it names is rewritten. Loads
// are additionally invalidated by a store to the SAME space (disjoint AEC
// spaces -- gmem/pmem/smem/lmem/cmem -- cannot alias) or by a barrier/fence.
#include "aec/passes.h"

#include <map>
#include <vector>

namespace aec {
namespace passes {

namespace {
bool isPure(const ir::Inst &in) {
  if (in.isTerminator()) return false;
  switch (in.op) {
    case ir::Op::LD: case ir::Op::ST: case ir::Op::ATOM: case ir::Op::LDC:
    case ir::Op::CMPP: case ir::Op::CMP:
    case ir::Op::SYNC_WG: case ir::Op::SYNC_CT: case ir::Op::SSYNC:
    case ir::Op::MBAR:
      return false;
    default:
      return true; // ADD/SUB/MUL/MAD/AND/CPY/LOADI/... are pure.
  }
}

bool isBarrier(const ir::Inst &in) {
  switch (in.op) {
    case ir::Op::SYNC_WG: case ir::Op::SYNC_CT: case ir::Op::SSYNC:
    case ir::Op::MBAR:
      return true;
    default:
      return false;
  }
}

// One available (already-computed) pure value in the current block.
struct Avail {
  ir::Op op; ir::Type ty;
  ir::Operand s1, s2, s3;
  bool hasImm; uint32_t imm;   // LOADI etc. distinguish by immediate.
  uint32_t dst;                // vreg holding the value.
};

// One available (already-loaded) memory value in the current block.
struct AvailLoad {
  uint32_t space;   // memory space (in.modifier).
  uint32_t addr;    // address register (canonicalized).
  ir::Type ty;      // load width/type.
  int guard;        // guarding predicate (must match to reuse); -1 if none.
  uint32_t dst;     // vreg holding the loaded value.
};

bool sameOperand(const ir::Operand &a, const ir::Operand &b) {
  return a.kind == b.kind && a.value == b.value;
}
bool opUses(const ir::Operand &o, uint32_t r) {
  return o.kind == ir::Operand::Reg && o.value == r;
}
} // namespace

// Local (per-block) common-subexpression + redundant-load elimination via value
// numbering. The IR is NOT SSA (a PTX reg can be redefined), so availability is
// invalidated whenever any operand/def register is rewritten.
bool cse(ir::Function &fn, const Options &opt) {
  if (!opt.cse) return false;
  bool changed = false;

  // Which block each vreg is used in: block index, or -1 if used in more than
  // one. LOCAL (per-block) elimination may only drop a definition whose value is
  // used solely within the current block -- otherwise the per-block rename can't
  // reach cross-block uses and leaves a dangling reference.
  std::map<uint32_t, int> useBlock;
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    for (unsigned ii = 0; ii < fn.blocks[bi].insts.size(); ++ii) {
      const ir::Inst &in = fn.blocks[bi].insts[ii];
      const ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 3; ++k) {
        if (srcs[k]->kind != ir::Operand::Reg) continue;
        std::map<uint32_t, int>::iterator it = useBlock.find(srcs[k]->value);
        if (it == useBlock.end()) useBlock[srcs[k]->value] = (int)bi;
        else if (it->second != (int)bi) it->second = -1;
      }
    }
  }

  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    std::vector<Avail> avail;
    std::vector<AvailLoad> loads;
    std::map<uint32_t, uint32_t> rename;  // eliminated vreg -> canonical vreg.
    std::vector<ir::Inst> kept;
    kept.reserve(b.insts.size());

    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst in = b.insts[ii];

      // Rewrite source operands to their canonical (already-available) registers.
      ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 3; ++k) {
        if (srcs[k]->kind == ir::Operand::Reg) {
          std::map<uint32_t, uint32_t>::iterator r = rename.find(srcs[k]->value);
          if (r != rename.end()) srcs[k]->value = r->second;
        }
      }

      // A def is safe to eliminate locally only if its value is used solely in
      // this block (the per-block rename can't reach cross-block uses).
      bool localOnly = false;
      if (in.dst.kind == ir::Operand::Reg) {
        std::map<uint32_t, int>::iterator ub = useBlock.find(in.dst.value);
        localOnly = (ub == useBlock.end()) || ub->second == (int)bi;
      }

      // Redundant pure computation -> alias its dst to the earlier one, drop it.
      if (isPure(in) && in.dst.kind == ir::Operand::Reg) {
        int found = -1;
        for (unsigned a = 0; a < avail.size(); ++a) {
          if (avail[a].op == in.op && avail[a].ty == in.type &&
              avail[a].hasImm == in.hasImm && avail[a].imm == in.imm &&
              sameOperand(avail[a].s1, in.s1) &&
              sameOperand(avail[a].s2, in.s2) &&
              sameOperand(avail[a].s3, in.s3)) { found = (int)a; break; }
        }
        if (found >= 0 && localOnly) {
          rename[in.dst.value] = avail[found].dst;
          changed = true;
          continue;                       // drop the redundant instruction.
        }
      }

      // Redundant plain load from an unchanged address -> reuse the loaded reg.
      // Restricted to 32-bit loads (a 64-bit pair load has a two-register
      // footprint we don't value-number).
      const bool pairLoad = (in.type == ir::Type::B64 || in.type == ir::Type::F64);
      if (in.op == ir::Op::LD && in.dst.kind == ir::Operand::Reg &&
          in.s1.kind == ir::Operand::Reg && !pairLoad) {
        int found = -1;
        for (unsigned a = 0; a < loads.size(); ++a) {
          if (loads[a].space == in.modifier && loads[a].addr == in.s1.value &&
              loads[a].ty == in.type && loads[a].guard == in.guard) {
            found = (int)a; break;
          }
        }
        if (found >= 0 && localOnly) {
          rename[in.dst.value] = loads[found].dst;
          changed = true;
          continue;                       // drop the redundant load.
        }
      }

      // Kept instruction. First, a store/atomic to a space invalidates every
      // available load of that (only aliasing) space; a barrier invalidates all.
      if (in.op == ir::Op::ST || in.op == ir::Op::ATOM) {
        std::vector<AvailLoad> nl;
        nl.reserve(loads.size());
        for (unsigned a = 0; a < loads.size(); ++a)
          if (loads[a].space != in.modifier) nl.push_back(loads[a]);
        loads.swap(nl);
      } else if (isBarrier(in)) {
        loads.clear();
      }

      // A new register definition invalidates any available value / rename /
      // load that referenced that register (and, for a 64-bit pair def, its
      // high half too).
      if (in.dst.kind == ir::Operand::Reg) {
        uint32_t D = in.dst.value;
        bool pair = (in.type == ir::Type::B64 || in.type == ir::Type::F64);
        uint32_t D1 = (D + 1) & 0xff;
        std::vector<Avail> na;
        na.reserve(avail.size());
        for (unsigned a = 0; a < avail.size(); ++a) {
          const Avail &e = avail[a];
          bool hit = e.dst == D || opUses(e.s1, D) || opUses(e.s2, D) || opUses(e.s3, D);
          if (pair)
            hit = hit || e.dst == D1 || opUses(e.s1, D1) || opUses(e.s2, D1) || opUses(e.s3, D1);
          if (!hit) na.push_back(e);
        }
        avail.swap(na);

        std::vector<AvailLoad> nl;
        nl.reserve(loads.size());
        for (unsigned a = 0; a < loads.size(); ++a) {
          const AvailLoad &e = loads[a];
          bool hit = e.dst == D || e.addr == D || (pair && (e.dst == D1 || e.addr == D1));
          if (!hit) nl.push_back(e);
        }
        loads.swap(nl);

        rename.erase(D);
        if (pair) rename.erase(D1);
        for (std::map<uint32_t, uint32_t>::iterator it = rename.begin(); it != rename.end(); ) {
          if (it->second == D || (pair && it->second == D1)) rename.erase(it++);
          else ++it;
        }

        if (isPure(in)) {
          Avail e; e.op = in.op; e.ty = in.type; e.s1 = in.s1; e.s2 = in.s2;
          e.s3 = in.s3; e.hasImm = in.hasImm; e.imm = in.imm; e.dst = D;
          avail.push_back(e);
        }
        if (in.op == ir::Op::LD && in.s1.kind == ir::Operand::Reg && !pair) {
          AvailLoad e; e.space = in.modifier; e.addr = in.s1.value;
          e.ty = in.type; e.guard = in.guard; e.dst = D;
          loads.push_back(e);
        }
      }

      kept.push_back(in);
    }

    b.insts.swap(kept);
  }

  return changed;
}

} // namespace passes
} // namespace aec
