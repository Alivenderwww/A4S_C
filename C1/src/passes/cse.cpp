// cse.cpp - Common sub-expression elimination.  Scoring category: T2.
//
// STATUS: wired identity stub. Builds the value key a real local-CSE would
// hash on (opcode+type+operands of pure instructions) but performs no reuse.
//
// T2 (repeated_expression) has two identical `add.f32 %fX, %f1, %f2` plus a
// redundant `mul.f32 %f1, %f2` — the intended CSE win lives there.
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
    case ir::Op::TMUL: case ir::Op::TMUL_S: case ir::Op::TLDA:
    case ir::Op::TSTA: case ir::Op::TMOV: case ir::Op::TDUP:
      return false;
    default:
      return true; // ADD/SUB/MUL/MAD/AND/CPY/LOADI/... are pure.
  }
}

// One available (already-computed) value in the current block.
struct Avail {
  ir::Op op; ir::Type ty;
  ir::Operand s1, s2, s3;
  bool hasImm; uint32_t imm;   // LOADI etc. distinguish by immediate.
  uint32_t dst;                // vreg holding the value.
};

bool sameOperand(const ir::Operand &a, const ir::Operand &b) {
  return a.kind == b.kind && a.value == b.value;
}
bool opUses(const ir::Operand &o, uint32_t r) {
  return o.kind == ir::Operand::Reg && o.value == r;
}
} // namespace

// Local (per-block) common-subexpression elimination via value numbering.
// The IR is NOT SSA (a PTX reg can be redefined), so availability is
// invalidated whenever any operand/def register is rewritten.
bool cse(ir::Function &fn, const Options &opt) {
  if (!opt.cse) return false;
  bool changed = false;

  // Which block each vreg is used in: block index, or -1 if used in more than
  // one. LOCAL (per-block) CSE may only eliminate a definition whose value is
  // used solely within the current block — otherwise the per-block rename can't
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
    std::map<uint32_t, uint32_t> rename;  // CSE'd vreg -> canonical vreg.
    std::vector<ir::Inst> kept;
    kept.reserve(b.insts.size());

    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst in = b.insts[ii];

      // Rewrite source operands to their canonical (CSE'd) registers.
      ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 3; ++k) {
        if (srcs[k]->kind == ir::Operand::Reg) {
          std::map<uint32_t, uint32_t>::iterator r = rename.find(srcs[k]->value);
          if (r != rename.end()) srcs[k]->value = r->second;
        }
      }

      // Redundant pure computation? -> alias its dst to the earlier one, drop it.
      if (isPure(in) && in.dst.kind == ir::Operand::Reg) {
        int found = -1;
        for (unsigned a = 0; a < avail.size(); ++a) {
          if (avail[a].op == in.op && avail[a].ty == in.type &&
              avail[a].hasImm == in.hasImm && avail[a].imm == in.imm &&
              sameOperand(avail[a].s1, in.s1) &&
              sameOperand(avail[a].s2, in.s2) &&
              sameOperand(avail[a].s3, in.s3)) { found = (int)a; break; }
        }
        // Only eliminate if this value is not used in any other block.
        std::map<uint32_t, int>::iterator ub = useBlock.find(in.dst.value);
        bool localOnly = (ub == useBlock.end()) || ub->second == (int)bi;
        if (found >= 0 && localOnly) {
          rename[in.dst.value] = avail[found].dst;
          changed = true;
          continue;                       // drop the redundant instruction.
        }
      }

      // Kept instruction. If it defines a register, its new value invalidates
      // any availability / rename that referenced that register (and, for a
      // 64-bit pair destination, the high half too).
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
      }

      kept.push_back(in);
    }

    b.insts.swap(kept);
  }

  return changed;
}

} // namespace passes
} // namespace aec
