// cse.cpp - Common sub-expression elimination.  Scoring category: T2.
//
// STATUS: wired identity stub. Builds the value key a real local-CSE would
// hash on (opcode+type+operands of pure instructions) but performs no reuse.
//
// PTX-02 (invariant_poly) has two identical `add.f32 %fX, %f1, %f2` in a loop
// body plus a redundant `mul.f32` — the intended CSE win lives there.
#include "aec/passes.h"

#include <map>
#include <string>

namespace aec {
namespace passes {

namespace {
bool isPure(const ir::Inst &in) {
  if (in.isTerminator()) return false;
  switch (in.op) {
    case ir::Op::LD: case ir::Op::ST: case ir::Op::ATOM:
    case ir::Op::CMPP: case ir::Op::CMP:
    case ir::Op::SYNC_WG: case ir::Op::SYNC_CT:
    case ir::Op::TMUL: case ir::Op::TLDA: case ir::Op::TSTA:
      return false;
    default:
      return true; // ADD/SUB/MUL/MAD/AND/... are pure.
  }
}

std::string opKey(const ir::Inst &in) {
  // A textual value-number key over (op,type,src operands).
  std::string k;
  k += (char)((int)in.op & 0xff);
  k += (char)((int)in.type & 0xff);
  const ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
  for (int i = 0; i < 3; ++i) {
    k += (char)('0' + (int)s[i]->kind);
    uint32_t v = s[i]->value;
    for (int b = 0; b < 4; ++b) k += (char)((v >> (b * 8)) & 0xff);
  }
  return k;
}
} // namespace

bool cse(ir::Function &fn, const Options &opt) {
  if (!opt.cse) return false;
  bool changed = false;

  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    std::map<std::string, uint32_t> avail; // value key -> defining vreg.

    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst &in = b.insts[ii];
      if (!isPure(in) || in.dst.kind != ir::Operand::Reg) continue;

      std::string key = opKey(in);
      std::map<std::string, uint32_t>::iterator it = avail.find(key);
      if (it != avail.end()) {
        // TODO(T2): replace this instruction with a CPY from it->second, and
        // rewrite later uses of in.dst to it->second, then set changed=true.
        (void)it;
      } else {
        avail[key] = in.dst.value;
      }
      // TODO(T2): invalidate `avail` entries whose operands were just
      // redefined (needed once rewriting is enabled).
    }
  }

  return changed;
}

} // namespace passes
} // namespace aec
