// mad_contract.cpp - contract `MUL; ADD` into one `MAD`.  Categories: T2-T5.
//
// The graded metric is dynamic instruction count, and AEC MAD is NON-fused
// (spec §6.2: the multiply rounds, then the add rounds), so
//     MUL t,a,b ; ADD d,t,c   ==   MAD d,a,b,c
// bit-for-bit for both float (round-then-round) and integer (mod 2^32), while
// removing the MUL. Contract when the MUL result feeds exactly one same-type,
// same-guard ADD as one addend, and the multiply operands are unchanged in
// between. (A MUL result used more than once — e.g. a shared address offset
// idx*4 feeding several ADDs — has use-count > 1 and is left alone.)
#include "aec/passes.h"

#include <map>
#include <vector>

namespace aec {
namespace passes {

namespace {
// FP32 only. Integer MUL;ADD chains are address arithmetic: a single-use one is
// exactly the `MUL off,idx,4; ADD addr,base,off` pattern strength reduction
// turns into an add-recurrence (better inside a loop), so contracting it here
// would hide it from SR. Shared address offsets have use-count > 1 and are never
// contracted regardless.
bool contractible(ir::Type t) {
  return t == ir::Type::F32;
}
bool usesReg(const ir::Inst &in, uint32_t r) {
  const ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
  for (int k = 0; k < 3; ++k)
    if (s[k]->kind == ir::Operand::Reg && s[k]->value == r) return true;
  return false;
}
} // namespace

bool madContract(ir::Function &fn, const Options &opt) {
  if (!opt.mad_contract) return false;
  bool changedAny = false;

  // Function-wide source use-count per vreg.
  std::map<uint32_t, int> uc;
  for (unsigned b = 0; b < fn.blocks.size(); ++b)
    for (unsigned i = 0; i < fn.blocks[b].insts.size(); ++i) {
      const ir::Inst &in = fn.blocks[b].insts[i];
      const ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 3; ++k)
        if (s[k]->kind == ir::Operand::Reg) uc[s[k]->value]++;
    }

  for (unsigned b = 0; b < fn.blocks.size(); ++b) {
    ir::BasicBlock &blk = fn.blocks[b];
    std::vector<char> del(blk.insts.size(), 0);
    bool blockChanged = false;

    for (unsigned i = 0; i < blk.insts.size(); ++i) {
      const ir::Inst mul = blk.insts[i];
      if (mul.op != ir::Op::MUL || mul.dst.kind != ir::Operand::Reg) continue;
      if (!contractible(mul.type)) continue;
      if (mul.s1.kind != ir::Operand::Reg || mul.s2.kind != ir::Operand::Reg) continue;
      const uint32_t t = mul.dst.value, a = mul.s1.value, bb = mul.s2.value;
      std::map<uint32_t, int>::iterator it = uc.find(t);
      if (it == uc.end() || it->second != 1) continue;   // t used exactly once

      // Walk forward to t's single use; a/b must be unchanged until then.
      int jj = -1;
      for (unsigned j = i + 1; j < blk.insts.size(); ++j) {
        const ir::Inst &in = blk.insts[j];
        if (in.dst.kind == ir::Operand::Reg &&
            (in.dst.value == a || in.dst.value == bb)) break;   // operand clobbered
        if (!usesReg(in, t)) continue;
        // This is the use. It must be a clean same-type/guard ADD with t as
        // exactly one of the two addends.
        if (in.op == ir::Op::ADD && in.type == mul.type && in.guard == mul.guard) {
          bool t1 = in.s1.kind == ir::Operand::Reg && in.s1.value == t;
          bool t2 = in.s2.kind == ir::Operand::Reg && in.s2.value == t;
          bool t3 = in.s3.kind == ir::Operand::Reg && in.s3.value == t;
          if ((t1 ^ t2) && !t3) jj = (int)j;
        }
        break;                                          // first (only) use of t
      }
      if (jj < 0) continue;

      ir::Inst &add = blk.insts[jj];
      ir::Operand other = (add.s1.kind == ir::Operand::Reg && add.s1.value == t)
                              ? add.s2 : add.s1;
      add.op = ir::Op::MAD;
      add.s1 = ir::Operand::reg(a);
      add.s2 = ir::Operand::reg(bb);
      add.s3 = other;
      del[i] = 1;
      blockChanged = true;
      changedAny = true;
    }

    if (blockChanged) {
      std::vector<ir::Inst> keep;
      keep.reserve(blk.insts.size());
      for (unsigned i = 0; i < blk.insts.size(); ++i)
        if (!del[i]) keep.push_back(blk.insts[i]);
      blk.insts.swap(keep);
    }
  }

  return changedAny;
}

} // namespace passes
} // namespace aec
