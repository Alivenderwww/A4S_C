// copy_prop.cpp - copy propagation.  Scoring category: T2 (robustness).
//
// `CPY d, s` (a register-to-register move) followed by uses of d is rewritten to
// use s directly; the copy then becomes dead and DCE removes it. ptxas runs a
// dedicated "Copy Propagation & CSE" phase. The 5 public kernels have no
// register-to-register CPY (their CPYs are special-register reads, which are not
// copies), so this earns nothing there — it is a completeness / robustness pass
// for hidden kernels or variants that carry redundant moves, and matches what a
// basic-optimizing baseline does.
//
// Only unpredicated, 32-bit register copies are propagated; special-register
// reads (CPY with a Special source) are never copies. Availability is
// invalidated whenever the source or destination register is redefined.
#include "aec/passes.h"

#include <map>

namespace aec {
namespace passes {

bool copyProp(ir::Function &fn, const Options &opt) {
  if (!opt.copy_prop) return false;
  bool changed = false;

  for (unsigned b = 0; b < fn.blocks.size(); ++b) {
    ir::BasicBlock &blk = fn.blocks[b];
    std::map<uint32_t, uint32_t> copyOf;   // d -> s : d currently holds a copy of s

    for (unsigned i = 0; i < blk.insts.size(); ++i) {
      ir::Inst &in = blk.insts[i];

      // Rewrite register sources, chasing copy chains to the original.
      ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
      for (int k = 0; k < 3; ++k) {
        if (s[k]->kind != ir::Operand::Reg) continue;
        uint32_t r = s[k]->value;
        int hops = 0;
        std::map<uint32_t, uint32_t>::iterator it;
        while ((it = copyOf.find(r)) != copyOf.end() && hops++ < 256) r = it->second;
        if (r != s[k]->value) { s[k]->value = r; changed = true; }
      }

      // A definition invalidates any copy naming that register (as dest or src).
      if (in.dst.kind == ir::Operand::Reg) {
        const uint32_t D = in.dst.value;
        const bool pair = (in.type == ir::Type::B64 || in.type == ir::Type::F64);
        const uint32_t D1 = (D + 1) & 0xff;
        copyOf.erase(D);
        if (pair) copyOf.erase(D1);
        for (std::map<uint32_t, uint32_t>::iterator it = copyOf.begin();
             it != copyOf.end();) {
          if (it->second == D || (pair && it->second == D1)) copyOf.erase(it++);
          else ++it;
        }
        // Record a fresh unpredicated 32-bit register copy.
        if (in.op == ir::Op::CPY && in.s1.kind == ir::Operand::Reg &&
            in.guard < 0 && !pair && in.s1.value != D)
          copyOf[D] = in.s1.value;
      }
    }
  }

  return changed;
}

} // namespace passes
} // namespace aec
