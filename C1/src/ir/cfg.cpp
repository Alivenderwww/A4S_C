// cfg.cpp - Basic-block control-flow graph construction.
//
// buildIR (ir_builder.cpp) already split the instruction stream into blocks at
// labels and after branches. Here we wire BasicBlock::succ/pred from the block
// terminators so the optimization passes (T2) can reason about control flow.
#include "aec/passes.h"

#include <map>

namespace aec {

void buildCFG(ir::Function &fn) {
  const int n = (int)fn.blocks.size();

  // label -> block index.
  std::map<std::string, int> labelIndex;
  for (int i = 0; i < n; ++i)
    if (!fn.blocks[i].label.empty())
      labelIndex[fn.blocks[i].label] = i;

  // Reset any prior edges (buildCFG may be re-run after transforms).
  for (int i = 0; i < n; ++i) {
    fn.blocks[i].succ.clear();
    fn.blocks[i].pred.clear();
  }

  for (int i = 0; i < n; ++i) {
    ir::BasicBlock &b = fn.blocks[i];
    const ir::Inst *term =
        b.insts.empty() ? 0 : &b.insts.back();

    bool fallsThrough = true;
    if (term && term->isBranch()) {
      // Resolve the branch target label to a block index.
      std::map<std::string, int>::iterator it = labelIndex.find(term->target);
      if (it != labelIndex.end())
        b.succ.push_back(it->second);
      // Unconditional branch (BR/JMP) does not fall through; BRX does.
      if (term->op == ir::Op::BR || term->op == ir::Op::JMP)
        fallsThrough = false;
    } else if (term && (term->op == ir::Op::RET || term->op == ir::Op::HALT)) {
      fallsThrough = false;
    }

    if (fallsThrough && i + 1 < n)
      b.succ.push_back(i + 1);
  }

  // Derive predecessors from successors.
  for (int i = 0; i < n; ++i)
    for (unsigned s = 0; s < fn.blocks[i].succ.size(); ++s)
      fn.blocks[fn.blocks[i].succ[s]].pred.push_back(i);
}

} // namespace aec
