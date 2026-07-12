// lower.cpp - Final legalization + flatten to a linear instruction stream.
// Scoring categories: T1 (basic lowering) + shared back end.
//
// Runs last, after regalloc/scheduling. It assigns every block a flattened
// program-counter (instruction index), resolves branch target labels to
// absolute PCs (written into the branch immediate), and returns the linear
// instruction list the encoder consumes 1:1.
#include "aec/passes.h"

#include <map>
#include <vector>

namespace aec {
namespace codegen {

std::vector<ir::Inst> lower(ir::Function &fn, const Options & /*opt*/) {
  // 1. Assign each block its first PC and record label -> PC.
  std::map<std::string, int> labelPC;
  int pc = 0;
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    fn.blocks[bi].firstPC = pc;
    if (!fn.blocks[bi].label.empty())
      labelPC[fn.blocks[bi].label] = pc;
    pc += (int)fn.blocks[bi].insts.size();
  }

  // 2. Flatten, resolving branch targets to absolute instruction indices.
  std::vector<ir::Inst> flat;
  flat.reserve((size_t)pc);
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst in = b.insts[ii];
      if (in.isBranch() && !in.target.empty()) {
        std::map<std::string, int>::iterator it = labelPC.find(in.target);
        int dest = (it != labelPC.end()) ? it->second : (int)pc; // fall off end
        in.hasImm = true;
        in.imm = (uint32_t)dest;
      }
      flat.push_back(in);
    }
  }

  fn.instructionCount = (uint32_t)flat.size();
  return flat;
}

} // namespace codegen
} // namespace aec
