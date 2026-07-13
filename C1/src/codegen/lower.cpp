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
  // 0. Legalize 64-bit memory ops to the Track-B legal-type matrix (§4.1/§8.2):
  //    LD .f64      -> LD .b64        (one op, loads the 8-byte pair {Rd,Rd+1})
  //    ST .f64/.b64 -> two ST .b32    (low word at [Ra], high word at [Ra+4])
  // ST has no 64-bit width, so the pair is stored as two 32-bit words. R0 is the
  // scratch register (regalloc hands out R1..255) used to hold Ra+4. Runs before
  // PC assignment so the inserted instructions are counted.
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    std::vector<ir::Inst> out;
    out.reserve(b.insts.size());
    for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
      ir::Inst in = b.insts[ii];
      if (in.op == ir::Op::LD && in.type == ir::Type::F64) {
        in.type = ir::Type::B64;
        out.push_back(in);
      } else if (in.op == ir::Op::ST &&
                 (in.type == ir::Type::F64 || in.type == ir::Type::B64)) {
        const uint32_t addr = in.s1.value;   // address register (physical)
        const uint32_t val  = in.s2.value;   // value pair base (physical)
        ir::Inst li; li.op = ir::Op::LOADI; li.type = ir::Type::NONE;
        li.dst = ir::Operand::phys(0); li.hasImm = true; li.imm = 4;
        out.push_back(li);
        ir::Inst ad; ad.op = ir::Op::ADD; ad.type = ir::Type::U32;
        ad.dst = ir::Operand::phys(0);
        ad.s1 = ir::Operand::phys(addr); ad.s2 = ir::Operand::phys(0);
        out.push_back(ad);
        ir::Inst lo = in; lo.type = ir::Type::B32;   // ST.b32 [addr], Rlow
        out.push_back(lo);
        ir::Inst hi = in; hi.type = ir::Type::B32;   // ST.b32 [R0], Rhigh
        hi.s1 = ir::Operand::phys(0);
        hi.s2 = ir::Operand::phys(val + 1);
        out.push_back(hi);
      } else {
        out.push_back(in);
      }
    }
    b.insts.swap(out);
  }

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
