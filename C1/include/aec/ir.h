// ir.h - C1 internal IR: functions -> basic blocks -> target-oriented insts.
//
// The IR is deliberately close to AEC MIR: each ir::Inst already carries an
// AEC opcode + type + operands. Optimization passes rewrite these in place;
// the encoder consumes them almost 1:1. Registers are *virtual* until the
// register allocator rewrites Operand::Reg into Operand::Phys.
#ifndef AEC_IR_H
#define AEC_IR_H

#include <cstdint>
#include <string>
#include <vector>

#include "aec/isa.h"

namespace aec {
namespace ir {

using isa::Op;
using isa::Type;

// A virtual/physical register, immediate, special selector or predicate.
struct Operand {
  enum Kind { None, Reg, Phys, Imm, Special, Pred };
  Kind     kind  = None;
  uint32_t value = 0; // vreg id / phys reg / imm / special selector / pred id.

  static Operand reg(uint32_t v)     { Operand o; o.kind = Reg;     o.value = v; return o; }
  static Operand phys(uint32_t v)    { Operand o; o.kind = Phys;    o.value = v; return o; }
  static Operand imm(uint32_t v)     { Operand o; o.kind = Imm;     o.value = v; return o; }
  static Operand special(uint32_t v) { Operand o; o.kind = Special; o.value = v; return o; }
  static Operand pred(uint32_t v)    { Operand o; o.kind = Pred;    o.value = v; return o; }

  bool isReg()  const { return kind == Reg || kind == Phys; }
  bool none()   const { return kind == None; }
};

// One target-oriented instruction.
struct Inst {
  Op       op   = Op::RET;
  Type     type = Type::NONE;
  int      guard = -1;         // guarding predicate id (P0..P7); -1 = none.
                               // For BRX this is the branch predicate.
  bool     guardNeg = false;   // predicate negated: `@!%p` / BRX takes when !P.
  Operand  dst, s1, s2, s3;
  bool     hasImm = false;
  uint32_t imm    = 0;
  uint32_t modifier = 0;       // cmp op / mem space / tensor layout.
  std::string target;          // branch target label (BR/BRX).
  std::string note;            // optional annotation surfaced by objdump.

  bool isBranch() const { return op == Op::BR || op == Op::BRX || op == Op::JMP; }
  bool isTerminator() const { return isBranch() || op == Op::RET || op == Op::HALT; }
};

struct BasicBlock {
  std::string label;              // block label ("" for the entry block).
  std::vector<Inst> insts;
  std::vector<int>  succ;         // successor block indices (filled by cfg.cpp).
  std::vector<int>  pred;         // predecessor block indices.
  int firstPC = -1;               // flattened instruction index (encoder pass).
};

// Per-function register bookkeeping (populated by the IR builder / regalloc).
struct RegInfo {
  uint32_t nextVReg = 1;          // R0 reserved as scratch/zero.
  uint32_t nextPred = 0;          // P0..P7.
  uint32_t maxPhys  = 0;          // highest physical reg assigned (diagnostic).
  uint32_t spillCount = 0;        // spill slots emitted (diagnostic).
};

struct Param {
  std::string name;
  Type   type = Type::NONE;
  unsigned bytes = 0;
  unsigned offset = 0;            // byte offset in the param block (pmem).
};

struct Function {
  std::string name;
  std::vector<Param> params;
  std::vector<BasicBlock> blocks;
  RegInfo regs;

  // Diagnostics filled by the back end (surfaced in the perf report).
  uint32_t instructionCount = 0;
  uint32_t dualIssuePairs   = 0;

  // PTX mnemonics the front end did not lower (compile fails unless --lenient),
  // so an unsupported op is a loud error, never silently-wrong code.
  std::vector<std::string> unhandled;
};

struct Program {
  std::string ptxVersion;
  std::string ptxTarget;
  std::vector<Function> functions;
};

} // namespace ir
} // namespace aec

#endif // AEC_IR_H
