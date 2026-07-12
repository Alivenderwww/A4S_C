// ptx_ast.h - Faithful AST for the PTX subset consumed by C1.
//
// The lexer/parser fill these structures directly from the .ptx text; no
// target semantics leak in here. Instruction selection happens later in the
// IR builder (src/ir/ir_builder.cpp).
#ifndef AEC_PTX_AST_H
#define AEC_PTX_AST_H

#include <cstdint>
#include <string>
#include <vector>

namespace aec {
namespace ptx {

// One operand of a PTX instruction.
struct Operand {
  enum Kind {
    None,
    Reg,      // %r5, %rd1, %f3, %p1     -> name holds the full token.
    Special,  // %tid.x, %ctaid.x ...    -> name holds the special name.
    Imm,      // 4, 0, 31 ...            -> imm holds the value.
    FloatImm, // 0f00000000              -> imm holds the raw 32-bit pattern.
    Mem,      // [%rd5] / [param_a]      -> name holds the inner token.
    Label     // branch target label     -> name holds the label.
  };
  Kind kind = None;
  std::string name;   // register/special/label/mem-inner token.
  uint64_t    imm = 0;
};

// A single PTX statement, e.g. "mad.lo.u32 %r5, %r3, %r4, %r2;".
struct Instruction {
  std::string mnemonic;              // "mad", "ld", "setp", "bra" ...
  std::vector<std::string> mods;     // dotted modifiers: {"lo","u32"} / {"global","f32"}
  std::vector<Operand> operands;
  std::string guardPred;             // "%p1" when written as "@%p1 ...", else empty.
  bool guardNegated = false;         // "@!%p1"
  std::string label;                 // set when this "statement" is a label def.
  int line = 0;                      // source line for diagnostics.
};

// A .param declaration in the kernel signature.
struct Param {
  std::string name;
  std::string type;   // "u64", "u32", "f32" ...
  unsigned    bytes = 0;
};

// A .reg declaration, e.g. ".reg .b32 %r<6>;".
struct RegDecl {
  std::string type;   // "b32", "b64", "f32", "pred" ...
  std::string prefix; // "%r", "%rd", "%f", "%p" ...
  unsigned    count = 0;
};

struct Kernel {
  std::string name;
  std::vector<Param>   params;
  std::vector<RegDecl> regs;
  std::vector<Instruction> body; // statements + label markers, in order.
};

struct Module {
  std::string version;      // ".version" value.
  std::string target;       // ".target" value.
  unsigned    addressSize = 64;
  std::vector<Kernel> kernels;
};

} // namespace ptx
} // namespace aec

#endif // AEC_PTX_AST_H
