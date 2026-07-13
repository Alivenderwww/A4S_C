// ir_builder.cpp - PTX AST -> AEC IR (instruction selection + block splitting).
//
// Walks each kernel's statement list and lowers every PTX instruction to one
// or more ir::Inst carrying an AEC opcode/type/operands. PTX virtual registers
// become IR virtual registers (rewritten to physical regs later by the
// register allocator). Basic blocks are cut at labels and after branches so
// that cfg.cpp can wire up succ/pred.
//
// This is the T1 "basic lowering" surface: getting these encodings right is
// what makes PTX-01 (vector_add) produce a correct .aecbin.
#include "aec/passes.h"
#include "aec/ptx_ast.h"

#include <map>
#include <string>

namespace aec {

using isa::Op;
using isa::Type;
using isa::Cmp;
using isa::Space;

namespace {

bool isTypeName(const std::string &s) {
  static const char *kTypes[] = {
      "f32","f64","f16","bf16","s32","u32","s8","u8",
      "b32","b64","u64","s64","u16","s16","b16","pred"};
  for (unsigned i = 0; i < sizeof(kTypes) / sizeof(kTypes[0]); ++i)
    if (s == kTypes[i]) return true;
  return false;
}

Type mapType(const std::string &s) {
  if (s == "f32") return Type::F32;
  if (s == "f64") return Type::F64;
  if (s == "f16") return Type::F16;
  if (s == "bf16") return Type::BF16;
  if (s == "s32") return Type::S32;
  if (s == "u32") return Type::U32;
  if (s == "s8") return Type::S8;
  if (s == "u8") return Type::U8;
  if (s == "b32") return Type::B32;
  if (s == "b64" || s == "u64" || s == "s64") return Type::B64;
  if (s == "u16" || s == "s16" || s == "b16") return Type::B32; // 16b in a 32b reg
  return Type::NONE;
}

bool isFloatType(Type t) {
  return t == Type::F32 || t == Type::F64 || t == Type::F16 || t == Type::BF16;
}

// AEC addresses are 32-bit byte offsets and there is no 64-bit integer ALU
// (ADD/SUB/MUL have no .b64). PTX uses .u64/.b64 only for pointer arithmetic,
// so collapse those to 32-bit low-word ops: LD of a .u64 param reads the low
// 32 bits (a valid <4 GiB offset) into a single register, and add/sub/mul on
// pointers become 32-bit. This also avoids the illegal ADD.b64 and the b64
// register-pair clobber (a pair def writes {Rd,Rd+1}). See §1.2.
Type narrowAddr(Type t) { return t == Type::B64 ? Type::U32 : t; }

// First type-looking modifier ("mad.lo.u32" -> u32; "cvt.f32.f16" -> f32).
Type typeOfMods(const std::vector<std::string> &mods) {
  for (unsigned i = 0; i < mods.size(); ++i)
    if (isTypeName(mods[i])) return mapType(mods[i]);
  return Type::NONE;
}

// Second type-looking modifier ("cvt.f32.f16" -> f16), else NONE.
Type secondTypeOfMods(const std::vector<std::string> &mods) {
  int seen = 0;
  for (unsigned i = 0; i < mods.size(); ++i) {
    if (isTypeName(mods[i])) {
      if (seen == 1) return mapType(mods[i]);
      ++seen;
    }
  }
  return Type::NONE;
}

Cmp cmpOfMods(const std::vector<std::string> &mods) {
  for (unsigned i = 0; i < mods.size(); ++i) {
    if (mods[i] == "eq") return Cmp::EQ;
    if (mods[i] == "ne") return Cmp::NE;
    if (mods[i] == "lt") return Cmp::LT;
    if (mods[i] == "le") return Cmp::LE;
    if (mods[i] == "gt") return Cmp::GT;
    if (mods[i] == "ge") return Cmp::GE;
  }
  return Cmp::EQ;
}

uint32_t specialSelector(const std::string &name) {
  if (name == "tid.x")    return isa::TID_X;
  if (name == "ntid.x")   return isa::NTID_X;
  if (name == "ctaid.x")  return isa::CTAID_X;
  if (name == "nctaid.x") return isa::NCTAID_X;
  if (name == "laneid")   return isa::LANEID;
  if (name == "warpid")   return isa::WARPID;
  if (name == "tid.y")    return isa::TID_Y;
  if (name == "ntid.y")   return isa::NTID_Y;
  if (name == "ctaid.y")  return isa::CTAID_Y;
  if (name == "nctaid.y") return isa::NCTAID_Y;
  if (name == "tid.z")    return isa::TID_Z;
  if (name == "ntid.z")   return isa::NTID_Z;
  if (name == "ctaid.z")  return isa::CTAID_Z;
  if (name == "nctaid.z") return isa::NCTAID_Z;
  return isa::TID_X; // unknown special: default to tid.x (diagnostic only).
}

// The whole per-function lowering state, kept in a struct to avoid deep
// lambda recursion (C++11-friendly).
struct Builder {
  ir::Function *fn;
  std::map<std::string, uint32_t> vreg;   // "%rd1" -> vreg id.
  std::map<std::string, unsigned> paramOff;
  bool pendingSplit;                      // start a fresh block on next inst.

  Builder() : fn(0), pendingSplit(false) {}

  uint32_t freshReg() { return fn->regs.nextVReg++; }

  uint32_t regFor(const std::string &name) {
    std::map<std::string, uint32_t>::iterator it = vreg.find(name);
    if (it != vreg.end()) return it->second;
    uint32_t id = freshReg();
    vreg[name] = id;
    return id;
  }

  // "%p1" / "%p3" -> predicate id 1..7 (clamped to 0..7).
  static uint32_t predId(const std::string &name) {
    const char *p = name.c_str();
    while (*p && (*p < '0' || *p > '9')) ++p;
    uint32_t v = 0;
    while (*p >= '0' && *p <= '9') { v = v * 10 + (uint32_t)(*p - '0'); ++p; }
    return v & 0x7u;
  }

  void ensureBlock() {
    if (fn->blocks.empty() || pendingSplit) {
      ir::BasicBlock b;
      fn->blocks.push_back(b);
      pendingSplit = false;
    }
  }

  void startLabel(const std::string &label) {
    // Reuse a trailing empty, unlabeled block if we just split.
    if (!fn->blocks.empty() && fn->blocks.back().insts.empty() &&
        fn->blocks.back().label.empty()) {
      fn->blocks.back().label = label;
      pendingSplit = false;
      return;
    }
    ir::BasicBlock b;
    b.label = label;
    fn->blocks.push_back(b);
    pendingSplit = false;
  }

  void emit(const ir::Inst &in) {
    ensureBlock();
    fn->blocks.back().insts.push_back(in);
    if (in.isTerminator()) pendingSplit = true;
  }

  // Turn a PTX operand into an IR register operand, materializing immediates
  // through a LOADI so arithmetic ops always see registers.
  ir::Operand asReg(const ptx::Operand &op, Type ty) {
    if (op.kind == ptx::Operand::Reg)
      return ir::Operand::reg(regFor(op.name));
    if (op.kind == ptx::Operand::Special) {
      uint32_t d = freshReg();
      ir::Inst c;
      c.op = Op::CPY; c.type = ty == Type::NONE ? Type::U32 : ty;
      c.dst = ir::Operand::reg(d);
      c.s1  = ir::Operand::special(specialSelector(op.name));
      emit(c);
      return ir::Operand::reg(d);
    }
    if (op.kind == ptx::Operand::Imm || op.kind == ptx::Operand::FloatImm) {
      uint32_t d = freshReg();
      ir::Inst li;
      li.op = Op::LOADI;
      li.type = ty == Type::NONE ? Type::U32 : ty;
      li.dst = ir::Operand::reg(d);
      li.hasImm = true;
      li.imm = (uint32_t)op.imm;
      emit(li);
      return ir::Operand::reg(d);
    }
    if (op.kind == ptx::Operand::Mem)
      return ir::Operand::reg(regFor(op.name));
    return ir::Operand();
  }

  void lowerBinary(Op opc, Type ty, const ptx::Instruction &s) {
    if (s.operands.size() < 3) return;
    ty = narrowAddr(ty);                       // 64-bit pointer math -> 32-bit.
    ir::Inst in;
    in.op = opc; in.type = ty;
    in.dst = ir::Operand::reg(regFor(s.operands[0].name));
    in.s1  = asReg(s.operands[1], ty);
    in.s2  = asReg(s.operands[2], ty);
    emit(in);
  }

  void lowerUnary(Op opc, Type ty, const ptx::Instruction &s) {
    if (s.operands.size() < 2) return;
    ir::Inst in;
    in.op = opc; in.type = narrowAddr(ty);
    in.dst = ir::Operand::reg(regFor(s.operands[0].name));
    in.s1  = asReg(s.operands[1], ty);
    emit(in);
  }

  void lowerFMA(Op opc, Type ty, const ptx::Instruction &s) {
    if (s.operands.size() < 4) return;
    ir::Inst in;
    in.op = opc; in.type = ty;
    in.dst = ir::Operand::reg(regFor(s.operands[0].name));
    in.s1  = asReg(s.operands[1], ty);
    in.s2  = asReg(s.operands[2], ty);
    in.s3  = asReg(s.operands[3], ty);
    emit(in);
  }

  void translate(const ptx::Instruction &s);
  void run(const ptx::Kernel &k);
};

void Builder::translate(const ptx::Instruction &s) {
  const std::string &m = s.mnemonic;
  Type ty = typeOfMods(s.mods);

  if (m == "ld") {
    // ld.param.* / ld.global.* / ld.shared.*
    bool isParam = false, isShared = false;
    for (unsigned i = 0; i < s.mods.size(); ++i) {
      if (s.mods[i] == "param") isParam = true;
      if (s.mods[i] == "shared") isShared = true;
    }
    ir::Inst in;
    in.op = Op::LD; in.type = narrowAddr(ty == Type::NONE ? Type::B32 : ty);
    in.dst = ir::Operand::reg(regFor(s.operands[0].name));
    if (isParam) {
      in.modifier = (uint32_t)Space::PMEM;
      in.hasImm = true;
      std::map<std::string, unsigned>::iterator it =
          paramOff.find(s.operands.size() > 1 ? s.operands[1].name : "");
      in.imm = (it != paramOff.end()) ? it->second : 0;
      in.note = "param:" + (s.operands.size() > 1 ? s.operands[1].name : "");
    } else {
      in.modifier = isShared ? (uint32_t)Space::SMEM : (uint32_t)Space::GMEM;
      if (s.operands.size() > 1)
        in.s1 = ir::Operand::reg(regFor(s.operands[1].name)); // address reg
    }
    emit(in);
    return;
  }

  if (m == "st") {
    bool isShared = false;
    for (unsigned i = 0; i < s.mods.size(); ++i)
      if (s.mods[i] == "shared") isShared = true;
    ir::Inst in;
    in.op = Op::ST; in.type = ty == Type::NONE ? Type::B32 : ty;
    in.modifier = isShared ? (uint32_t)Space::SMEM : (uint32_t)Space::GMEM;
    if (s.operands.size() >= 2) {
      in.s1 = ir::Operand::reg(regFor(s.operands[0].name)); // address reg
      in.s2 = asReg(s.operands[1], in.type);                // value
    }
    emit(in);
    return;
  }

  if (m == "mov") {
    if (s.operands.size() < 2) return;
    ir::Inst in;
    in.dst = ir::Operand::reg(regFor(s.operands[0].name));
    const ptx::Operand &src = s.operands[1];
    if (src.kind == ptx::Operand::Special) {
      in.op = Op::CPY; in.type = ty == Type::NONE ? Type::U32 : ty;
      in.s1 = ir::Operand::special(specialSelector(src.name));
    } else if (src.kind == ptx::Operand::Reg) {
      in.op = Op::CPY; in.type = narrowAddr(ty == Type::NONE ? Type::B32 : ty);
      in.s1 = ir::Operand::reg(regFor(src.name));
    } else { // Imm / FloatImm
      in.op = Op::LOADI; in.type = ty == Type::NONE ? Type::U32 : ty;
      in.hasImm = true; in.imm = (uint32_t)src.imm;
    }
    emit(in);
    return;
  }

  // PTX mad.f32 on sm_70 is FUSED (single rounding = fma); only integer mad
  // maps to AEC MAD (mul rounds, then add rounds). See C1_实现流程分析.md §1.5.
  if (m == "mad")  { lowerFMA(isFloatType(ty) ? Op::FMA : Op::MAD, ty, s); return; }
  if (m == "fma")  { lowerFMA(Op::FMA, ty, s); return; }

  if (m == "add")  { lowerBinary(Op::ADD, ty, s); return; }
  if (m == "sub")  { lowerBinary(Op::SUB, ty, s); return; }
  if (m == "mul")  { lowerBinary(Op::MUL, ty, s); return; }
  if (m == "div")  { lowerBinary(Op::DIV, ty, s); return; }
  if (m == "min")  { lowerBinary(Op::MIN, ty, s); return; }
  if (m == "max")  { lowerBinary(Op::MAX, ty, s); return; }
  if (m == "rem")  { lowerBinary(Op::DIV, ty, s); return; }  // TODO: true remainder

  // Unary arithmetic / bit / SFU ops (dst, src). Common in activation-style and
  // math-heavy kernels beyond the 5 public examples.
  if (m == "neg")  { lowerUnary(Op::NEG, ty, s); return; }
  if (m == "abs")  { lowerUnary(Op::ABS, ty, s); return; }
  if (m == "not")  { lowerUnary(Op::NOT, ty == Type::NONE ? Type::B32 : ty, s); return; }
  if (m == "popc") { lowerUnary(Op::POPC, ty == Type::NONE ? Type::B32 : ty, s); return; }
  if (m == "bfind"){ lowerUnary(Op::FLO, ty == Type::NONE ? Type::U32 : ty, s); return; }
  if (m == "sqrt") { lowerUnary(Op::SQRT, ty == Type::NONE ? Type::F32 : ty, s); return; }
  if (m == "rcp")  { lowerUnary(Op::RCP, ty == Type::NONE ? Type::F32 : ty, s); return; }
  if (m == "rsqrt"){ lowerUnary(Op::RSQ, ty == Type::NONE ? Type::F32 : ty, s); return; }
  if (m == "sin")  { lowerUnary(Op::SIN, ty == Type::NONE ? Type::F32 : ty, s); return; }
  if (m == "cos")  { lowerUnary(Op::COS, ty == Type::NONE ? Type::F32 : ty, s); return; }
  if (m == "ex2")  { lowerUnary(Op::EXP, ty == Type::NONE ? Type::F32 : ty, s); return; }
  if (m == "lg2")  { lowerUnary(Op::LOG, ty == Type::NONE ? Type::F32 : ty, s); return; }
  if (m == "and")  { lowerBinary(Op::AND, ty == Type::NONE ? Type::B32 : ty, s); return; }
  if (m == "or")   { lowerBinary(Op::OR,  ty == Type::NONE ? Type::B32 : ty, s); return; }
  if (m == "xor")  { lowerBinary(Op::XOR, ty == Type::NONE ? Type::B32 : ty, s); return; }
  if (m == "shl")  { lowerBinary(Op::SHL, ty == Type::NONE ? Type::B32 : ty, s); return; }
  if (m == "shr")  { lowerBinary(Op::SHR, ty == Type::NONE ? Type::B32 : ty, s); return; }

  if (m == "setp") {
    if (s.operands.size() < 3) return;
    ir::Inst in;
    in.op = Op::CMPP; in.type = ty == Type::NONE ? Type::U32 : ty;
    in.dst = ir::Operand::pred(predId(s.operands[0].name));
    in.s1  = asReg(s.operands[1], in.type);
    in.s2  = asReg(s.operands[2], in.type);
    in.modifier = (uint32_t)cmpOfMods(s.mods);
    emit(in);
    return;
  }

  if (m == "cvt") {
    if (s.operands.size() < 2) return;
    ir::Inst in;
    Type dstT = ty == Type::NONE ? Type::F32 : ty;
    Type srcT = secondTypeOfMods(s.mods);
    if (srcT == Type::NONE) srcT = Type::F32;
    // Opcode by float/int kind of (dst, src); source type must be ENCODED
    // (goes to Pred/Ctrl[13:10] via modifier) — dropping it made the golden
    // read cvt.f32.f16 as a no-op f32 copy. See C1_实现流程分析.md §1.4.
    bool df = isFloatType(dstT), sf = isFloatType(srcT);
    in.op = df ? (sf ? Op::CVTFF : Op::CVTIF) : (sf ? Op::CVTFI : Op::CVTII);
    in.type = dstT;
    in.modifier = (uint32_t)srcT;   // encoded as source type in [13:10].
    in.dst = ir::Operand::reg(regFor(s.operands[0].name));
    in.s1  = asReg(s.operands[1], srcT);
    in.note = "cvt";
    emit(in);
    return;
  }

  if (m == "bra") {
    ir::Inst in;
    std::string label;
    for (unsigned i = 0; i < s.operands.size(); ++i)
      if (s.operands[i].kind == ptx::Operand::Label) label = s.operands[i].name;
    in.target = label;
    if (!s.guardPred.empty()) {
      in.op = Op::BRX;
      in.guard = (int)predId(s.guardPred);
      if (s.guardNegated) in.note = "negated-guard(TODO)";
    } else {
      in.op = Op::BR;
      in.hasImm = true; // target resolved to imm in lower.cpp
    }
    emit(in);
    return;
  }

  if (m == "bar" || m == "barrier") {
    ir::Inst in; in.op = Op::SYNC_WG; emit(in); return;
  }

  // Top-level `ret` in an .entry kernel exits the thread -> AEC HALT, which
  // per Track-B §A.2 1.1 is uniform and completes the warp. An AEC RET pops the
  // per-warp call stack, empty at kernel level -> execution error.
  if (m == "ret") { ir::Inst in; in.op = Op::HALT; emit(in); return; }
  if (m == "exit"){ ir::Inst in; in.op = Op::HALT; emit(in); return; }

  // Unknown mnemonic: RECORD it (so the driver fails loudly by default rather
  // than emitting silently-wrong code) and leave a breadcrumb for --lenient.
  fn->unhandled.push_back(m);
  ir::Inst in;
  in.op = Op::CPY; in.type = Type::B32;
  if (!s.operands.empty() && s.operands[0].kind == ptx::Operand::Reg) {
    in.dst = ir::Operand::reg(regFor(s.operands[0].name));
    in.s1  = in.dst;
  }
  in.note = "UNHANDLED:" + m;
  emit(in);
}

void Builder::run(const ptx::Kernel &k) {
  // Param block layout (byte offsets) + IR param table.
  unsigned off = 0;
  for (unsigned i = 0; i < k.params.size(); ++i) {
    ir::Param p;
    p.name = k.params[i].name;
    p.type = mapType(k.params[i].type);
    p.bytes = k.params[i].bytes ? k.params[i].bytes : 4;
    // Natural alignment.
    if (p.bytes && (off % p.bytes)) off += p.bytes - (off % p.bytes);
    p.offset = off;
    off += p.bytes;
    paramOff[p.name] = p.offset;
    fn->params.push_back(p);
  }

  for (unsigned i = 0; i < k.body.size(); ++i) {
    const ptx::Instruction &s = k.body[i];
    if (!s.label.empty() && s.mnemonic.empty()) {
      startLabel(s.label);
      continue;
    }
    if (s.mnemonic.empty()) continue;
    translate(s);
  }

  // Guarantee a terminator so the CFG/encoder are well-formed.
  if (fn->blocks.empty()) { ir::BasicBlock b; fn->blocks.push_back(b); }
  ir::BasicBlock &last = fn->blocks.back();
  if (last.insts.empty() || !last.insts.back().isTerminator()) {
    ir::Inst r; r.op = Op::HALT; last.insts.push_back(r);  // kernel exit = HALT.
  }
}

} // namespace

ir::Program buildIR(const ptx::Module &mod, const Options & /*opt*/) {
  ir::Program prog;
  prog.ptxVersion = mod.version;
  prog.ptxTarget = mod.target;
  for (unsigned ki = 0; ki < mod.kernels.size(); ++ki) {
    ir::Function fn;
    fn.name = mod.kernels[ki].name;
    Builder b;
    b.fn = &fn;
    b.run(mod.kernels[ki]);
    prog.functions.push_back(fn);
  }
  return prog;
}

} // namespace aec
