// driver.cpp - Pipeline orchestration, IR->encoding, disassembly, cycle model.
//
// Ties every phase together (frontend -> IR -> CFG -> passes -> regalloc ->
// sched -> lower -> encode -> image) and provides the disassembler and a
// heuristic cycle estimate. -O0 skips the optimization passes; -O2/-O3 run
// them in order. Register allocation, scheduling and final lowering always run
// (they are required to produce a legal image, not optional optimizations).
#include "aec/driver.h"
#include "aec/passes.h"
#include "aec/isa.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <set>
#include <vector>

namespace aec {

namespace {

// ir::Inst -> isa::Fields for encoding.
isa::Fields toFields(const ir::Inst &in) {
  isa::Fields f;
  f.op = in.op;
  f.type = in.type;

  if (in.op == isa::Op::BRX) {
    f.predicate = (in.guard >= 0) ? (uint8_t)(in.guard & 0x7) : 0;
  } else if (in.guard >= 0) {
    f.predicate = (uint8_t)(in.guard & 0x7);
  } else {
    f.predicate = isa::kPredicateNone;
  }
  f.pred_neg = in.guardNeg;

  f.dst  = (uint16_t)in.dst.value;   // Phys / Pred id.
  f.src1 = (uint16_t)in.s1.value;    // Phys / Special selector.
  f.src2 = (uint16_t)in.s2.value;
  f.src3 = (uint16_t)in.s3.value;
  if (in.hasImm) f.imm = in.imm;
  f.modifier = in.modifier;
  return f;
}

uint32_t paramBlockBytes(const ir::Function &fn) {
  uint32_t bytes = 0;
  for (unsigned i = 0; i < fn.params.size(); ++i) {
    uint32_t end = (uint32_t)fn.params[i].offset + fn.params[i].bytes;
    if (end > bytes) bytes = end;
  }
  return bytes;
}

// --- Latency-aware static cycle estimate (drives the Agent's config choice) --
//
// The old estimate summed a per-instruction cost, which is ANTI-correlated with
// real cycles for latency-bound loops: unrolling ADDS instructions, so it made
// -O3 look worse than -O0 and the agent picked the slowest config. This model
// instead, per block, computes a latency-weighted critical-path makespan via a
// scoreboard over the (already scheduled) order, and multiplies a counted
// self-loop's body by its trip count. Unrolling then wins correctly: fewer loop
// iterations + overlapped independent loads -> lower estimate. Mirrors the
// sim's cycle model (sched/list_sched.cpp latencyOf) so static ranking tracks
// the simulator. Computed pre-regalloc, on vregs (so LOADI constants are
// unique and self-loop trips are recoverable).

int estLatency(const ir::Inst &in) {   // cycles until the result is ready
  switch (in.op) {
    case ir::Op::LD:
      return (in.modifier == (uint32_t)isa::Space::GMEM ||
              in.modifier == (uint32_t)isa::Space::LMEM) ? 32 : 2;
    case ir::Op::TMUL: case ir::Op::TMUL_S: return 16;
    case ir::Op::TLDA: case ir::Op::TSTA: return 6;
    case ir::Op::DIV: case ir::Op::RCP: case ir::Op::RSQ: case ir::Op::SQRT:
    case ir::Op::SIN: case ir::Op::COS: case ir::Op::EXP: case ir::Op::LOG:
      return 12;
    default: return 1;
  }
}

// Value of a vreg if it is defined by exactly one LOADI immediate.
bool loadiConst(const ir::Function &fn, uint32_t reg, uint32_t &val) {
  int found = 0;
  for (unsigned b = 0; b < fn.blocks.size(); ++b)
    for (unsigned i = 0; i < fn.blocks[b].insts.size(); ++i) {
      const ir::Inst &in = fn.blocks[b].insts[i];
      if (in.op == ir::Op::LOADI && in.dst.kind == ir::Operand::Reg &&
          in.dst.value == reg && in.hasImm) { val = in.imm; ++found; }
    }
  return found == 1;
}

// Exact trip of a constant single-block self-loop (role-based, so it survives
// scheduling reorder). Returns 0 = UNKNOWN (not a recognizable constant loop);
// a fully-unrolled loop legitimately returns 1, which must be distinct from
// unknown so the caller does not fall back to a default trip and over-weight it.
uint32_t blockTrip(const ir::Function &fn, const ir::BasicBlock &b, unsigned bi) {
  bool self = false;
  for (unsigned s = 0; s < b.succ.size(); ++s) self |= (b.succ[s] == (int)bi);
  if (!self || b.insts.empty()) return 0;
  const ir::Inst &term = b.insts.back();
  if (term.op != ir::Op::BRX || term.guard < 0) return 0;
  const uint32_t pg = (uint32_t)(term.guard & 0x7);
  const ir::Inst *cmp = 0;
  for (unsigned i = 0; i < b.insts.size(); ++i) {
    const ir::Inst &in = b.insts[i];
    if (in.op == ir::Op::CMPP && in.dst.kind == ir::Operand::Pred &&
        (in.dst.value & 0x7) == pg) cmp = &in;
  }
  if (!cmp || cmp->s1.kind != ir::Operand::Reg || cmp->s2.kind != ir::Operand::Reg)
    return 0;
  uint32_t bound = 0, step = 0;
  if (!loadiConst(fn, cmp->s2.value, bound)) return 0;
  bool haveStep = false;
  for (unsigned i = 0; i < b.insts.size(); ++i) {
    const ir::Inst &in = b.insts[i];
    if (in.op == ir::Op::ADD && in.dst.kind == ir::Operand::Reg &&
        in.dst.value == cmp->s1.value && in.s1.kind == ir::Operand::Reg &&
        in.s1.value == cmp->s1.value && in.s2.kind == ir::Operand::Reg)
      haveStep = loadiConst(fn, in.s2.value, step);
  }
  if (!haveStep || step == 0 || bound == 0) return 0;
  uint32_t t = bound / step;
  return t < 1 ? 1 : t;
}

// Latency-aware makespan of one block: max of the dependency critical path
// (scoreboard) and a 2-wide dual-issue bound.
uint64_t blockMakespan(const ir::BasicBlock &b) {
  const uint32_t PRED_NS = 0x10000u;
  std::map<uint32_t, int> ready;   // reg / predicate -> cycle result available
  int critical = 0;
  for (unsigned i = 0; i < b.insts.size(); ++i) {
    const ir::Inst &in = b.insts[i];
    int t = 0;
    const ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
    for (int k = 0; k < 3; ++k)
      if (s[k]->kind == ir::Operand::Reg) {
        std::map<uint32_t, int>::iterator it = ready.find(s[k]->value);
        if (it != ready.end()) t = std::max(t, it->second);
      }
    if (in.guard >= 0) {
      std::map<uint32_t, int>::iterator it = ready.find(PRED_NS | (uint32_t)(in.guard & 0x7));
      if (it != ready.end()) t = std::max(t, it->second);
    }
    int done = t + estLatency(in);
    if (in.dst.kind == ir::Operand::Reg) ready[in.dst.value] = done;
    else if (in.dst.kind == ir::Operand::Pred)
      ready[PRED_NS | (in.dst.value & 0x7)] = done;
    critical = std::max(critical, done);
  }
  uint64_t issueBound = (b.insts.size() + 1) / 2;   // 2-wide dual issue
  return std::max<uint64_t>((uint64_t)critical, issueBound);
}

uint64_t estimateCyclesIR(const ir::Function &fn) {
  const unsigned nb = fn.blocks.size();
  const uint64_t DEFAULT_TRIP = 32;   // representative trip for an unknown bound
  // Per-block loop multiplier. A successor edge to an index <= the current
  // block is a back-edge (our CFGs are reducible); weight every block in the
  // loop body [header..tail] by the trip so a hot loop body dominates the
  // estimate. Without this, a multi-block / param-trip loop (e.g. GEMM's
  // K-loop) is counted once and -O0's bloated body looks as cheap as -O2's
  // optimized one -> the agent would wrongly pick -O0. A constant single-block
  // self-loop uses its exact trip; anything else uses DEFAULT_TRIP. Nested
  // loops compound (product), which is what we want.
  std::vector<uint64_t> mult(nb, 1);
  for (unsigned bi = 0; bi < nb; ++bi)
    for (unsigned s = 0; s < fn.blocks[bi].succ.size(); ++s) {
      int hdr = fn.blocks[bi].succ[s];
      if (hdr < 0 || hdr > (int)bi) continue;                // forward edge
      // Exact trip for a recognizable constant single-block self-loop (blockTrip
      // returns the real count, incl. 1 for a fully-unrolled loop); DEFAULT_TRIP
      // when the bound is unreadable (0 = unknown: multi-block, or a param trip
      // like GEMM's K-loop), which must NOT leave the hot body unweighted.
      uint32_t ct = (hdr == (int)bi) ? blockTrip(fn, fn.blocks[bi], bi) : 0;
      uint64_t trip = (ct >= 1) ? (uint64_t)ct : DEFAULT_TRIP;
      for (int i = hdr; i <= (int)bi; ++i) mult[(unsigned)i] *= trip;
    }
  uint64_t total = 0;
  for (unsigned bi = 0; bi < nb; ++bi)
    total += blockMakespan(fn.blocks[bi]) * mult[bi];
  return total;
}

void runOptPasses(ir::Function &fn, const Options &opt) {
  // NOTE: do NOT early-return on -O0. The performance passes are each gated by
  // their own flag (all false at -O0, so this is a no-op there), but pred_opt
  // is a CORRECTNESS transform that must run at every level -- see
  // Options::applyOptLevel. A blanket `if (O0) return;` here previously skipped
  // it and made -O0 miscompile divergent bounds guards.
  //
  // Two light rounds so an implemented pass can feed the next.
  for (int round = 0; round < 2; ++round) {
    if (opt.const_prop)   passes::constProp(fn, opt);
    if (opt.copy_prop)    passes::copyProp(fn, opt);
    if (opt.cse)          passes::cse(fn, opt);
    if (opt.mad_contract) passes::madContract(fn, opt);
    if (opt.licm)         passes::licm(fn, opt);
    if (opt.dce)          passes::dce(fn, opt);
  }
  if (opt.pred_opt)     passes::predOpt(fn, opt);   // correctness: every level.
  buildCFG(fn); // transforms may have changed control flow.
}

const char *specialName(uint16_t sel) {
  switch (sel) {
    case isa::TID_X: return "%tid.x";   case isa::NTID_X: return "%ntid.x";
    case isa::CTAID_X: return "%ctaid.x"; case isa::NCTAID_X: return "%nctaid.x";
    case isa::LANEID: return "%laneid"; case isa::WARPID: return "%warpid";
    case isa::TID_Y: return "%tid.y";   case isa::NTID_Y: return "%ntid.y";
    case isa::CTAID_Y: return "%ctaid.y"; case isa::NCTAID_Y: return "%nctaid.y";
    case isa::TID_Z: return "%tid.z";   case isa::NTID_Z: return "%ntid.z";
    case isa::CTAID_Z: return "%ctaid.z"; case isa::NCTAID_Z: return "%nctaid.z";
    default: return 0;
  }
}

const char *cmpName(uint32_t c) {
  static const char *n[6] = {"eq", "ne", "lt", "le", "gt", "ge"};
  return (c < 6) ? n[c] : "?";
}

const char *spaceName(uint32_t s) {
  static const char *n[5] = {"gmem", "smem", "cmem", "lmem", "pmem"};
  return (s < 5) ? n[s] : "?";
}

} // namespace

// --- Compile pipeline -----------------------------------------------------
bool compile(const ptx::Module &m, const Options &opt, binfmt::Image &image,
             ir::Program &prog, CompileReport &report, std::string &err) {
  prog = buildIR(m, opt);
  if (prog.functions.empty()) { err = "no kernel to compile"; return false; }

  // Fail loudly on any PTX op we did not lower (never emit silently-wrong code).
  if (!opt.lenient) {
    std::string bad;
    for (unsigned i = 0; i < prog.functions.size(); ++i)
      for (unsigned k = 0; k < prog.functions[i].unhandled.size(); ++k) {
        if (!bad.empty()) bad += ", ";
        bad += prog.functions[i].unhandled[k];
      }
    if (!bad.empty()) {
      err = "unhandled PTX op(s): " + bad + "  (add a lowering in ir_builder.cpp, "
            "or pass --lenient to emit a placeholder)";
      return false;
    }
  }

  // The scaffold emits the first kernel as the image entry (public tests are
  // single-kernel). TODO: emit every kernel with per-kernel symbols.
  ir::Function &fn = prog.functions[0];

  buildCFG(fn);
  runOptPasses(fn, opt);
  if (opt.unroll) {
    while (passes::loopRotate(fn, opt)) {}          // while -> do-while (enables the below)
    while (passes::strengthReduce(fn, opt)) {}      // address multiplies -> add-recurrence
    passes::unrollLoops(fn, opt);                   // cut per-iteration loop control.
  }
  sched::listSchedule(fn, opt);   // pre-RA: schedule on vregs (fewer false deps).
  regalloc::predAlloc(fn, opt);   // map virtual predicates -> P0..P7 (+GPR spill).
  const uint64_t estCycles = estimateCyclesIR(fn);   // pre-RA: vregs unique.
  regalloc::linearScan(fn, opt);

  std::vector<ir::Inst> flat = codegen::lower(fn, opt);

  // Encode + collect param relocations.
  image = binfmt::Image();
  image.code.reserve(flat.size());
  for (unsigned i = 0; i < flat.size(); ++i) {
    const ir::Inst &in = flat[i];
    image.code.push_back(isa::encode(toFields(in)));
    if (in.op == isa::Op::LD &&
        in.modifier == (uint32_t)isa::Space::PMEM) {
      binfmt::RelocEntry r;
      r.instrIndex = i;
      r.kind = binfmt::RELOC_PARAM_ADDR;
      r.addend = in.imm;
      image.relocs.push_back(r);
    }
  }

  // Symbols: kernel entry + block labels.
  binfmt::SymbolEntry entry;
  entry.name = fn.name; entry.value = 0; entry.kind = 0;
  image.symbols.push_back(entry);
  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    if (fn.blocks[bi].label.empty() || fn.blocks[bi].firstPC < 0) continue;
    binfmt::SymbolEntry s;
    s.name = fn.blocks[bi].label;
    s.value = (uint32_t)fn.blocks[bi].firstPC;
    s.kind = 1;
    image.symbols.push_back(s);
  }

  const uint32_t pbytes = paramBlockBytes(fn);
  image.data.assign(pbytes, 0);
  image.header.entryPC = 0;
  image.header.instructionCount = (uint32_t)image.code.size();
  image.header.paramBytes = pbytes;

  report.kernel = fn.name;
  report.instructionCount = (uint32_t)flat.size();
  report.numBasicBlocks = (uint32_t)fn.blocks.size();
  report.numVirtualRegisters = fn.regs.nextVReg > 1 ? fn.regs.nextVReg - 1 : 0;
  report.numPhysicalRegisters = fn.regs.maxPhys;
  report.spillLoads = 0;          // spiller is a stub; no spill code emitted
  report.spillStores = 0;
  report.dualIssuePairs = fn.dualIssuePairs;
  report.paramBytes = pbytes;
  report.estCycles = estCycles;   // latency-aware heuristic (not the graded metric).

  // Count input PTX instructions (statements with a mnemonic, first kernel).
  for (unsigned ki = 0; ki < m.kernels.size(); ++ki)
    for (unsigned s = 0; s < m.kernels[ki].body.size(); ++s)
      if (!m.kernels[ki].body[s].mnemonic.empty()) ++report.numPtxInstructions;

  // Static instruction-mix + predicate + data-flow-depth diagnostics (spec §B.3),
  // over the final flattened stream.
  std::set<uint32_t> preds;
  std::map<uint32_t, uint32_t> regDepth;   // last-def data-flow depth per reg
  uint32_t maxDepth = 0;
  for (unsigned i = 0; i < flat.size(); ++i) {
    const ir::Inst &in = flat[i];
    if (in.op == isa::Op::BR || in.op == isa::Op::BRX || in.op == isa::Op::JMP)
      ++report.branchCount;
    if (in.op == isa::Op::LD || in.op == isa::Op::LDC) ++report.loadCount;
    if (in.op == isa::Op::ST) ++report.storeCount;
    if (in.guard >= 0) preds.insert((uint32_t)(in.guard & 7));
    if (in.op == isa::Op::CMPP && in.dst.kind == ir::Operand::Pred)
      preds.insert(in.dst.value & 7);
    uint32_t d = 0;
    const ir::Operand *srcs[3] = {&in.s1, &in.s2, &in.s3};
    for (int k = 0; k < 3; ++k)
      if (srcs[k]->isReg()) {
        std::map<uint32_t, uint32_t>::iterator it = regDepth.find(srcs[k]->value);
        if (it != regDepth.end() && it->second > d) d = it->second;
      }
    ++d;
    if (in.dst.isReg()) regDepth[in.dst.value] = d;
    if (d > maxDepth) maxDepth = d;
  }
  report.numPredicates = (uint32_t)preds.size();
  report.dependencyDepth = maxDepth;

  if (opt.verbose) {
    std::fprintf(stderr,
        "[driver] kernel=%s aec_insts=%u bb=%u vreg=%u phys=%u pred=%u "
        "ld=%u st=%u br=%u depth=%u\n",
        fn.name.c_str(), report.instructionCount, report.numBasicBlocks,
        report.numVirtualRegisters, report.numPhysicalRegisters,
        report.numPredicates, report.loadCount, report.storeCount,
        report.branchCount, report.dependencyDepth);
  }
  return true;
}

bool compileFile(const std::string &inPath, const std::string &outPath,
                 const Options &opt, CompileReport &report, std::string &err) {
  FILE *f = std::fopen(inPath.c_str(), "rb");
  if (!f) { err = "cannot open input: " + inPath; return false; }
  std::string src;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) src.append(buf, n);
  std::fclose(f);

  ptx::Module mod;
  if (!ptx::parse(src, mod, err)) return false;

  ir::Program prog;
  binfmt::Image image;
  if (!compile(mod, opt, image, prog, report, err)) return false;

  if (!binfmt::writeFile(image, outPath)) {
    err = "cannot write output: " + outPath;
    return false;
  }
  return true;
}

// --- Heuristic cycle estimate (NOT the official AEC cycle model) ----------
uint64_t estimateCycles(const binfmt::Image &image) {
  uint64_t cyc = 0;
  for (unsigned i = 0; i < image.code.size(); ++i) {
    isa::Op op = isa::decodeOp(image.code[i]);
    switch (op) {
      case isa::Op::LD: case isa::Op::ST: cyc += 8; break;
      case isa::Op::TMUL: case isa::Op::TMUL_S: cyc += 16; break;
      case isa::Op::TLDA: case isa::Op::TSTA: cyc += 6; break;
      case isa::Op::DIV: case isa::Op::SQRT: case isa::Op::RCP:
      case isa::Op::RSQ: case isa::Op::SIN: case isa::Op::COS:
      case isa::Op::EXP: case isa::Op::LOG: cyc += 12; break;
      case isa::Op::BR: case isa::Op::BRX: case isa::Op::JMP:
      case isa::Op::CALL: cyc += 2; break;
      default: cyc += 1; break;
    }
  }
  return cyc;
}

// --- Disassembler ---------------------------------------------------------
std::string disassemble(const binfmt::Image &image) {
  std::string out;
  char line[256];

  std::snprintf(line, sizeof(line),
      "; .aecbin  entry=%u  instructions=%u  param_bytes=%u  relocs=%u  symbols=%u\n",
      image.header.entryPC, (uint32_t)image.code.size(), image.header.paramBytes,
      (uint32_t)image.relocs.size(), (uint32_t)image.symbols.size());
  out += line;

  // pc -> label for annotation.
  std::map<uint32_t, std::string> labelAt;
  for (unsigned i = 0; i < image.symbols.size(); ++i)
    labelAt[image.symbols[i].value] = image.symbols[i].name;

  for (unsigned pc = 0; pc < image.code.size(); ++pc) {
    std::map<uint32_t, std::string>::iterator lit = labelAt.find(pc);
    if (lit != labelAt.end()) { out += lit->second; out += ":\n"; }

    const isa::Word128 &w = image.code[pc];
    isa::Op op = isa::decodeOp(w);
    uint16_t ctrl = (uint16_t)(w.word3 & 0xffff);
    uint8_t  ty   = (uint8_t)((ctrl >> 3) & 0xf);
    bool     pEn  = (ctrl & isa::kPredEnable) != 0;
    bool     pNeg = ((ctrl >> 14) & 1) != 0;
    uint8_t  pred = (uint8_t)(ctrl & 0x7);
    uint16_t dst  = (uint16_t)(w.word2 >> 16);
    uint16_t src1 = (uint16_t)(w.word2 & 0xffff);
    uint16_t src2 = (uint16_t)(w.word1 & 0xffff);
    uint32_t imm  = w.word0;
    const char *tn = isa::typeName((isa::Type)ty);

    std::string s;
    char b[192];
    std::snprintf(b, sizeof(b), "  %4u: ", pc); s += b;
    if (pEn) { std::snprintf(b, sizeof(b), "@%sP%u ", pNeg ? "!" : "", pred); s += b; }

    switch (op) {
      case isa::Op::LOADI:
        std::snprintf(b, sizeof(b), "LOADI.%s R%u, #0x%x", tn, dst, imm); s += b; break;
      case isa::Op::CPY: {
        const char *sp = specialName(src1);
        if (sp) std::snprintf(b, sizeof(b), "CPY.%s R%u, %s", tn, dst, sp);
        else    std::snprintf(b, sizeof(b), "CPY.%s R%u, R%u", tn, dst, src1);
        s += b; break;
      }
      case isa::Op::CMPP: case isa::Op::CMP:
        std::snprintf(b, sizeof(b), "%s.%s.%s P%u, R%u, R%u",
            isa::opName(op), cmpName((ctrl >> 8) & 0x7), tn, dst, src1, src2);
        s += b; break;
      case isa::Op::BR:
        std::snprintf(b, sizeof(b), "BR ->%u", imm); s += b; break;
      case isa::Op::BRX:
        std::snprintf(b, sizeof(b), "BRX %sP%u, ->%u", pNeg ? "!" : "", pred, imm);
        s += b; break;
      case isa::Op::LD: {
        uint32_t sp = (ctrl >> 11) & 0x3;
        if (sp == (uint32_t)isa::Space::PMEM || ((ctrl >> 11) & 0x7) == 4)
          std::snprintf(b, sizeof(b), "LD.pmem.%s R%u, [param+0x%x]", tn, dst, imm);
        else
          std::snprintf(b, sizeof(b), "LD.%s.%s R%u, [R%u]", spaceName(sp), tn, dst, src1);
        s += b; break;
      }
      case isa::Op::ST: {
        uint32_t sp = (ctrl >> 11) & 0x3;
        std::snprintf(b, sizeof(b), "ST.%s.%s [R%u], R%u", spaceName(sp), tn, src1, src2);
        s += b; break;
      }
      case isa::Op::MAD: case isa::Op::FMA:
      case isa::Op::TMUL: case isa::Op::TMUL_S:
        std::snprintf(b, sizeof(b), "%s.%s R%u, R%u, R%u, R%u",
            isa::opName(op), tn, dst, src1, src2, (uint16_t)(imm & 0xffff));
        s += b; break;
      case isa::Op::RET:
        std::snprintf(b, sizeof(b), "RET"); s += b; break;
      case isa::Op::HALT:
        std::snprintf(b, sizeof(b), "HALT"); s += b; break;
      default:
        std::snprintf(b, sizeof(b), "%s.%s R%u, R%u, R%u",
            isa::opName(op), tn, dst, src1, src2);
        s += b; break;
    }
    s += "\n";
    out += s;
  }

  if (!image.relocs.empty()) {
    out += "; relocations (instrIndex kind addend)\n";
    for (unsigned i = 0; i < image.relocs.size(); ++i) {
      std::snprintf(line, sizeof(line), ";   %u %u 0x%x\n",
          image.relocs[i].instrIndex, image.relocs[i].kind, image.relocs[i].addend);
      out += line;
    }
  }
  if (!image.symbols.empty()) {
    out += "; symbols (name value kind)\n";
    for (unsigned i = 0; i < image.symbols.size(); ++i) {
      std::snprintf(line, sizeof(line), ";   %s %u %u\n",
          image.symbols[i].name.c_str(), image.symbols[i].value, image.symbols[i].kind);
      out += line;
    }
  }
  return out;
}

} // namespace aec
