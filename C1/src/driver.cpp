// driver.cpp - Pipeline orchestration, IR->encoding, disassembly, cycle model.
//
// Ties every phase together (frontend -> IR -> CFG -> passes -> regalloc ->
// sched -> gemm/lower -> encode -> image) and provides the disassembler and a
// heuristic cycle estimate. -O0 skips the optimization passes; -O2/-O3 run
// them in order. Register allocation, scheduling and final lowering always run
// (they are required to produce a legal image, not optional optimizations).
#include "aec/driver.h"
#include "aec/passes.h"
#include "aec/isa.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <map>
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

void runOptPasses(ir::Function &fn, const Options &opt) {
  if (opt.opt == OptLevel::O0) return;
  // Two light rounds so an implemented pass can feed the next (identity now).
  for (int round = 0; round < 2; ++round) {
    if (opt.const_prop) passes::constProp(fn, opt);
    if (opt.cse)        passes::cse(fn, opt);
    if (opt.licm)       passes::licm(fn, opt);
    if (opt.dce)        passes::dce(fn, opt);
  }
  if (opt.mem_coalesce) passes::memCoalesce(fn, opt);
  if (opt.pred_opt)     passes::predOpt(fn, opt);
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
  if (opt.gemm_tmul) codegen::lowerGemmToTmul(fn, opt);
  if (opt.unroll) passes::unrollLoops(fn, opt);   // expose independent loads (-O3).
  sched::listSchedule(fn, opt);   // pre-RA: schedule on vregs (fewer false deps).
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
  report.spillCount = fn.regs.spillCount;
  report.dualIssuePairs = fn.dualIssuePairs;
  report.paramBytes = pbytes;
  report.estCycles = estimateCycles(image);

  if (opt.verbose) {
    std::fprintf(stderr,
        "[driver] kernel=%s insts=%u spills=%u dual_pairs=%u est_cycles=%llu\n",
        fn.name.c_str(), report.instructionCount, report.spillCount,
        report.dualIssuePairs, (unsigned long long)report.estCycles);
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
    uint8_t  pred = (uint8_t)(ctrl & 0x7);
    uint16_t dst  = (uint16_t)(w.word2 >> 16);
    uint16_t src1 = (uint16_t)(w.word2 & 0xffff);
    uint16_t src2 = (uint16_t)(w.word1 & 0xffff);
    uint32_t imm  = w.word0;
    const char *tn = isa::typeName((isa::Type)ty);

    std::string s;
    char b[192];
    std::snprintf(b, sizeof(b), "  %4u: ", pc); s += b;
    if (pEn) { std::snprintf(b, sizeof(b), "@P%u ", pred); s += b; }

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
        std::snprintf(b, sizeof(b), "BRX P%u, ->%u", pred, imm); s += b; break;
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
