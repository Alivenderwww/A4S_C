// driver.h - Front-end entry + full compile pipeline + disassembler API.
//
// This is the one header that ties the phases together. tools/aec-cc.cpp and
// tools/aec-objdump.cpp only need this (plus target.h for Options). The phase
// functions themselves are declared in passes.h; here we expose the high-level
// glue the CLI drivers call.
#ifndef AEC_DRIVER_H
#define AEC_DRIVER_H

#include <string>

#include "aec/target.h"
#include "aec/ir.h"
#include "aec/binfmt.h"
#include "aec/ptx_ast.h"

namespace aec {

// --- Front end ------------------------------------------------------------
namespace ptx {
// Tokenize + parse PTX source text into a Module. On any structural problem
// returns false and puts a one-line reason in err.
bool parse(const std::string &src, Module &out, std::string &err);
}

// --- Compile report (spec §12 + scoring §B.3 diagnostics) -----------------
// Filled by compile(); serialized to JSON by aec-cc --report. The graded perf
// metric is warp-level dynamic instruction count, measured by the grader's own
// executor; this report carries the STATIC diagnostics the spec asks for.
struct CompileReport {
  std::string kernel;
  uint32_t numPtxInstructions   = 0;  // input PTX instructions
  uint32_t instructionCount     = 0;  // emitted AEC instructions (num_aec)
  uint32_t numBasicBlocks       = 0;
  uint32_t numVirtualRegisters  = 0;
  uint32_t numPhysicalRegisters = 0;
  uint32_t numPredicates        = 0;
  uint32_t spillLoads           = 0;  // emitted LMEM reload instructions
  uint32_t spillStores          = 0;  // emitted LMEM spill instructions
  uint32_t branchCount          = 0;
  uint32_t loadCount            = 0;
  uint32_t storeCount           = 0;
  uint32_t dependencyDepth      = 0;  // longest def->use chain (data-flow depth)
  uint32_t dualIssuePairs       = 0;
  uint32_t paramBytes           = 0;
  uint64_t estCycles            = 0;  // heuristic; no official cycle model shipped
};

// --- Whole-program pipeline ----------------------------------------------
// Runs frontend->IR->CFG->passes->regalloc->sched->lower->encode and fills the
// binfmt::Image. `prog` receives the (post-pipeline) IR for inspection and
// `report` the diagnostics. Returns false + err on failure.
bool compile(const ptx::Module &m, const Options &opt, binfmt::Image &image,
             ir::Program &prog, CompileReport &report, std::string &err);

// Convenience: read `inPath`, parse, compile, write `outPath`. `report` is
// filled on success so callers (aec-cc) can emit a perf report.
bool compileFile(const std::string &inPath, const std::string &outPath,
                 const Options &opt, CompileReport &report, std::string &err);

// Heuristic cycle estimate for an encoded image (used by the perf report and
// the auto-tuning agent; NOT the official AEC cycle model).
uint64_t estimateCycles(const binfmt::Image &image);

// --- Disassembler ---------------------------------------------------------
// Render an image back to human-readable AEC assembly (aec-objdump).
std::string disassemble(const binfmt::Image &image);

} // namespace aec

#endif // AEC_DRIVER_H
