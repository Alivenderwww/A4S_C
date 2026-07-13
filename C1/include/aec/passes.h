// passes.h - Declarations for every optimization/back-end pass.
//
// Each pass is a plain function over an ir::Function. It returns true when it
// changed the IR (so the driver can iterate to a fixpoint later). In this
// scaffold every optimization is a structurally-complete no-op with a TODO
// mapping it to its scoring category; the C1 owner fills in the real logic.
#ifndef AEC_PASSES_H
#define AEC_PASSES_H

#include "aec/ir.h"
#include "aec/target.h"

namespace aec {

// PTX AST -> IR (instruction selection + CFG-ready block splitting).
namespace ptx { struct Module; }
ir::Program buildIR(const ptx::Module &m, const Options &opt);

// CFG construction: fill BasicBlock::succ/pred from branch targets.
void buildCFG(ir::Function &fn);

namespace passes {

// --- Scalar / control optimizations (scoring category T2). ----------------
bool constProp(ir::Function &fn, const Options &opt);   // const_prop.cpp
bool dce(ir::Function &fn, const Options &opt);          // dce.cpp
bool cse(ir::Function &fn, const Options &opt);          // cse.cpp
bool licm(ir::Function &fn, const Options &opt);         // licm.cpp

// --- GPGPU memory optimizations (scoring category T3). --------------------
bool memCoalesce(ir::Function &fn, const Options &opt);  // mem_coalesce.cpp

// --- Predicate optimization (scoring category T2). ------------------------
bool predOpt(ir::Function &fn, const Options &opt);      // pred_opt.cpp

// --- Loop unrolling: expose independent loads for latency hiding (T4). -----
bool unrollLoops(ir::Function &fn, const Options &opt);  // unroll.cpp

} // namespace passes

// --- Register allocation (scoring category T4). ---------------------------
namespace regalloc {
void linearScan(ir::Function &fn, const Options &opt);   // linear_scan.cpp
}

// --- Instruction scheduling (scoring category T4). ------------------------
namespace sched {
void listSchedule(ir::Function &fn, const Options &opt); // list_sched.cpp
}

// --- Code generation / lowering (scoring categories T1 + T5). -------------
namespace codegen {
// Final legalization + flatten to a linear stream, resolving branch labels
// to absolute instruction indices. Produces the encodable instruction list.
std::vector<ir::Inst> lower(ir::Function &fn, const Options &opt); // lower.cpp
}

} // namespace aec

#endif // AEC_PASSES_H
