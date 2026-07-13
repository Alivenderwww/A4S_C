// target.h - AEC target description + compiler driver options.
//
// Central place for machine limits (register/predicate counts) and the
// knobs the driver/agent tune. Keep this header free of heavy includes so
// every translation unit can pull it in cheaply.
#ifndef AEC_TARGET_H
#define AEC_TARGET_H

#include <cstdint>
#include <string>

namespace aec {

// Machine limits (from Track-B / aec_isa.h). These are hard ISA facts.
static const unsigned kRegisterCount   = 256; // R0..R255, 32-bit each.
static const unsigned kPredicateCount  = 8;   // P0..P7.
static const unsigned kPredicateNone   = 15;  // "no predicate" selector.
static const unsigned kWarpSize        = 32;
static const unsigned kInstructionBytes = 16; // 128-bit fixed length.

// Optimization level requested on the command line (-O0/-O2/-O3).
enum class OptLevel { O0 = 0, O2 = 2, O3 = 3 };

// Driver + agent tunables. The Agent (agent/run_agent.py) sweeps these and
// keeps the configuration that yields the best cycle count.
struct Options {
  OptLevel opt = OptLevel::O2;

  // Individual pass switches. -O0 clears them all; -O2/-O3 set the defaults
  // below. The agent may override any single flag to explore the space.
  bool const_prop   = true;
  bool dce          = true;
  bool cse          = true;
  bool licm         = true;
  bool mem_coalesce = true;
  bool pred_opt     = true;

  // Back-end knobs.
  bool dual_issue   = true;   // list scheduler pairs independent ops.
  bool gemm_tmul    = true;   // detect GEMM idiom and lower to TMUL.
  int  sched_window = 16;     // list-scheduler lookahead (instructions).
  bool unroll       = false;  // loop unrolling (opt-in, -O3): expose ILP.
  int  unroll_factor = 4;     // unroll count for counted loops.

  bool verbose      = false;  // dump pipeline progress to stderr.
  bool lenient      = false;  // keep going past unhandled PTX ops (default: fail).

  // Derive the boolean pass switches from an -O level.
  //
  // CORRECTNESS FIRST AT EVERY LEVEL. Two transforms are AEC semantic
  // requirements, not optimizations, so they stay ON even at -O0:
  //   gemm_tmul : GEMM idiom lowering -- must run to emit a matmul at all.
  //   pred_opt  : if-convert divergent bounds guards. AEC has no SIMT
  //               reconvergence, so a divergent BRX is a WRONG RESULT (not just
  //               slow); a partial last block (N % blockDim != 0) would fault.
  //               Gating this off at -O0 made -O0 miscompile -> instant zero.
  // Everything else below is a pure performance choice:
  //   -O0 = correct standard codegen (perf opts off)
  //   -O2 = default: all safe perf opts on (earns the bulk of the perf score)
  //   -O3 = aggressive: adds opts that don't fit every PTX (unroll, wider
  //         window) but MUST remain correct.
  void applyOptLevel(OptLevel level) {
    opt = level;
    gemm_tmul  = true;                 // correctness: always on.
    pred_opt   = true;                 // correctness: always on (see above).
    const bool o2 = (level != OptLevel::O0);   // O2/O3 performance opts.
    const_prop = dce = cse = o2;
    licm = mem_coalesce = o2;
    dual_issue = o2;
    // Unrolling is now a DEFAULT (-O2) optimization: it is shape-conservative
    // (only constant-trip single-block self-loops) and guarded against register
    // pressure (see unroll.cpp), so it is correctness-safe on every kernel while
    // giving large latency-hiding wins (e.g. reuse 2308->820). -O3 pushes a
    // more aggressive unroll factor + wider scheduling window.
    unroll = o2;
    unroll_factor = (level == OptLevel::O3) ? 8 : 4;
    sched_window  = (level == OptLevel::O3) ? 32 : 16;
  }
};

} // namespace aec

#endif // AEC_TARGET_H
