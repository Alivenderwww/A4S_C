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
  void applyOptLevel(OptLevel level) {
    opt = level;
    const bool on = (level != OptLevel::O0);
    const_prop = dce = cse = on;
    licm = mem_coalesce = pred_opt = on;
    dual_issue = on;
    gemm_tmul  = true;                 // lowering correctness, always on.
    sched_window = (level == OptLevel::O3) ? 32 : 16;
    unroll = (level == OptLevel::O3);  // unrolling is opt-in at -O3 only.
  }
};

} // namespace aec

#endif // AEC_TARGET_H
