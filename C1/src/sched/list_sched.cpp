// list_sched.cpp - Dependency-aware list scheduling + dual-issue pairing.
// Scoring category: T4.
//
// Runs PRE-register-allocation (on virtual registers), where only real data
// dependencies constrain the order (physical-register reuse would add spurious
// WAR/WAW edges). Per basic block it builds the data-dependence graph
// (RAW/WAR/WAW over vregs + predicates, plus a conservative total order over
// memory ops), computes each node's latency-weighted critical-path height, and
// emits a new order from a ready list picking the highest-height node first.
// High-latency loads (32-cycle GMEM) sit on long critical paths, so they are
// issued early and independent work fills the shadow -> memory latency hiding.
// A modern GPU encodes exactly this statically (SASS stall counts / scoreboards);
// see C1_工业界参考与知识库.md §4.
//
// T4 (mixed_load_compute) interleaves independent loads with FP math
// specifically to reward this.
#include "aec/passes.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <vector>

namespace aec {
namespace sched {

namespace {

// AEC instruction latencies (cycles until the result is available). This is a
// scheduling heuristic, not a graded model: GMEM/LMEM use the external memory
// service (~32c assumed); on-chip spaces are 1c.
int latencyOf(const ir::Inst &in) {
  switch (in.op) {
    case ir::Op::LD:
      return (in.modifier == (uint32_t)isa::Space::GMEM ||
              in.modifier == (uint32_t)isa::Space::LMEM) ? 32 : 2;
    case ir::Op::ST: case ir::Op::ATOM: return 1;
    case ir::Op::TMUL: case ir::Op::TMUL_S: return 16;
    case ir::Op::TLDA: case ir::Op::TSTA: return 6;
    case ir::Op::DIV: return 12;
    case ir::Op::RCP: case ir::Op::RSQ: case ir::Op::SQRT:
    case ir::Op::SIN: case ir::Op::COS: case ir::Op::EXP: case ir::Op::LOG:
      return 12;
    default: return 1;
  }
}

bool isMemOrBarrier(const ir::Inst &in) {
  switch (in.op) {
    case ir::Op::LD: case ir::Op::ST: case ir::Op::LDC: case ir::Op::ATOM:
    case ir::Op::TLDA: case ir::Op::TSTA: case ir::Op::TMOV:
    case ir::Op::SYNC_CT: case ir::Op::SYNC_WG: case ir::Op::SSYNC:
    case ir::Op::MBAR:
      return true;
    default: return false;
  }
}

const uint32_t PRED_NS = 0x10000u;   // predicate key namespace (vs registers).

void defUse(const ir::Inst &in, std::vector<uint32_t> &uses,
            std::vector<uint32_t> &defs) {
  const ir::Operand *s[3] = {&in.s1, &in.s2, &in.s3};
  for (int i = 0; i < 3; ++i)
    if (s[i]->kind == ir::Operand::Reg) uses.push_back(s[i]->value);
  if (in.guard >= 0) uses.push_back(PRED_NS | (uint32_t)(in.guard & 0x7));
  if (in.dst.kind == ir::Operand::Reg) defs.push_back(in.dst.value);
  else if (in.dst.kind == ir::Operand::Pred) defs.push_back(PRED_NS | (in.dst.value & 0x7));
}

} // namespace

void listSchedule(ir::Function &fn, const Options &opt) {
  // Gated by dual_issue (off at -O0 / --no-dual-issue). AEC_NO_SCHED also keeps
  // program order (for A/B measuring the scheduler's effect).
  if (!opt.dual_issue || std::getenv("AEC_NO_SCHED")) { fn.dualIssuePairs = 0; return; }
  uint32_t pairs = 0;

  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    ir::BasicBlock &b = fn.blocks[bi];
    const int total = (int)b.insts.size();
    if (total < 2) continue;

    const bool hasTerm = b.insts.back().isTerminator();
    const int N = hasTerm ? total - 1 : total;   // schedulable node count.
    if (N < 2) continue;

    // --- build DDG (edges always go low-index -> high-index) ---------------
    std::vector<std::vector<int> > succ(N);
    std::vector<int> indeg(N, 0);
    std::map<uint32_t, int> lastWriter;
    std::map<uint32_t, std::vector<int> > readers;
    int lastMem = -1;

    // dedup helper for edges.
    std::vector<std::vector<char> > hasEdge(N, std::vector<char>(N, 0));
    // (N is small for these kernels; a dense matrix keeps it simple.)
    // NOTE: falls back gracefully if N is large — memory is N^2 chars.

    for (int i = 0; i < N; ++i) {
      std::vector<uint32_t> uses, defs;
      defUse(b.insts[i], uses, defs);
      if (b.insts[i].dst.kind == ir::Operand::Reg) {
        std::map<uint32_t, uint32_t>::const_iterator hi =
            fn.regs.loadPairHi.find(b.insts[i].dst.value);
        if (hi != fn.regs.loadPairHi.end()) defs.push_back(hi->second);
      }

      for (size_t u = 0; u < uses.size(); ++u) {                 // RAW
        std::map<uint32_t, int>::iterator w = lastWriter.find(uses[u]);
        if (w != lastWriter.end() && w->second != i && !hasEdge[w->second][i]) {
          hasEdge[w->second][i] = 1; succ[w->second].push_back(i); ++indeg[i];
        }
      }
      for (size_t d = 0; d < defs.size(); ++d) {
        std::map<uint32_t, int>::iterator w = lastWriter.find(defs[d]);
        if (w != lastWriter.end() && w->second != i && !hasEdge[w->second][i]) { // WAW
          hasEdge[w->second][i] = 1; succ[w->second].push_back(i); ++indeg[i];
        }
        std::map<uint32_t, std::vector<int> >::iterator r = readers.find(defs[d]);
        if (r != readers.end())
          for (size_t k = 0; k < r->second.size(); ++k) {                       // WAR
            int rk = r->second[k];
            if (rk != i && !hasEdge[rk][i]) { hasEdge[rk][i] = 1; succ[rk].push_back(i); ++indeg[i]; }
          }
      }
      if (isMemOrBarrier(b.insts[i])) {                          // conservative mem order
        if (lastMem >= 0 && !hasEdge[lastMem][i]) { hasEdge[lastMem][i] = 1; succ[lastMem].push_back(i); ++indeg[i]; }
        lastMem = i;
      }
      for (size_t u = 0; u < uses.size(); ++u) readers[uses[u]].push_back(i);
      for (size_t d = 0; d < defs.size(); ++d) { lastWriter[defs[d]] = i; readers[defs[d]].clear(); }
    }

    // --- latency-weighted critical-path height (edges go forward) ----------
    std::vector<int> height(N, 0);
    for (int i = N - 1; i >= 0; --i) {
      int h = 0;
      for (size_t k = 0; k < succ[i].size(); ++k) h = std::max(h, height[succ[i][k]]);
      height[i] = latencyOf(b.insts[i]) + h;
    }

    // --- list schedule: highest height first (tie: original order) ---------
    std::vector<int> order;
    order.reserve(N);
    std::vector<char> done(N, 0);
    std::vector<int> deg = indeg;
    for (int placed = 0; placed < N; ++placed) {
      int best = -1;
      for (int i = 0; i < N; ++i)
        if (!done[i] && deg[i] == 0)
          if (best < 0 || height[i] > height[best] ||
              (height[i] == height[best] && i < best))
            best = i;
      if (best < 0) { for (int i = 0; i < N; ++i) if (!done[i]) { best = i; break; } } // safety
      done[best] = 1; order.push_back(best);
      for (size_t k = 0; k < succ[best].size(); ++k) --deg[succ[best][k]];
    }

    // --- rebuild block + count adjacent dual-issue pairs -------------------
    std::vector<ir::Inst> out;
    out.reserve(total);
    for (int i = 0; i < N; ++i) out.push_back(b.insts[order[i]]);
    if (hasTerm) out.push_back(b.insts.back());
    b.insts.swap(out);

    if (opt.dual_issue) {
      for (int i = 0; i + 1 < N; ) {
        if (!hasEdge[order[i] < order[i + 1] ? order[i] : order[i + 1]]
                    [order[i] < order[i + 1] ? order[i + 1] : order[i]] &&
            !b.insts[i].isTerminator() && !b.insts[i + 1].isTerminator() &&
            b.insts[i].op != ir::Op::LD && b.insts[i + 1].op != ir::Op::LD) {
          ++pairs; i += 2;
        } else { ++i; }
      }
    }
  }

  fn.dualIssuePairs = pairs;
}

} // namespace sched
} // namespace aec
