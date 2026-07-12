// gemm_tmul.cpp - GEMM idiom detection + TMUL lowering.  Scoring category: T5.
//
// STATUS: detection scaffold + identity stub. Recognises the accumulate-in-a-
// loop shape (a `mad` whose dst == one of its sources inside a back-edge loop)
// that signals a GEMM inner product, but does NOT yet rewrite it into the
// tensor-core sequence. This is the single highest-weight correctness/perf
// category (T5 = 16-weight correctness, 11 perf points) so it is the most
// valuable TODO in the tree.
//
// PTX-05 (gemm_f16) is the canonical target:
//   K_LOOP: ... mad.f32 %f1,%f2,%f3,%f1 ...  ->  TLDA/TLDA/TMUL/TSTA tiles.
#include "aec/passes.h"

namespace aec {
namespace codegen {

namespace {

// A block is a GEMM accumulation latch if it has a back-edge and contains a
// multiply-accumulate whose destination is also a source (the running sum).
bool looksLikeGemmLatch(const ir::Function &fn, unsigned bi) {
  const ir::BasicBlock &b = fn.blocks[bi];
  bool backEdge = false;
  for (unsigned s = 0; s < b.succ.size(); ++s)
    if (b.succ[s] <= (int)bi) backEdge = true;
  if (!backEdge) return false;

  for (unsigned ii = 0; ii < b.insts.size(); ++ii) {
    const ir::Inst &in = b.insts[ii];
    if ((in.op == ir::Op::MAD || in.op == ir::Op::FMA) &&
        in.dst.kind != ir::Operand::None &&
        (in.dst.value == in.s3.value)) {
      return true;
    }
  }
  return false;
}

} // namespace

bool lowerGemmToTmul(ir::Function &fn, const Options &opt) {
  if (!opt.gemm_tmul) return false;
  bool changed = false;

  for (unsigned bi = 0; bi < fn.blocks.size(); ++bi) {
    if (!looksLikeGemmLatch(fn, bi)) continue;

    // TODO(T5): a real implementation would, for the detected tile:
    //   1. choose the tensor precision mode from the accumulate types
    //      (FP4/FP8/FP16/BF16/FP32/FP64/INT4/INT8/INT32 per scoring.md 8),
    //   2. replace the scalar K-loop with tile loads + a TMUL:
    //        TLDA.type Ra_tile,[Ra];  TLDA.type Rb_tile,[Rb];
    //        TMUL.type Racc,Ra_tile,Rb_tile,Racc;
    //      and a TSTA.type [Rc],Racc after the loop,
    //   3. auto-tune the tile size (16x16 base; handle non-16-multiple edges),
    //   4. set changed=true.
    // The encoder (src/isa/encoder.cpp) already emits bit-exact TMUL/TLDA/TSTA
    // and passes the golden TMUL.f16 / TSTA.f16 vectors, so only the rewrite is
    // missing.
    (void)bi;
  }

  return changed;
}

} // namespace codegen
} // namespace aec
