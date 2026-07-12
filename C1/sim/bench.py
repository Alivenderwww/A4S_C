"""bench.py — broad C1 robustness harness covering all competition mutations.

Mutation coverage (scoring.md robustness list):
  1. param / matrix size .......... size sweep (per-kernel launch configs)
  2. register renaming ............ mutate.rename_registers        (preserving)
  3. basic-block reorder .......... mutate.reorder_blocks          (preserving)
  4. loop-count change ............ mutate.set_loop_count + reref  (poly/reuse)
  5. dead-code insertion .......... mutate.insert_dead_code        (preserving)
  6. register-pressure increase ... mutate.increase_register_pressure (-O0)
  7. dtype change (GEMM) .......... mutate.gemm_to_{bf16,f32} + reref
  8. memory-reuse pattern ......... reuse block sweep (col-vs-idx reuse degree)

Every variant is compiled and checked on the AEC simulator against an
independent numpy reference. Results: PASS / DIVERGE (non-uniform BRX, the §1.3
gap until if-conversion) / FAIL / SIM-ERR / COMPILE-ERR.

Usage (from C1/sim, compiler built via `make` in C1/):
    py -3.13 bench.py                 # everything
    py -3.13 bench.py --mutants 8     # more preserving-mutation variants
    py -3.13 bench.py gemm            # one kernel
"""
import argparse
import os
import subprocess
import numpy as np

import cases as C
import mutate
from aec_sim import simulate, ExecError

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.dirname(HERE)
AECCC = os.path.join(C1, "bin", "aec-cc.exe")
if not os.path.exists(AECCC):
    AECCC = os.path.join(C1, "bin", "aec-cc")
PTXDIR = os.path.normpath(os.path.join(
    C1, "..", "public", "Track-C", "C1-compiler", "testcases"))
BUILD = os.path.join(HERE, "build", "bench")
TOL = 1e-3

SWEEP = {
    "vadd": [dict(n=256, block=64), dict(n=1024, block=256), dict(n=4096, block=256),
             dict(n=300, block=64), dict(n=1000, block=256), dict(n=257, block=256)],
    "poly": [dict(n=256, block=256), dict(n=128, block=128), dict(n=200, block=256),
             dict(n=100, block=128), dict(n=250, block=256)],
    # reuse block sweep also covers mutation #8 (col-vs-idx memory-reuse degree).
    "reuse": [dict(grid=4, block=32), dict(grid=2, block=256), dict(grid=8, block=64),
              dict(grid=16, block=256)],
    "reg":  [dict(n=128, block=64), dict(n=512, block=128), dict(n=200, block=64),
             dict(n=999, block=256)],
    "gemm": [dict(M=16, N=16, K=16), dict(M=32, N=32, K=32), dict(M=64, N=48, K=32),
             dict(M=17, N=16, K=16), dict(M=16, N=16, K=24), dict(M=48, N=33, K=16)],
}

tally = {"PASS": 0, "FAIL": 0, "DIVERGE": 0, "SIM-ERR": 0, "COMPILE-ERR": 0}
fails = []


def compile_src(src_path, opt, tag):
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, tag + ".aecbin")
    r = subprocess.run([AECCC, src_path, "-" + opt, "-o", out],
                       capture_output=True, text=True)
    return (None, (r.stdout + r.stderr).strip()) if r.returncode else (out, "")


def check(aecbin, case):
    off, count, dt = case["out"]
    try:
        gmem, cyc, _ = simulate(aecbin, case["grid"], case["block"],
                                param_block=case["param"], gmem_init=case["gmem"])
    except ExecError as e:
        if "non-uniform BRX" in str(e):
            return "DIVERGE", "divergent bounds guard (needs pred_opt)"
        return "SIM-ERR", str(e)
    got = np.frombuffer(bytes(gmem[off:off + count * dt.itemsize]), dt).astype(np.float32)
    ref = np.asarray(case["ref"], np.float32).reshape(-1)
    if got.shape != ref.shape:
        return "FAIL", "shape %s != %s" % (got.shape, ref.shape)
    if np.allclose(got, ref, rtol=TOL, atol=TOL):
        return "PASS", "cyc=%d" % cyc
    return "FAIL", "max_abs_diff=%.3g" % float(np.max(np.abs(got - ref)))


def record(kind, detail, label):
    tally[kind] += 1
    if kind in ("FAIL", "SIM-ERR", "COMPILE-ERR"):
        fails.append((label, "%s %s" % (kind, detail)))
    return kind


def run_variant(src_text, case, opt, tag, label):
    path = os.path.join(BUILD, tag + ".ptx")
    with open(path, "w") as f:
        f.write(src_text)
    aecbin, err = compile_src(path, opt, tag)
    if aecbin is None:
        return record("COMPILE-ERR", err[:90], label)
    kind, detail = check(aecbin, case)
    return record(kind, detail, label)


def read_src(ptx_name):
    with open(os.path.join(PTXDIR, ptx_name)) as f:
        return f.read()


def divides(name, cfg):
    if name == "reuse":
        return True
    if name == "gemm":
        return cfg["M"] % 16 == 0 and cfg["N"] % 16 == 0
    return cfg["n"] % cfg["block"] == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default="O2")
    ap.add_argument("--mutants", type=int, default=4)
    ap.add_argument("kernels", nargs="*", default=list(C.ALL))
    args = ap.parse_args()
    if not os.path.exists(AECCC):
        print("aec-cc not found; build it first (cd C1 && make)"); return 2

    for name in args.kernels:
        builder = C.ALL[name]
        base_name = builder()["ptx"]
        base_src = read_src(base_name)
        print("\n== %s ==" % name)

        # 1 & 8) size / memory-reuse sweep: same base PTX, many launch configs.
        base_bin, err = compile_src(os.path.join(PTXDIR, base_name), args.opt, name + "_base")
        if base_bin is None:
            print("  COMPILE-ERR base: %s" % err); record("COMPILE-ERR", err, name); continue
        for cfg in SWEEP[name]:
            k, d = check(base_bin, builder(**cfg))
            record(k, d, "%s size %s" % (name, cfg))
            print("  size%s %-26s %-8s %s" % (" " if divides(name, cfg) else "*", cfg, k, d))

        # 2,3,5,6) semantic-preserving source mutations (checked vs default ref).
        default_case = builder()
        for mname, mut in mutate.PRESERVING.items():
            opt = "O0" if mname == "pressure" else args.opt   # pressure needs -O0.
            res = [run_variant(mut(base_src, seed=s), default_case, opt,
                               "%s_%s_%d" % (name, mname, s), "%s %s#%d" % (name, mname, s))
                   for s in range(args.mutants)]
            print("  mutate %-9s x%d @%s -> %s" % (
                mname, args.mutants, opt,
                ", ".join("%d PASS" % res.count("PASS") if res.count("PASS") == len(res)
                          else "%s" % {r: res.count(r) for r in set(res)})))

        # 4) loop-count change (poly / reuse only; rebuild reference).
        if name in ("poly", "reuse"):
            counts = [8, 24, 48] if name == "poly" else [4, 8, 12]
            for L in counts:
                case = builder(loop_count=L)
                k = run_variant(mutate.set_loop_count(base_src, L), case, args.opt,
                                "%s_loop%d" % (name, L), "%s loop=%d" % (name, L))
                print("  loop-count=%-3d -> %s" % (L, k))

        # 7) dtype change (gemm only; rebuild reference in that dtype).
        if name == "gemm":
            for dt, mut in (("bf16", mutate.gemm_to_bf16), ("f32", mutate.gemm_to_f32)):
                case = builder(M=32, N=32, K=32, dtype=dt)
                k = run_variant(mut(base_src), case, args.opt,
                                "gemm_%s" % dt, "gemm dtype=%s" % dt)
                print("  dtype=%-4s -> %s" % (dt, k))

    total = sum(tally.values())
    print("\n==== summary (%d variants) ====" % total)
    for k in ("PASS", "DIVERGE", "FAIL", "SIM-ERR", "COMPILE-ERR"):
        print("  %-12s %d" % (k, tally[k]))
    for label, d in fails[:50]:
        print("    %s  %s" % (label, d))
    print("\nDIVERGE = N not a multiple of blockDim -> divergent bounds guard;"
          " expected until pred_opt (if-conversion).")
    print("Note: fp8/fp4/int GEMM dtypes need T5 multi-precision (TMUL) support — not yet.")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
