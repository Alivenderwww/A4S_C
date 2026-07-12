"""dogfood.py — C1 correctness harness.

For each public kernel: compile the PTX with the C1 compiler, run the compiled
`.aecbin` on the AEC functional simulator, and compare the resulting GMEM
against an independent numpy reference. This is the local correctness oracle
that stands in for the (unreleased) official golden model.

Usage (from C1/sim, with the compiler already built via `make` in C1/):
    py -3.13 dogfood.py                 # all cases
    py -3.13 dogfood.py vadd poly       # selected cases
    py -3.13 dogfood.py --opt O0        # choose -O level
    py -3.13 dogfood.py --strict        # flag ISA-illegal ops as errors

Exit code is non-zero if any selected case fails, so it works in CI.
"""
import os
import subprocess
import sys
import numpy as np

import cases as C
from aec_sim import simulate, ExecError

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.dirname(HERE)
AECCC = os.path.join(C1, "bin", "aec-cc.exe")
if not os.path.exists(AECCC):
    AECCC = os.path.join(C1, "bin", "aec-cc")
PTXDIR = os.path.normpath(os.path.join(
    C1, "..", "public", "Track-C", "C1-compiler", "testcases"))
BUILD = os.path.join(HERE, "build")


def compile_ptx(ptx_name, opt):
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(PTXDIR, ptx_name)
    out = os.path.join(BUILD, ptx_name.replace(".ptx", ".aecbin"))
    r = subprocess.run([AECCC, src, "-" + opt, "-o", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr).strip()
    return out, ""


def run_case(name, opt, strict, tol):
    case = C.ALL[name]()
    aecbin, err = compile_ptx(case["ptx"], opt)
    if aecbin is None:
        return False, "compile failed: " + err, None
    off, count, dt = case["out"]
    try:
        gmem, cycles, warps = simulate(
            aecbin, case["grid"], case["block"], param_block=case["param"],
            gmem_init=case["gmem"], strict=strict)
    except ExecError as e:
        return False, "sim error: %s" % e, None
    got = np.frombuffer(bytes(gmem[off:off + count * dt.itemsize]), dt).astype(np.float32)
    ref = np.asarray(case["ref"], np.float32).reshape(-1)
    if got.shape != ref.shape:
        return False, "shape %s != ref %s" % (got.shape, ref.shape), cycles
    mad = float(np.max(np.abs(got - ref))) if got.size else 0.0
    ok = np.allclose(got, ref, rtol=tol, atol=tol)
    detail = "max_abs_diff=%.3g cycles=%d warps=%d" % (mad, cycles, warps)
    if not ok:
        bad = int(np.argmax(np.abs(got - ref)))
        detail += "  first@%d got=%.6g ref=%.6g" % (bad, got[bad], ref[bad])
    return ok, detail, cycles


def main(argv):
    opt = "O2"; strict = False; tol = 1e-4
    names = []
    it = iter(argv)
    for a in it:
        if a == "--opt": opt = next(it)
        elif a.startswith("--opt="): opt = a.split("=", 1)[1]
        elif a == "--strict": strict = True
        elif a == "--tol": tol = float(next(it))
        elif a in C.ALL: names.append(a)
        else:
            print("unknown arg or case: %s (cases: %s)" % (a, ", ".join(C.ALL)))
            return 2
    names = names or list(C.ALL)
    if not os.path.exists(AECCC):
        print("aec-cc not found at %s — build it first (cd C1 && make)" % AECCC)
        return 2
    npass = 0
    print("compiler: %s   opt=-%s" % (AECCC, opt))
    for n in names:
        ok, detail, _ = run_case(n, opt, strict, tol)
        note = C.ALL[n]().get("note", "")
        tag = "PASS" if ok else "FAIL"
        print("  [%s] %-6s %s%s" % (tag, n, detail, ("   (%s)" % note if note and not ok else "")))
        npass += ok
    print("=== %d/%d passed ===" % (npass, len(names)))
    return 0 if npass == len(names) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
