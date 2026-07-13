"""run_manifest.py - validate the compiler against the new C1 package.

For each testcase folder (kernel.ptx + manifest.json) under the new package
`public/C1编译器赛题/testcases`, this:
  1. compiles kernel.ptx -> .aecbin with compiler/aec-cc (-O2),
  2. builds the .pmem parameter block per the spec §7 ABI (declaration order,
     natural alignment, block padded to 8 bytes; pointer params hold a .gmem
     base address, value params hold the literal),
  3. lays out the gmem buffers and initializes them,
  4. runs the raw .aecbin on the AEC simulator,
  5. checks the output buffer against the manifest's reference
     (elementwise formula, or matmul C = A @ B).

Problem sizes are REDUCED from the manifest (params of kind "value" that look
like a total element count / matrix dim are overridden) so the pure-Python sim
finishes quickly; correctness is size-independent. Run: `py run_manifest.py`.
"""
import json
import os
import struct
import subprocess
import sys

import numpy as np

from aec_sim import simulate, ExecError

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.dirname(HERE)
AECCC = os.path.join(C1, "compiler", "aec-cc.exe")
if not os.path.exists(AECCC):
    AECCC = os.path.join(C1, "compiler", "aec-cc")
PKG = os.path.normpath(os.path.join(C1, "..", "public", "C1编译器赛题", "testcases"))
SZ = {"u32": 4, "s32": 4, "b32": 4, "f32": 4, "u64": 8, "b64": 8}


def pmem_block(params, ptr_base, n, gemm):
    """Pack the parameter block; return (bytes, value_map)."""
    out = bytearray()
    off = 0
    for p in params:
        sz = SZ[p["type"]]
        off = (off + sz - 1) // sz * sz
        while len(out) < off:
            out.append(0)
        if p["kind"] == "gmem_ptr":
            out += struct.pack("<Q", ptr_base[p["buffer"]])
        else:
            v = p.get("value", 0)
            if gemm is not None and p["name"][-1] in "MNK":
                v = gemm[p["name"][-1]]
            elif v > 100000:            # a total element count -> reduced n
                v = n
            out += (struct.pack("<Q", v) if p["type"] in ("u64", "b64")
                    else struct.pack("<I", v & 0xffffffff))
        off = len(out)
    while len(out) % 8:
        out.append(0)
    return bytes(out)


def run_case(folder):
    m = json.load(open(os.path.join(PKG, folder, "manifest.json"), encoding="utf-8"))
    ptx = os.path.join(PKG, folder, "kernel.ptx")
    out = os.path.join(HERE, "build", folder + ".aecbin")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = subprocess.run([AECCC, ptx, "-O2", "-o", out], capture_output=True, text=True)
    if r.returncode:
        return "COMPILE-ERR", (r.stdout + r.stderr).strip()[:80]

    chk = m["check"]
    rng = np.random.default_rng(7)
    gmem = bytearray()
    base = {}
    arr = {}
    gemm = None

    if chk["type"] == "matmul":                       # C = A @ B, reduced dims
        M, N, K = 32, 32, 128           # full K: exercises the real FP32 accum depth
        gemm = {"M": M, "N": N, "K": K}
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        shapes = {"A": A, "B": B, "C": np.zeros((M, N), np.float32)}
        for name, a in shapes.items():
            base[name] = len(gmem); arr[name] = a; gmem += a.tobytes()
        grid, block = tuple(m["gridDim"]), tuple(m["blockDim"])
        grid = ((N + block[0] - 1) // block[0], (M + block[1] - 1) // block[1], 1)
        ref = A @ B
    else:                                             # elementwise
        n = 512
        for name, b in m["buffers"].items():
            base[name] = len(gmem)
            a = (rng.standard_normal(n).astype(np.float32) if b["init"] == "rand_uniform"
                 else np.zeros(n, np.float32))
            arr[name] = a; gmem += a.tobytes()
        grid, block = (2, 1, 1), (256, 1, 1)
        expr = chk["formula"].split("=", 1)[1].strip().replace("[i]", "")
        ref = eval(expr, {"__builtins__": {}},
                   {k: v.astype(np.float64) for k, v in arr.items()}).astype(np.float32)

    pm = pmem_block(m["params"], base, 512, gemm)
    try:
        g, cyc, _ = simulate(out, grid, block, param_block=pm, gmem_init=bytes(gmem))
    except ExecError as e:
        return "EXEC-ERR", str(e)[:80]
    ob = chk["output"]
    got = np.frombuffer(bytes(g[base[ob]:base[ob] + arr[ob].size * 4]), np.float32)
    ref = np.asarray(ref, np.float32).reshape(-1)
    atol = float(chk.get("atol", 1e-5))              # use the manifest's tolerance
    rtol = float(chk.get("rtol", 1e-5))
    if got.shape == ref.shape and np.allclose(got, ref, rtol=rtol, atol=atol):
        return "PASS", "cyc=%d tol=%g/%g" % (cyc, atol, rtol)
    return "FAIL", "max_diff=%.4g (tol %g/%g)" % (
        float(np.max(np.abs(got - ref))), atol, rtol)


def main():
    if not os.path.isdir(PKG):
        print("new package not found at", PKG); return 2
    folders = sorted(d for d in os.listdir(PKG)
                     if os.path.exists(os.path.join(PKG, d, "manifest.json")))
    npass = 0
    for f in folders:
        kind, detail = run_case(f)
        if kind == "PASS":
            npass += 1
        print("  %-28s %-12s %s" % (f, kind, detail))
    print("\n%d/%d PASS" % (npass, len(folders)))
    return 0 if npass == len(folders) else 1


if __name__ == "__main__":
    sys.exit(main())
