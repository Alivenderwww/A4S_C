"""t5_sizes.py - T5 scalar GEMM correctness across K and matrix sizes.

Compiles T5 once at -O2 and runs the rotated + unrolled K-loop over many
(M, N, K): integer-divisible and non-divisible K (predicated remainder),
zero/one-trip K, and non-blockDim-multiple M/N (thread-bound predication). This
guards the loop-rotate + unroll pipeline against the robustness variants
("scalar GEMM 矩阵大小变化 / 边界变化"). Run: `py -3.13 t5_sizes.py`.
"""
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec_sim import Sim
from aec_decode import load_aecbin

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.dirname(HERE)
AECCC = os.path.join(C1, "compiler", "aec-cc.exe")
if not os.path.exists(AECCC):
    AECCC = os.path.join(C1, "compiler", "aec-cc")
PTX = os.path.normpath(os.path.join(
    C1, "..", "public", "C1编译器赛题", "testcases", "T5_scalar_gemm", "kernel.ptx"))
OUT = os.path.join(HERE, "build", "t5_sizes.aecbin")


def run(img, M, N, K):
    rng = np.random.default_rng(1)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    gmem = bytearray(); base = {}
    for nm, a in [("A", A), ("B", B), ("C", np.zeros((M, N), np.float32))]:
        base[nm] = len(gmem); gmem += a.tobytes()
    pm = bytearray()
    pm += struct.pack("<Q", base["A"]); pm += struct.pack("<Q", base["B"])
    pm += struct.pack("<Q", base["C"]); pm += struct.pack("<III", M, N, K)
    while len(pm) % 8:
        pm.append(0)
    g = np.frombuffer(bytes(gmem), np.uint8).copy()
    sim = Sim(img, g, bytes(pm), 0, False)
    bx, by = 16, 16
    sim.run(((N + bx - 1) // bx, (M + by - 1) // by, 1), (bx, by, 1))
    got = np.frombuffer(bytes(g[base["C"]:base["C"] + M * N * 4]), np.float32).reshape(M, N)
    ref = A @ B if K > 0 else np.zeros((M, N), np.float32)
    return np.allclose(got, ref, rtol=1e-3, atol=1e-3), float(np.max(np.abs(got - ref)))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r = subprocess.run([AECCC, PTX, "-O2", "-o", OUT], capture_output=True, text=True)
    if r.returncode:
        print("COMPILE-ERR:", (r.stdout + r.stderr).strip()[:200]); return 2
    img = load_aecbin(OUT)
    cases = [(16, 16, K) for K in [0, 1, 3, 4, 7, 8, 15, 16, 24, 31, 32]]
    cases += [(17, 13, 7), (30, 17, 15), (1, 1, 5), (13, 31, 33)]
    npass = 0
    for M, N, K in cases:
        ok, d = run(img, M, N, K)
        npass += ok
        print("  M=%2d N=%2d K=%3d  %-4s max_diff=%.2g" %
              (M, N, K, "PASS" if ok else "FAIL", d))
    print("\n%d/%d PASS" % (npass, len(cases)))
    return 0 if npass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
