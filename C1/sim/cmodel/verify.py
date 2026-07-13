"""verify.py - end-to-end validation of the C1 compiler against the OFFICIAL
AEC golden model (public/aec-cmodel-release/bin/aec-precise-*).

For each of the 5 public cases this:
  1. builds the input files (pmem.bin + input_<buffer>.bin) with the official
     fixed GMEM/PMEM layout and seeds (PUBLIC_AEC_PRECISE_COMMANDS.md),
  2. compiles kernel.ptx -> .aecbin with our aec-cc (-O0 and -O2),
  3. runs aec-precise, reads the stdout JSON (status + `steps` = the graded
     warp-level dynamic instruction count), and --dumps the output buffer,
  4. spot-checks the output against a numpy-free reference.

Run inside WSL (the linux-x86_64 CModel):
    python3 C1/sim/cmodel/verify.py

Paths are resolved relative to this file, and aec-cc.exe is invoked through
WSL interop (Windows paths via `wslpath -w`), so it works from any clone.
"""
import json
import os
import random
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AEC = os.path.join(REPO, "public/aec-cmodel-release/bin/aec-precise-linux-x86_64")
PKG = os.path.join(REPO, "public/C1编译器赛题/testcases")
AECCC = os.path.join(REPO, "C1/compiler/aec-cc.exe")   # Windows exe via interop
if not os.path.exists(AECCC):
    AECCC = os.path.join(REPO, "C1/compiler/aec-cc")   # native (Linux/ARM grader)
BUILD = os.path.join(REPO, "C1/sim/build/cmodel")   # gitignored scratch
INROOT = "/tmp/c1_inputs"


def wpath(p):
    return subprocess.check_output(["wslpath", "-w", p]).decode().strip()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def rand_f32(numel, seed):
    rng = random.Random(seed)
    return [f32(rng.uniform(-1.0, 1.0)) for _ in range(numel)]


def rd1(path, i):
    with open(path, "rb") as fh:
        fh.seek(i * 4)
        return struct.unpack("<f", fh.read(4))[0]


PACK = {"u64": ("<Q", 8), "u32": ("<I", 4)}

# Each case: fixed layout from PUBLIC_AEC_PRECISE_COMMANDS.md.
#   buffers: (name, gmem_addr, numel, seed|None)   None seed => zero (output)
#   pmem:    (offset, type, value)
CASES = [
    dict(name="T1_basic_lowering", grid=(4096, 1, 1), block=(256, 1, 1), gmem=12591104,
         buffers=[("a", 256, 1048576, 1), ("b", 4194560, 1048576, 2), ("c", 8388864, 1048576, None)],
         pmem=[(0, "u64", 256), (8, "u64", 4194560), (16, "u64", 8388864), (24, "u32", 1048576)],
         out="c", check=lambda v: v["a"] + v["b"]),
    dict(name="T2_scalar_optimization", grid=(2048, 1, 1), block=(256, 1, 1), gmem=6299648,
         buffers=[("x", 256, 524288, 3), ("y", 2097408, 524288, 4), ("out", 4194560, 524288, None)],
         pmem=[(0, "u64", 256), (8, "u64", 2097408), (16, "u64", 4194560), (24, "u32", 524288)],
         out="out", check=lambda v: (v["x"] + v["y"]) * (v["x"] + v["y"]) + v["x"]),
    dict(name="T3_memory_reuse", grid=(2048, 1, 1), block=(256, 1, 1), gmem=8396800,
         buffers=[("x", 256, 524288, 5), ("y", 2097408, 524288, 6), ("z", 4194560, 524288, 7),
                  ("out", 6291712, 524288, None)],
         pmem=[(0, "u64", 256), (8, "u64", 2097408), (16, "u64", 4194560), (24, "u64", 6291712), (32, "u32", 524288)],
         out="out", check=lambda v: v["x"] * v["y"] + v["x"] * v["z"]),
    dict(name="T4_register_scheduling", grid=(2048, 1, 1), block=(256, 1, 1), gmem=10493952,
         buffers=[("a", 256, 524288, 8), ("b", 2097408, 524288, 9), ("c", 4194560, 524288, 10),
                  ("d", 6291712, 524288, 11), ("out", 8388864, 524288, None)],
         pmem=[(0, "u64", 256), (8, "u64", 2097408), (16, "u64", 4194560), (24, "u64", 6291712),
               (32, "u64", 8388864), (40, "u32", 524288)],
         out="out",
         check=lambda v: (v["a"] + v["b"]) * (v["c"] - v["d"]) + (v["a"] * v["c"]) * (v["b"] + v["d"])),
    dict(name="T5_scalar_gemm", grid=(8, 8, 1), block=(16, 16, 1), gmem=204800,
         buffers=[("A", 256, 16384, 12), ("B", 65792, 16384, 13), ("C", 131328, 16384, None)],
         pmem=[(0, "u64", 256), (8, "u64", 65792), (16, "u64", 131328),
               (24, "u32", 128), (28, "u32", 128), (32, "u32", 128)],
         out="C", check="matmul", MNK=(128, 128, 128)),
]


def gen_inputs(c):
    d = os.path.join(INROOT, c["name"])
    os.makedirs(d, exist_ok=True)
    for name, _, numel, seed in c["buffers"]:
        vals = rand_f32(numel, seed) if seed is not None else [0.0] * numel
        open(os.path.join(d, "input_%s.bin" % name), "wb").write(
            struct.pack("<%df" % numel, *vals))
    size = max(off + PACK[t][1] for off, t, _ in c["pmem"])
    size = (size + 7) & ~7
    pm = bytearray(size)
    for off, t, val in c["pmem"]:
        fmt, n = PACK[t]
        pm[off:off + n] = struct.pack(fmt, val)
    open(os.path.join(d, "pmem.bin"), "wb").write(pm)


def compile_case(c, opt, suffix):
    os.makedirs(BUILD, exist_ok=True)
    ptx = os.path.join(PKG, c["name"], "kernel.ptx")
    out = os.path.join(BUILD, c["name"] + suffix + ".aecbin")
    # aec-cc.exe (Windows) needs Windows paths; a native aec-cc takes WSL paths.
    args = ([AECCC, wpath(ptx), opt, "-o", wpath(out)] if AECCC.endswith(".exe")
            else [AECCC, ptx, opt, "-o", out])
    r = subprocess.run(args, capture_output=True, text=True)
    return out if r.returncode == 0 else None


def run_cmodel(c, aecbin):
    d = os.path.join(INROOT, c["name"])
    out_addr = dict((b[0], b[1]) for b in c["buffers"])[c["out"]]
    obytes = [b[2] for b in c["buffers"] if b[0] == c["out"]][0] * 4
    dump = "/tmp/%s_dump.bin" % c["name"]
    cmd = [AEC, "--program", aecbin, "--grid", "%d,%d,%d" % c["grid"],
           "--block", "%d,%d,%d" % c["block"], "--gmem-size", str(c["gmem"]),
           "--pmem-size", "65536", "--cmem-size", "65536", "--smem-size", "65536",
           "--lmem-size", "4096", "--max-steps", "50000000",
           "--load", "pmem:0:%s/pmem.bin" % d]
    for name, addr, _, _ in c["buffers"]:
        cmd += ["--load", "gmem:%d:%s/input_%s.bin" % (addr, d, name)]
    cmd += ["--dump", "%d:%d:%s" % (out_addr, obytes, dump)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip()), dump
    except Exception:
        return {"status": "ERR", "steps": -1}, dump


def spot(c, dump):
    d = os.path.join(INROOT, c["name"])
    rng = random.Random(0)
    if c["check"] == "matmul":
        M, N, K = c["MNK"]
        Af, Bf = "%s/input_A.bin" % d, "%s/input_B.bin" % d
        pts = [(0, 0), (M - 1, N - 1)] + [(rng.randrange(M), rng.randrange(N)) for _ in range(18)]
        mx = 0.0
        for (i, j) in pts:
            ref = sum(rd1(Af, i * K + k) * rd1(Bf, k * N + j) for k in range(K))
            mx = max(mx, abs(ref - rd1(dump, i * N + j)))
        return mx
    numel = [b[2] for b in c["buffers"] if b[0] == c["out"]][0]
    files = dict((b[0], "%s/input_%s.bin" % (d, b[0])) for b in c["buffers"])
    idxs = [0, 1, numel // 2, numel - 1] + [rng.randrange(numel) for _ in range(20)]
    mx = 0.0
    for i in idxs:
        vals = dict((n, rd1(files[n], i)) for n in files)
        mx = max(mx, abs(c["check"](vals) - rd1(dump, i)))
    return mx


def main():
    if not os.path.exists(AEC):
        print("CModel not found:", AEC); return 2
    print("%-24s %-7s %10s %10s %-7s %s" % ("case", "status", "O0 steps", "O2 steps", "O0/O2", "correct"))
    npass = 0
    for c in CASES:
        gen_inputs(c)
        o0bin = compile_case(c, "-O0", "_O0")
        o2bin = compile_case(c, "-O2", "")
        if not o0bin or not o2bin:
            print("%-24s COMPILE-ERR" % c["name"]); continue
        j0, _ = run_cmodel(c, o0bin)          # aec-precise is Linux: WSL paths
        j2, dump = run_cmodel(c, o2bin)
        tol = 1e-2 if c["check"] == "matmul" else 1e-3
        mx = spot(c, dump) if j2["status"] == "done" else float("nan")
        ok = j2["status"] == "done" and mx < tol
        npass += ok
        sp = (j0["steps"] / j2["steps"]) if j2["steps"] > 0 else 0.0
        print("%-24s %-7s %10s %10s %.3fx  %s (%.2g)" %
              (c["name"], j2["status"], j0["steps"], j2["steps"], sp,
               "PASS" if ok else "FAIL", mx))
    print("\n%d/%d correct on the official CModel" % (npass, len(CASES)))
    return 0 if npass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
