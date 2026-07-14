"""conformance.py - divergence + ISA-coverage regression against the OFFICIAL
AEC golden model (public/aec-cmodel-release/bin/aec-precise-*).

Motivation: the 5 public cases (verify.py) all use grids whose thread count is an
exact multiple of the problem size, so their bounds guards NEVER diverge. That
masked a real miscompile (a `cond ? a : b` branch left its else-assignment
unpredicated) and a broken `%laneid`. This suite deliberately exercises the
scenarios verify.py cannot:

  * a divergent `cond ? a : b` forward branch (single + multi-instruction body),
  * a bounds guard on a PARTIAL last block (thread count > N -> real divergence),
  * ISA ops the public cases never touch: %laneid, shl (SHL.u32 encoding per the
    organizer ruling), shr, and/or/xor, min/max.

Each kernel writes one u32 per thread to c[tid] and is checked against a pure
reference. Run inside WSL (the linux-x86_64 CModel):  python3 conformance.py
"""
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AEC = os.path.join(REPO, "public/aec-cmodel-release/bin/aec-precise-linux-x86_64")
AECCC = os.path.join(REPO, "C1/compiler/aec-cc.exe")
if not os.path.exists(AECCC):
    AECCC = os.path.join(REPO, "C1/compiler/aec-cc")   # native (Linux/ARM)
OBJ = os.path.join(REPO, "C1/bin/aec-objdump.exe")
if not os.path.exists(OBJ):
    OBJ = os.path.join(REPO, "C1/bin/aec-objdump")
SCRATCH = os.path.join(REPO, "C1/sim/build/conformance")
os.makedirs(SCRATCH, exist_ok=True)
WIN = AECCC.endswith(".exe")


def wpath(p):
    return subprocess.check_output(["wslpath", "-w", p]).decode().strip() if WIN else p


def compile_ptx(name, src, opt="-O0", extra_env=None):
    ptx = os.path.join(SCRATCH, name + ".ptx")
    aec = os.path.join(SCRATCH, name + ".aecbin")
    with open(ptx, "w") as f:
        f.write(src)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([AECCC, wpath(ptx), opt, "-o", wpath(aec)],
                       capture_output=True, text=True, env=env)
    return (aec, None) if r.returncode == 0 else (None, r.stderr.strip())


def disasm(aec):
    return subprocess.run([OBJ, wpath(aec)], capture_output=True, text=True).stdout


def run_cmodel(aec, grid, block, gmem, loads, dump_addr, dump_bytes):
    dump = os.path.join(SCRATCH, "dump.bin")
    ninstr = os.path.getsize(aec) // 16
    # aec-precise is the Linux CModel: it takes Linux paths directly. Only the
    # Windows aec-cc.exe / aec-objdump.exe need wpath()-converted paths.
    cmd = [AEC, "--program", aec, "--instructions", str(ninstr),
           "--grid", "%d,%d,%d" % grid, "--block", "%d,%d,%d" % block,
           "--gmem-size", str(gmem), "--pmem-size", "65536", "--cmem-size", "65536",
           "--smem-size", "65536", "--lmem-size", "4096", "--max-steps", "20000000"]
    for tgt, addr, path in loads:
        cmd += ["--load", "%s:%d:%s" % (tgt, addr, path)]
    cmd += ["--dump", "%d:%d:%s" % (dump_addr, dump_bytes, dump)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        status = json.loads(r.stdout)["status"]
    except Exception:
        status = (r.stdout.strip() or r.stderr.strip())[:80]
    data = b""
    if os.path.exists(dump):
        data = open(dump, "rb").read()
    return status, data


# --- per-thread kernels: result in %r9, stored to c[tid]; block = 32 ------------
_HDR = (".version 7.0\n.target sm_70\n.address_size 64\n"
        ".visible .entry k(.param .u64 pc, .param .u32 pn)\n{\n"
        "    .reg .pred %p<2>;\n    .reg .b32 %r<10>;\n    .reg .b64 %rd<4>;\n"
        "    ld.param.u64 %rd1, [pc];\n    mov.u32 %r1, %tid.x;\n")
_TAIL = ("    mul.wide.u32 %rd2, %r1, 4;\n    add.u64 %rd3, %rd1, %rd2;\n"
         "    st.global.u32 [%rd3], %r9;\n    ret;\n}\n")


def _pmem(*words):
    pm = bytearray(8 * len(words))
    for i, w in enumerate(words):
        struct.pack_into("<Q", pm, 8 * i, w)
    p = os.path.join(SCRATCH, "pm.bin")
    open(p, "wb").write(pm)
    return p


def _zeros(nbytes):
    p = os.path.join(SCRATCH, "z.bin")
    open(p, "wb").write(b"\x00" * nbytes)
    return p


results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))


def per_thread(name, body, ref, want_disasm=None):
    aec, err = compile_ptx(name, _HDR + body + _TAIL)
    if not aec:
        check(name, False, "compile-fail: " + (err or "")[:80]); return
    if want_disasm and want_disasm not in disasm(aec):
        check(name, False, "expected %r in disassembly" % want_disasm); return
    pm = _pmem(256, 32)
    status, data = run_cmodel(aec, (1, 1, 1), (32, 1, 1), 4096,
                              [("pmem", 0, pm), ("gmem", 256, _zeros(128))], 256, 128)
    if status != "done":
        check(name, False, "CModel=%s" % status); return
    got = list(struct.unpack("<32I", data[:128]))
    exp = [ref(t) & 0xffffffff for t in range(32)]
    check(name, got == exp, "" if got == exp else "got[:4]=%s exp[:4]=%s" % (got[:4], exp[:4]))


def main():
    if not os.path.exists(AEC):
        print("CModel not found:", AEC); return 2
    print("== ISA coverage (ops the public cases never exercise) ==")
    per_thread("laneid", "    mov.u32 %r9, %laneid;\n", lambda t: t)
    # SHL must encode as SHL.u32 (organizer ruling); also check the result.
    per_thread("shl_u32", "    shl.b32 %r9, %r1, 2;\n", lambda t: t << 2,
               want_disasm="SHL.u32")
    per_thread("shr", "    shr.u32 %r9, %r1, 1;\n", lambda t: t >> 1)
    per_thread("and", "    and.b32 %r9, %r1, 6;\n", lambda t: t & 6)
    per_thread("or", "    or.b32 %r9, %r1, 1;\n", lambda t: t | 1)
    per_thread("xor", "    xor.b32 %r9, %r1, 5;\n", lambda t: t ^ 5)
    per_thread("min", "    min.u32 %r9, %r1, 10;\n", lambda t: min(t, 10))
    per_thread("max", "    max.u32 %r9, %r1, 10;\n", lambda t: max(t, 10))

    print("== divergence (verify.py's exact-cover grids never trigger these) ==")
    # cond ? 100 : 200 -- a divergent forward branch (single-instruction body).
    per_thread("cond_select",
               "    mov.u32 %r9, 100;\n    setp.lt.u32 %p1, %r1, 16;\n"
               "    @%p1 bra L;\n    mov.u32 %r9, 200;\nL:\n",
               lambda t: 100 if t < 16 else 200)
    # cond ? (tid*3+7) : (tid+1000) -- multi-instruction escaping body.
    per_thread("cond_chain",
               "    add.u32 %r9, %r1, 1000;\n    setp.lt.u32 %p1, %r1, 16;\n"
               "    @%p1 bra L;\n    mul.lo.u32 %r4, %r1, 3;\n    add.u32 %r9, %r4, 7;\nL:\n",
               lambda t: (t + 1000) if t < 16 else (t * 3 + 7))

    # Partial last block: 1000 elements, 1024 threads -> lanes 1000..1023 diverge
    # on the bounds guard and must NOT store (out-of-bounds), leaving the sentinel.
    print("== bounds guard on a partial last block (real OOB divergence) ==")
    _partial_block()

    print("== unroll correctness regressions (official CModel differential) ==")
    _unroll_regressions()

    npass = sum(1 for _, ok in results if ok)
    print("\n%d/%d conformance checks passed" % (npass, len(results)))
    return 0 if npass == len(results) else 1


def _partial_block():
    ptx = os.path.join(REPO, "public/C1编译器赛题/testcases/T1_basic_lowering/kernel.ptx")
    if not os.path.exists(ptx):
        check("partial_block", False, "T1 kernel not found"); return
    CAP, N, BLK, GRID, SENT = 1024, 1000, 128, 8, 7777.0
    A, B, C = 256, 256 + CAP * 4, 256 + 2 * CAP * 4
    a = [float(i % 10) + 0.5 for i in range(CAP)]
    b = [float((i * 3) % 7) + 0.25 for i in range(CAP)]
    def wf(path, vals):
        open(path, "wb").write(b"".join(struct.pack("<f", v) for v in vals))
    pa, pb, pc = (os.path.join(SCRATCH, x) for x in ("a.bin", "b.bin", "c.bin"))
    wf(pa, a); wf(pb, b); wf(pc, [SENT] * CAP)
    pm = _pmem(A, B, C)  # 3 u64 ptrs; n goes in the 4th u32 slot below
    pmb = bytearray(open(pm, "rb").read()); pmb += struct.pack("<I", N)
    open(pm, "wb").write(pmb)
    for opt in ("-O0", "-O2"):
        aec, err = compile_ptx("va_partial", open(ptx).read(), opt)
        if not aec:
            check("partial_block%s" % opt, False, "compile-fail"); continue
        status, data = run_cmodel(
            aec, (GRID, 1, 1), (BLK, 1, 1), C + CAP * 4 + 256,
            [("pmem", 0, pm), ("gmem", A, pa), ("gmem", B, pb), ("gmem", C, pc)],
            C, CAP * 4)
        cout = struct.unpack("<%df" % CAP, data[:CAP * 4])
        in_ok = all(abs(cout[i] - (a[i] + b[i])) < 1e-4 for i in range(N))
        oob_ok = all(cout[i] == SENT for i in range(N, CAP))
        check("partial_block%s" % opt, status == "done" and in_ok and oob_ok,
              "status=%s in_range=%s oob_masked=%s" % (status, in_ok, oob_ok))


def _run_scalar_gemm(name, src, opt, env=None, a_offset=256, b_offset=512):
    """Compile/run a 1x1 scalar GEMM and return (status, C[0], disasm)."""
    A, B, C, K = a_offset, b_offset, 768, 16
    def wf(path, vals):
        open(path, "wb").write(b"".join(struct.pack("<f", v) for v in vals))
    pa, pb, pc = (os.path.join(SCRATCH, name + x) for x in ("_a.bin", "_b.bin", "_c.bin"))
    # Deliberately keep non-zero tail data after K: the old split-loop bound bug
    # executed indices 16..18 when the induction variable started at three.
    wf(pa, [0.25 + 0.03125 * i for i in range(32)])
    wf(pb, [0.5 - 0.0078125 * i for i in range(32)])
    wf(pc, [1.25])
    pm = os.path.join(SCRATCH, name + "_pm.bin")
    open(pm, "wb").write(struct.pack("<QQQIII", A, B, C, 1, 1, K))
    aec, err = compile_ptx(name, src, opt, env)
    if not aec:
        return "compile-fail: " + (err or "")[:80], None, ""
    status, data = run_cmodel(
        aec, (1, 1, 1), (1, 1, 1), 2048,
        [("pmem", 0, pm), ("gmem", A, pa), ("gmem", B, pb), ("gmem", C, pc)],
        C, 4)
    value = struct.unpack("<f", data[:4])[0] if len(data) >= 4 else None
    return status, value, disasm(aec)


def _unroll_regressions():
    ptx = os.path.join(REPO, "public/C1编译器赛题/testcases/T5_scalar_gemm/kernel.ptx")
    if not os.path.exists(ptx):
        check("unroll_regressions", False, "T5 kernel not found"); return
    base = open(ptx).read()

    # A store to A[1] between the A[0] and A[1] loads makes an early LD.b64
    # observably wrong.  Since PTX parameters may alias, any write-bearing loop
    # must retain scalar loads until alias analysis can prove disjointness.
    memdep = base.replace(
        "    mov.u32 %r14, 1;",
        "    mov.u32 %r14, 1;\n    mov.u64 %rd10, 4;\n"
        "    add.u64 %rd10, %rd1, %rd10;")
    memdep = memdep.replace(
        "    ld.global.f32 %f2, [%rd7];",
        "    ld.global.f32 %f2, [%rd7];\n    st.global.f32 [%rd10], %f1;")
    s0, v0, _ = _run_scalar_gemm("unroll_memdep_o0", memdep, "-O0")
    s2, v2, d2 = _run_scalar_gemm("unroll_memdep_o2", memdep, "-O2")
    ok = s0 == s2 == "done" and v0 == v2 and "LD.gmem.b64" not in d2
    check("unroll_memdep", ok, "O0=%r O2=%r b64=%s" % (v0, v2, "LD.gmem.b64" in d2))

    # kmain must be relative to the current counter, not to zero:
    #   start + floor((bound-start)/U)*U.
    nonzero = base.replace("    mov.u32 %r13, 0;", "    mov.u32 %r13, 3;")
    s0, v0, _ = _run_scalar_gemm("unroll_start3_o0", nonzero, "-O0")
    s2, v2, _ = _run_scalar_gemm(
        "unroll_start3_o2", nonzero, "-O2", {"AEC_NO_MEM_COALESCE": "1"})
    ok = s0 == s2 == "done" and v0 == v2
    check("unroll_nonzero_start", ok, "O0=%r O2=%r" % (v0, v2))

    # Same case with a compile-time bound.  `bound % group == 0` is not enough:
    # divisibility is based on (bound-start), which is 13 here.
    const_bound = nonzero.replace(
        "    ld.param.u32 %r3,  [param_K];", "    mov.u32 %r3, 16;")
    s0, v0, _ = _run_scalar_gemm("unroll_const_start3_o0", const_bound, "-O0")
    s2, v2, _ = _run_scalar_gemm(
        "unroll_const_start3_o2", const_bound, "-O2", {"AEC_NO_MEM_COALESCE": "1"})
    ok = s0 == s2 == "done" and v0 == v2
    check("unroll_const_nonzero_start", ok, "O0=%r O2=%r" % (v0, v2))

    # Scalar f32 permits 4-byte alignment, while LD.b64 requires eight.  The
    # optimized image may contain b64, but its entry guard must send this input
    # (A/B both 4 mod 8) through the original scalar loop.
    s0, v0, _ = _run_scalar_gemm(
        "unroll_unaligned_o0", base, "-O0", a_offset=260, b_offset=516)
    s2, v2, d2 = _run_scalar_gemm(
        "unroll_unaligned_o2", base, "-O2", a_offset=260, b_offset=516)
    ok = s0 == s2 == "done" and v0 == v2 and "LD.gmem.b64" in d2
    check("unroll_b64_alignment_fallback", ok,
          "O0=%r O2=%r optimized_has_b64=%s" % (v0, v2, "LD.gmem.b64" in d2))


if __name__ == "__main__":
    sys.exit(main())
