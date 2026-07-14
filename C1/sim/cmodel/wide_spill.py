#!/usr/bin/env python3
"""Force real 64-bit register-pair spills and verify them on official CModel."""
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AEC = os.path.join(REPO, "public/aec-cmodel-release/bin/aec-precise-linux-x86_64")
AECCC = os.path.join(REPO, "C1/compiler/aec-cc")
BUILD = os.path.join(REPO, "C1/sim/build/wide_spill")
N = 130


def generate():
    lines = [
        ".version 9.3", ".target sm_90", ".address_size 64",
        ".visible .entry wide_spill(.param .u64 pout)", "{",
        "    .reg .u32 %r<2>;", "    .reg .u64 %rd<2>;",
        "    .reg .f64 %%fd<%d>;" % (N + 1),
        "    ld.param.u64 %rd0, [pout];",
    ]
    for i in range(N):
        lines += [
            "    mov.u32 %%r0, %d;" % (256 + 8 * i),
            "    ld.global.f64 %%fd%d, [%%r0];" % i,
        ]
    lines += ["    bra REDUCE;", "REDUCE:",
              "    add.f64 %%fd%d, %%fd0, %%fd1;" % N]
    for i in range(2, N):
        lines.append("    add.f64 %%fd%d, %%fd%d, %%fd%d;" % (N, N, i))
    lines += ["    st.global.f64 [%%rd0], %%fd%d;" % N, "    ret;", "}"]
    return "\n".join(lines) + "\n"


def main():
    if not os.path.exists(AEC):
        print("CModel not found:", AEC); return 2
    os.makedirs(BUILD, exist_ok=True)
    ptx = os.path.join(BUILD, "wide_spill.ptx")
    out = os.path.join(BUILD, "wide_spill.aecbin")
    report = os.path.join(BUILD, "wide_spill.json")
    pmem = os.path.join(BUILD, "pmem.bin")
    gmem = os.path.join(BUILD, "gmem.bin")
    dump = os.path.join(BUILD, "dump.bin")
    open(ptx, "w").write(generate())
    open(pmem, "wb").write(struct.pack("<Q", 0))
    buf = bytearray(256 + N * 8)
    for i in range(N):
        struct.pack_into("<d", buf, 256 + 8 * i, 1.0)
    open(gmem, "wb").write(buf)

    c = subprocess.run([AECCC, ptx, "-O2", "-o", out, "--report", report],
                       capture_output=True, text=True)
    if c.returncode:
        print("COMPILE-FAIL", c.stderr); return 1
    rep = json.load(open(report))
    spill_loads = rep["spills"]["loads"]
    spill_stores = rep["spills"]["stores"]
    ninstr = os.path.getsize(out) // 16
    r = subprocess.run([
        AEC, "--program", out, "--instructions", str(ninstr),
        "--grid", "1,1,1", "--block", "1,1,1",
        "--gmem-size", "4096", "--pmem-size", "64", "--cmem-size", "64",
        "--smem-size", "64", "--lmem-size", "8192", "--max-steps", "2000000",
        "--load", "pmem:0:" + pmem, "--load", "gmem:0:" + gmem,
        "--dump", "0:8:" + dump,
    ], capture_output=True, text=True)
    try:
        status = json.loads(r.stdout)["status"]
    except Exception:
        status = r.stdout.strip() or r.stderr.strip()
    got = struct.unpack("<d", open(dump, "rb").read(8))[0] if os.path.exists(dump) else None
    ok = status == "done" and got == float(N) and spill_loads > 0 and spill_stores > 0
    print("wide pairs=%d spill_loads=%d spill_stores=%d output=%r expected=%r "
          "status=%s -> %s" %
          (N, spill_loads, spill_stores, got, float(N), status,
           "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
