"""pressure.py - register-pressure robustness on the official CModel.

Generates a kernel with 300 simultaneously-live runtime values (r_i = tid + i,
all summed at the end), which naively needs 300 physical registers. The pre-RA
list scheduler sinks each independent `add` next to its single use, so O2
allocates only a handful of physical registers and never spills. This verifies
the scheduler's register-pressure reduction keeps us correct without a spiller
(the C robustness variants stress "register pressure increase").

Run in WSL:  python3 C1/sim/cmodel/pressure.py
"""
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AEC = os.path.join(REPO, "public/aec-cmodel-release/bin/aec-precise-linux-x86_64")
AECCC = os.path.join(REPO, "C1/compiler/aec-cc.exe")
if not os.path.exists(AECCC):
    AECCC = os.path.join(REPO, "C1/compiler/aec-cc")
BUILD = os.path.join(REPO, "C1/sim/build/cmodel")
N = 300


def wpath(p):
    return subprocess.check_output(["wslpath", "-w", p]).decode().strip()


def gen():
    L = [".version 9.3", ".target sm_90", ".address_size 64",
         ".visible .entry hp(.param .u64 pout)", "{",
         ".reg .u64 %rd<2>;", ".reg .u32 %r<{}>;".format(N + 3),
         "ld.param.u64 %rd1, [pout];", "mov.u32 %r0, %tid.x;"]
    for i in range(1, N + 1):
        L.append("add.u32 %r{}, %r0, {};".format(i, i))     # runtime-live values
    acc = N + 1
    L.append("add.u32 %r{}, %r1, %r2;".format(acc))
    for i in range(3, N + 1):
        L.append("add.u32 %r{}, %r{}, %r{};".format(acc, acc, i))
    L.append("st.global.u32 [%rd1], %r{};".format(acc))
    L.append("ret;")
    L.append("}")
    os.makedirs(BUILD, exist_ok=True)
    p = os.path.join(BUILD, "hp.ptx")
    open(p, "w").write("\n".join(L))
    return p


def main():
    ptx = gen()
    out = os.path.join(BUILD, "hp.aecbin")
    rep = os.path.join(BUILD, "hp_report.json")
    args = ([AECCC, wpath(ptx), "-O2", "-o", wpath(out), "--report", wpath(rep)]
            if AECCC.endswith(".exe")
            else [AECCC, ptx, "-O2", "-o", out, "--report", rep])
    if subprocess.run(args, capture_output=True, text=True).returncode:
        print("COMPILE-ERR"); return 2
    import json
    d = json.load(open(rep, encoding="utf-8"))
    phys = d["num_physical_registers"]; spills = d["diagnostics"]["spill_count"]

    open("/tmp/hpo.bin", "wb").write(struct.pack("<Q", 256))
    cmd = [AEC, "--program", out, "--grid", "1,1,1", "--block", "1,1,1",
           "--gmem-size", "512", "--pmem-size", "32", "--lmem-size", "8192",
           "--max-steps", "200000", "--load", "pmem:0:/tmp/hpo.bin",
           "--dump", "256:4:/tmp/hpd.bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    val = struct.unpack("<I", open("/tmp/hpd.bin", "rb").read())[0]
    ref = N * (N + 1) // 2
    ok = (val == ref) and spills == 0
    print("cmodel:", r.stdout.strip())
    print("%d runtime-live vregs -> phys_regs=%d spills=%d ; out=%d (ref %d) -> %s"
          % (N, phys, spills, val, ref, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
