"""mad_semantics.py - verify `mad.f32` lowers to non-fused MAD (spec §6.2/§9),
NOT fused FMA, against the official CModel.

a*b rounds, then +c rounds: 4097*4097 = 16785409 -> f32 16785408 (round-even
drops the +1); c = -16785408.  So MAD.f32 (non-fused) = 0, FMA.f32 (fused) = 1.
This regression catches a mad.f32 -> FMA relapse (fused rounding disagrees with
the reference).  Run in WSL:  python3 C1/sim/cmodel/mad_semantics.py
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
PTX = os.path.join(HERE, "mad_probe.ptx")
BUILD = os.path.join(REPO, "C1/sim/build/cmodel")
OUT = os.path.join(BUILD, "mad_probe.aecbin")


def wpath(p):
    return subprocess.check_output(["wslpath", "-w", p]).decode().strip()


def main():
    os.makedirs(BUILD, exist_ok=True)
    args = ([AECCC, wpath(PTX), "-O2", "-o", wpath(OUT)] if AECCC.endswith(".exe")
            else [AECCC, PTX, "-O2", "-o", OUT])
    if subprocess.run(args, capture_output=True, text=True).returncode:
        print("COMPILE-ERR"); return 2

    open("/tmp/a.bin", "wb").write(struct.pack("<f", 4097.0))
    open("/tmp/b.bin", "wb").write(struct.pack("<f", 4097.0))
    open("/tmp/c.bin", "wb").write(struct.pack("<f", -16785408.0))
    pm = bytearray(40)
    for off, val in [(0, 256), (8, 260), (16, 264), (24, 268), (32, 272)]:
        pm[off:off + 8] = struct.pack("<Q", val)
    open("/tmp/pm.bin", "wb").write(pm)

    cmd = [AEC, "--program", OUT, "--grid", "1,1,1", "--block", "1,1,1",
           "--gmem-size", "512", "--pmem-size", "64", "--max-steps", "1000",
           "--load", "pmem:0:/tmp/pm.bin", "--load", "gmem:256:/tmp/a.bin",
           "--load", "gmem:260:/tmp/b.bin", "--load", "gmem:264:/tmp/c.bin",
           "--dump", "268:8:/tmp/mf.bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("cmodel:", r.stdout.strip())
    m, f = struct.unpack("<2f", open("/tmp/mf.bin", "rb").read())
    ok = (m == 0.0 and f == 1.0)
    print("MAD.f32=%s (non-fused, expect 0.0)  FMA.f32=%s (fused, expect 1.0)  -> %s"
          % (m, f, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
