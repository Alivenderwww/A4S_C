"""corners.py - §3 subset corner-case checks on the official CModel.

Covers constructs the 5 public cases don't exercise. Currently:
  * `@!%p bra` (negated-predicate branch) as a do-while back-edge -> a BRX with
    pred_neg[14] AND pred_en[15]; the public cases only use `@%p bra`, so this
    validates the negated branch on the real oracle (pred_en was just fixed).

Run in WSL:  python3 C1/sim/cmodel/corners.py
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

NEGLOOP = """.version 9.3
.target sm_90
.address_size 64
.visible .entry negloop(.param .u64 pout, .param .u32 pn)
{
    .reg .pred %p<2>;
    .reg .u64 %rd<2>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd1, [pout];
    ld.param.u32 %r1, [pn];
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
LOOP:
    add.u32 %r2, %r2, %r3;
    add.u32 %r3, %r3, 1;
    setp.ge.u32 %p1, %r3, %r1;
    @!%p1 bra LOOP;
    st.global.u32 [%rd1], %r2;
    ret;
}
"""


def wpath(p):
    return subprocess.check_output(["wslpath", "-w", p]).decode().strip()


def compile_run(ptx_path, opt, K):
    out = os.path.join(BUILD, "negloop_%s.aecbin" % opt.strip("-"))
    args = ([AECCC, wpath(ptx_path), opt, "-o", wpath(out)] if AECCC.endswith(".exe")
            else [AECCC, ptx_path, opt, "-o", out])
    if subprocess.run(args, capture_output=True, text=True).returncode:
        return None, "COMPILE-ERR"
    pm = struct.pack("<Q", 256) + struct.pack("<I", K) + b"\x00\x00\x00\x00"
    open("/tmp/nl_pm.bin", "wb").write(pm)
    cmd = [AEC, "--program", out, "--grid", "1,1,1", "--block", "1,1,1",
           "--gmem-size", "512", "--pmem-size", "32", "--max-steps", "100000",
           "--load", "pmem:0:/tmp/nl_pm.bin", "--dump", "256:4:/tmp/nl_o.bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    status = r.stdout.strip()
    val = struct.unpack("<I", open("/tmp/nl_o.bin", "rb").read())[0]
    return val, status


def main():
    os.makedirs(BUILD, exist_ok=True)
    ptx = os.path.join(BUILD, "negloop.ptx")
    open(ptx, "w").write(NEGLOOP)
    K = 100
    ref = K * (K - 1) // 2       # sum(0..K-1)
    ok = True
    for opt in ["-O0", "-O2"]:
        val, status = compile_run(ptx, opt, K)
        good = val == ref
        ok = ok and good
        print("@!%%p bra  %s  out=%s (ref %d)  %s   [%s]" %
              (opt, val, ref, "PASS" if good else "FAIL", status))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
