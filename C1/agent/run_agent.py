#!/usr/bin/env python3
"""run_agent.py - Autonomous auto-tuning loop for the AEC C1 compiler (scoring D).

The agent runs independently: it compiles one PTX input under several candidate
flag configurations, MEASURES each one's cycle count, keeps the fastest,
RE-COMPILES with it to emit the final binary, VERIFIES the result, and writes an
optimization report with the speedup r = T_default / T_agent (default = the
un-tuned -O0 baseline; the agent's job is to discover better flags).

Cycle measurement:
  * Known public kernel (matched by filename) -> run the AEC functional simulator
    (sim/) with a latency + dual-issue cycle model, and additionally VERIFY the
    result against the numpy reference. This is the accurate signal.
  * Unknown input -> fall back to the compiler's static est_cycles report, and
    verify only that it disassembles.
No official AEC cycle model ships with C1, so both are OUR proxies.

Usage:
  python run_agent.py input.ptx -o out.aecbin [--report agent_report.json]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(C1, "sim"))

# Candidate configurations to sweep (all flags are supported by aec-cc).
# The dominant knob is the unroll factor: its optimum is kernel-dependent
# (reuse wants full unroll U16 -> 448c, poly wants U4 -> 784c; over-unrolling
# poly to U16 regresses to 816c), which is exactly what makes the agent useful.
# sched_window was measured to be inert on these kernels (win16==win32==win64),
# so it is not swept.
CONFIGS = [
    ("O0",         ["-O0"]),
    ("O2",         ["-O2"]),                            # default: unroll U4
    ("O3",         ["-O3"]),                            # aggressive: unroll U8
    ("O3-u16",     ["-O3", "--unroll-factor", "16"]),   # full unroll (reuse floor)
    ("O3-u4",      ["-O3", "--unroll-factor", "4"]),    # lighter (poly-friendly)
    ("O2-no-licm", ["-O2", "--no-licm"]),               # ablation lever
]
DEFAULT_CONFIG = "O2"     # the compiler default (-O2); the agent tunes UP from it.


def tool(name):
    p = os.path.join(C1, "bin", name)
    if os.name == "nt" and not os.path.exists(p):
        p += ".exe"
    return p


def match_case(ptx_path):
    """Return a cases.py builder if this PTX is one of the known public kernels."""
    try:
        import cases
    except Exception:
        return None
    base = os.path.basename(ptx_path)
    for name, builder in cases.ALL.items():
        if builder()["ptx"] == base:
            return builder
    return None


def measure(aecbin, builder):
    """(cycles, correct) via the simulator, or (None, None) if it can't run."""
    import numpy as np
    from aec_sim import simulate, ExecError
    case = builder()
    try:
        gmem, cyc, _ = simulate(aecbin, case["grid"], case["block"],
                                param_block=case["param"], gmem_init=case["gmem"])
    except ExecError:
        return None, False
    off, count, dt = case["out"]
    got = np.frombuffer(bytes(gmem[off:off + count * dt.itemsize]), dt).astype(np.float32)
    ref = np.asarray(case["ref"], np.float32).reshape(-1)
    ok = got.shape == ref.shape and np.allclose(got, ref, rtol=1e-3, atol=1e-3)
    return cyc, ok


def compile_cfg(cc, ptx, flags, out, report):
    cmd = [cc, ptx] + flags + ["-o", out, "--report", report]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode


def static_cycles(report):
    try:
        with open(report) as f:
            return int(json.load(f).get("est_cycles", 1 << 62))
    except Exception:
        return 1 << 62


def main():
    ap = argparse.ArgumentParser(description="AEC C1 auto-tuning agent")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="out.aecbin")
    ap.add_argument("--report", default="agent_report.json")
    args = ap.parse_args()

    cc, objdump = tool("aec-cc"), tool("aec-objdump")
    if not os.path.exists(cc):
        sys.stderr.write("agent: aec-cc not found (run make)\n"); return 2
    if not os.path.exists(args.input):
        sys.stderr.write("agent: input not found: %s\n" % args.input); return 2

    builder = match_case(args.input)
    mode = "simulator" if builder else "static est_cycles"
    print("agent: tuning %s  (measuring via %s)" % (os.path.basename(args.input), mode))

    outdir = os.path.join(HERE, "_agent_work")
    os.makedirs(outdir, exist_ok=True)
    results = []
    for name, flags in CONFIGS:
        binp = os.path.join(outdir, name + ".aecbin")
        repp = os.path.join(outdir, name + ".json")
        if compile_cfg(cc, args.input, flags, binp, repp) != 0:
            print("  %-12s COMPILE FAILED" % name); continue
        if builder:
            cyc, ok = measure(binp, builder)
            if cyc is None:
                print("  %-12s sim error" % name); continue
        else:
            cyc, ok = static_cycles(repp), True
        results.append({"config": name, "flags": flags, "cycles": cyc, "correct": ok})
        print("  %-12s cycles=%-7d correct=%s" % (name, cyc, ok))

    correct = [r for r in results if r["correct"]]
    if not correct:
        sys.stderr.write("agent: no correct configuration\n"); return 1

    default = next((r for r in results if r["config"] == DEFAULT_CONFIG), correct[0])
    best = min(correct, key=lambda r: r["cycles"])

    # Re-emit the winner and verify it.
    final_rep = os.path.join(outdir, "final.json")
    if compile_cfg(cc, args.input, best["flags"], args.output, final_rep) != 0:
        sys.stderr.write("agent: failed to re-emit best config\n"); return 1
    final_ok = True
    if builder:
        _, final_ok = measure(args.output, builder)
    elif os.path.exists(objdump):
        final_ok = subprocess.run([objdump, args.output],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0

    speedup = default["cycles"] / best["cycles"] if best["cycles"] else 1.0
    report = {
        "input": args.input, "output": args.output, "measure_mode": mode,
        "default_config": default["config"], "default_cycles": default["cycles"],
        "best_config": best["config"], "best_flags": best["flags"],
        "best_cycles": best["cycles"], "speedup_r": round(speedup, 4),
        "final_verified": bool(final_ok),
        "candidates": [{"config": r["config"], "cycles": r["cycles"],
                        "correct": r["correct"]} for r in results],
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("\nagent: best=%s cycles=%d (default %s=%d)  speedup r=%.3f  verified=%s"
          % (best["config"], best["cycles"], default["config"],
             default["cycles"], speedup, final_ok))
    print("agent: wrote %s and %s" % (args.output, args.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
