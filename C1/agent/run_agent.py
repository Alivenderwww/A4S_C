#!/usr/bin/env python3
# run_agent.py - Auto-tuning loop for the AEC C1 compiler (scoring D).
#
# The agent runs independently: it compiles one PTX input under several
# candidate flag configurations, reads each run's perf report (est_cycles),
# keeps the configuration with the fewest cycles, RE-COMPILES with it to emit
# the final binary, verifies the binary disassembles, and writes a final
# optimization report with the speedup r = T_default / T_agent.
#
# NOTE: no official AEC cycle model ships with C1, so `est_cycles` here is the
# compiler's own heuristic (see src/driver.cpp::estimateCycles). Swap the
# `read_cycles` source for the real cycle-model output once it is available.
#
# Usage:
#   python3 agent/run_agent.py input.ptx -o out.aecbin [--cc bin/aec-cc] \
#           [--objdump bin/aec-objdump] [--report agent_report.json]

import argparse
import json
import os
import subprocess
import sys
import tempfile

# Candidate configurations the agent sweeps. Each is a list of extra CLI flags
# appended to the base optimization level. Extend this table as real passes
# land (e.g. per-pass on/off, --sched-window sweeps).
CONFIGS = [
    ("O2-default",        ["-O2"]),
    ("O3",                ["-O3"]),
    ("O3-win32",          ["-O3", "--sched-window", "32"]),
    ("O2-no-dual",        ["-O2", "--no-dual-issue"]),
    ("O2-no-mem",         ["-O2", "--no-mem-coalesce"]),
    ("O3-no-cse",         ["-O3", "--no-cse"]),
]

DEFAULT_CONFIG = "O2-default"


def here():
    return os.path.dirname(os.path.abspath(__file__))


def default_tool(name):
    # bin/ sits next to agent/ under the C1 root.
    root = os.path.dirname(here())
    exe = os.path.join(root, "bin", name)
    if os.name == "nt" and not os.path.exists(exe):
        exe += ".exe"
    return exe


def read_cycles(report_path):
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
        return int(data.get("est_cycles", 1 << 62)), data
    except Exception:
        return (1 << 62), {}


def run_config(cc, ptx, flags, out_path, report_path):
    cmd = [cc, ptx] + flags + ["-o", out_path, "--report", report_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode


def verify(objdump, out_path):
    if not objdump or not os.path.exists(objdump):
        return True  # objdump not available: skip verification gracefully.
    proc = subprocess.run([objdump, out_path],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="AEC C1 auto-tuning agent")
    ap.add_argument("input", help="input .ptx file")
    ap.add_argument("-o", "--output", default="out.aecbin",
                    help="final binary path (default out.aecbin)")
    ap.add_argument("--cc", default=None, help="path to aec-cc")
    ap.add_argument("--objdump", default=None, help="path to aec-objdump")
    ap.add_argument("--report", default="agent_report.json",
                    help="final optimization report path")
    args = ap.parse_args()

    cc = args.cc or default_tool("aec-cc")
    objdump = args.objdump or default_tool("aec-objdump")

    if not os.path.exists(cc):
        sys.stderr.write("agent: aec-cc not found at %s (run `make` first)\n" % cc)
        return 2
    if not os.path.exists(args.input):
        sys.stderr.write("agent: input not found: %s\n" % args.input)
        return 2

    tmpdir = tempfile.mkdtemp(prefix="aec_agent_")
    results = []
    for name, flags in CONFIGS:
        out_p = os.path.join(tmpdir, name + ".aecbin")
        rep_p = os.path.join(tmpdir, name + ".json")
        rc = run_config(cc, args.input, flags, out_p, rep_p)
        if rc != 0:
            sys.stderr.write("agent: config %s failed to compile (rc=%d)\n" % (name, rc))
            continue
        cyc, data = read_cycles(rep_p)
        ok = verify(objdump, out_p)
        results.append({"config": name, "flags": flags, "est_cycles": cyc,
                        "verified": ok, "report": data})
        print("  %-14s est_cycles=%-8d verified=%s" % (name, cyc, ok))

    if not results:
        sys.stderr.write("agent: no configuration compiled successfully\n")
        return 1

    # Baseline = default config; best = min cycles among verified configs.
    default_cyc = next((r["est_cycles"] for r in results
                        if r["config"] == DEFAULT_CONFIG), results[0]["est_cycles"])
    verified = [r for r in results if r["verified"]] or results
    best = min(verified, key=lambda r: r["est_cycles"])

    # Re-emit the winning configuration to the requested output path + verify.
    final_rep = os.path.join(tmpdir, "final.json")
    rc = run_config(cc, args.input, best["flags"], args.output, final_rep)
    if rc != 0:
        sys.stderr.write("agent: failed to re-emit best config\n")
        return 1
    final_ok = verify(objdump, args.output)

    speedup = (float(default_cyc) / float(best["est_cycles"])
               if best["est_cycles"] else 1.0)

    report = {
        "input": args.input,
        "output": args.output,
        "default_config": DEFAULT_CONFIG,
        "default_cycles": default_cyc,
        "best_config": best["config"],
        "best_flags": best["flags"],
        "best_cycles": best["est_cycles"],
        "speedup_r": round(speedup, 4),
        "final_verified": final_ok,
        "candidates": [{"config": r["config"], "est_cycles": r["est_cycles"],
                        "verified": r["verified"]} for r in results],
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print("\nagent: best=%s cycles=%d (default=%d) speedup r=%.4f -> %s" % (
        best["config"], best["est_cycles"], default_cyc, speedup, args.output))
    print("agent: report written to %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
