#!/usr/bin/env python3
"""Device oracle scanner: evaluate all GEMM variants on a representative grid.

Runs on Linux with libaec_device.so loaded via libaec.so.
Outputs scan_results.json for offline analysis.

Usage (from C2 root):
    python3 -B tests/device_oracle_scan.py [--output scan_results.json]
"""
import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
C2 = os.path.dirname(HERE)

sys.path.insert(0, C2)
sys.path.insert(0, os.path.join(C2, "grader"))
from public_grade import (  # noqa: E402
    Runtime, DTYPE_BY_NAME, SUCCESS, DeviceCompletion,
)

DTYPES = list(DTYPE_BY_NAME.keys())

SHAPES = [
    (1, 1, 1), (2, 2, 2),
    (8, 8, 8), (16, 16, 16), (64, 64, 64), (128, 128, 128), (256, 256, 256),
    (12, 12, 12), (20, 20, 20), (36, 36, 36),
    (7, 9, 5), (17, 13, 9),
    (1, 256, 1), (256, 1, 256), (2, 128, 2), (1, 1, 256),
]

ALIGNMENTS = [8, 16, 64]
WORKSPACES = [0, 4096, 8192]

VARIANTS = [
    ("naive", 10, 1, 0, 1, 1),
    ("tiled", 11, 2, 4096, 1, 4),
    ("vectorized", 12, 3, 8192, 16, 8),
]


def _legal(vname_ws_align_div, m, n, k, alignment, workspace):
    _, ws, align, div = vname_ws_align_div
    if ws > workspace:
        return False
    if align > alignment:
        return False
    if div > 1 and (m % div or n % div or k % div):
        return False
    return True


def scan(runtime):
    results = []
    for dtype_name in DTYPES:
        dtype_val = DTYPE_BY_NAME[dtype_name]
        for (m, n, k) in SHAPES:
            for alignment in ALIGNMENTS:
                for workspace in WORKSPACES:
                    case = {
                        "dtype": dtype_name,
                        "dtype_val": dtype_val,
                        "m": m, "n": n, "k": k,
                        "alignment": alignment,
                        "workspace": workspace,
                        "variants": {},
                    }
                    for vname, kid, variant, ws, align, div in VARIANTS:
                        if not _legal((vname, ws, align, div),
                                      m, n, k, alignment, workspace):
                            continue
                        comp = DeviceCompletion()
                        status = runtime.device.aecDeviceEvaluateKernel(
                            kid, dtype_val, variant,
                            m, n, k, alignment, workspace,
                            ctypes.byref(comp),
                        )
                        if status != SUCCESS:
                            case["variants"][vname] = {"status": status}
                            continue
                        case["variants"][vname] = {
                            "cycles": comp.virtual_cycles,
                            "retired": comp.instructions_retired,
                            "digest": comp.trace_digest,
                        }
                    results.append(case)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=".")
    ap.add_argument("--output", default="scan_results.json")
    args = ap.parse_args()

    runtime = Runtime(Path(args.submission))
    results = scan(runtime)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    multi = sum(1 for c in results if len(c["variants"]) >= 2)
    print("scanned %d cases -> %s" % (len(results), args.output))
    print("  cases with 2+ legal variants: %d" % multi)


if __name__ == "__main__":
    main()
