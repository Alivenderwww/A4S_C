#!/usr/bin/env python3
"""C3.1 CLI: parse an ONNX model and export its DAG as JSON.

Usage (command template submitted at 报名):

    python tools/export_dag.py --onnx {onnx} --output {output}

Writes a JSON document with ``format_version`` / ``graph_inputs`` /
``graph_outputs`` / ``nodes`` / ``edges`` (field names reuse the original ONNX
node & tensor names).  Exits 0 on success; non-zero on failure.  stdout is not
graded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the C3 package importable no matter the caller's cwd.
_C3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3_ROOT not in sys.path:
    sys.path.insert(0, _C3_ROOT)

from scheduler.graph import import_onnx_graph  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export ONNX computation graph to DAG JSON (C3.1)")
    ap.add_argument("--onnx", required=True, help="input ONNX model path")
    ap.add_argument("--output", required=True, help="output DAG JSON path")
    args = ap.parse_args(argv)

    # C3.1 exports graph structure only -- skip the external-data weight blob so
    # BigFormer's 19 GB .onnx.data is never read (instant, no timeout risk).
    graph = import_onnx_graph(args.onnx, load_weights=False)
    dag = graph.to_dag_dict()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dag, f, indent=2, ensure_ascii=False)

    # stdout is ignored by the grader; print a short summary for humans.
    print(
        f"[export_dag] {args.onnx}: {len(dag['nodes'])} nodes, "
        f"{len(dag['edges'])} edges -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # ensure a non-zero exit on any failure
        print(f"[export_dag] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
