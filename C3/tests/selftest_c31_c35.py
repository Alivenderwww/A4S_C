#!/usr/bin/env python3
"""End-to-end self-test for C3.1 (DAG export) and C3.5 (inference).

Runs the actual CLI entry points against the three public models and checks:
  * C3.1: DAG JSON is well-formed, has the expected inputs/outputs, and
    ``Graph.validate()`` passes.
  * C3.5: inference output matches ``golden/logits.npy`` under
    ``allclose(rtol=atol=1e-3)`` and (for MLP/ResNet) top-1 accuracy clears the
    gate.

Exit code 0 iff every check passes.  Skips models whose files are missing.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

_C3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3_ROOT not in sys.path:
    sys.path.insert(0, _C3_ROOT)

from scheduler.graph import import_onnx_graph
from tools import export_dag, infer

_PUBLIC = os.path.normpath(os.path.join(
    _C3_ROOT, "..", "public", "Agentic4SystemSummerSchoolContest", "Track-C", "C3-scheduler",
    "testcases", "release_to_competitors",
))
_MODELS = os.path.join(_PUBLIC, "models")
_TESTDATA = os.path.join(_PUBLIC, "testdata", "c35")

MODELS = {
    "mlp_v1": {"onnx": "mlp_v1.onnx", "in": "input", "out": "logits", "acc_gate": 0.98},
    "resnet_v1": {"onnx": "resnet_v1.onnx", "in": "input", "out": "logits", "acc_gate": 0.85},
    "transformer_v1": {"onnx": "transformer_v1.onnx", "in": "input_ids", "out": "logits", "acc_gate": None},
}

_PASS, _FAIL = 0, 0


def check(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {msg}")
    else:
        _FAIL += 1
        print(f"  FAIL  {msg}")
    return cond


def test_c31(model_key, cfg, tmp):
    onnx_path = os.path.join(_MODELS, cfg["onnx"])
    out_path = os.path.join(tmp, f"{model_key}_dag.json")
    rc = export_dag.main(["--onnx", onnx_path, "--output", out_path])
    check(rc == 0, f"[C3.1 {model_key}] export exit 0")
    with open(out_path, encoding="utf-8") as f:
        dag = json.load(f)
    check(dag.get("format_version") == "1.0", f"[C3.1 {model_key}] format_version")
    check(len(dag["nodes"]) > 0, f"[C3.1 {model_key}] nodes present ({len(dag['nodes'])})")
    check(len(dag["edges"]) > 0, f"[C3.1 {model_key}] edges present ({len(dag['edges'])})")
    in_names = [t["name"] for t in dag["graph_inputs"]]
    out_names = [t["name"] for t in dag["graph_outputs"]]
    check(cfg["in"] in in_names, f"[C3.1 {model_key}] input '{cfg['in']}' listed")
    check(cfg["out"] in out_names, f"[C3.1 {model_key}] output '{cfg['out']}' listed")
    node0 = dag["nodes"][0]
    check(all(k in node0 for k in ("name", "op_type", "inputs", "outputs")),
          f"[C3.1 {model_key}] node schema")
    # structural validity
    g = import_onnx_graph(onnx_path)
    try:
        ok = g.validate()
    except Exception as exc:
        ok = False
        print(f"        validate() raised: {exc}")
    check(ok, f"[C3.1 {model_key}] Graph.validate()")


def test_c35(model_key, cfg, tmp):
    onnx_path = os.path.join(_MODELS, cfg["onnx"])
    in_dir = os.path.join(_TESTDATA, model_key, "input")
    gold_path = os.path.join(_TESTDATA, model_key, "golden", "logits.npy")
    if not os.path.isdir(in_dir) or not os.path.exists(gold_path):
        print(f"  SKIP  [C3.5 {model_key}] testdata missing")
        return
    out_dir = os.path.join(tmp, f"{model_key}_out")
    rc = infer.main(["--onnx", onnx_path, "--input", in_dir, "--output", out_dir,
                     "--batch-size", "512"])
    check(rc == 0, f"[C3.5 {model_key}] infer exit 0")

    out = np.load(os.path.join(out_dir, f"{cfg['out']}.npy"))
    gold = np.load(gold_path)
    check(out.shape == gold.shape, f"[C3.5 {model_key}] shape {out.shape} == {gold.shape}")
    ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
    md = float(np.max(np.abs(out - gold)))
    check(ok, f"[C3.5 {model_key}] allclose(1e-3) max_abs_diff={md:.2e}")

    labels_path = os.path.join(_TESTDATA, model_key, "labels.npy")
    if cfg["acc_gate"] is not None and os.path.exists(labels_path):
        labels = np.load(labels_path)
        acc = float((out.argmax(axis=-1) == labels).mean())
        check(acc >= cfg["acc_gate"], f"[C3.5 {model_key}] top1 {acc:.4f} >= {cfg['acc_gate']}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        for key, cfg in MODELS.items():
            onnx_path = os.path.join(_MODELS, cfg["onnx"])
            if not os.path.exists(onnx_path):
                print(f"\n# {key}: model missing, skipping")
                continue
            print(f"\n# {key}")
            test_c31(key, cfg, tmp)
            test_c35(key, cfg, tmp)
    print(f"\n=== {_PASS} passed, {_FAIL} failed ===")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
