#!/usr/bin/env python3
"""C3.5 persistent worker — stdin/stdout JSON task protocol.

Spec ref: ``C35_WORKER_PROTOCOL.md``. The grader starts this process once
(without task arguments), then feeds it one task per line on stdin::

    {"onnx": "<path>", "input": "<dir>", "output": "<dir>", "batch_size": 256}

For each task the worker loads the model, runs inference, writes the output
files, and replies with exactly one line on stdout::

    {"status": "ok", "samples": 10000}

On failure it replies ``{"status": "error", "error": "..."}``. The grader ends
the session with ``{"cmd": "exit"}``, after which the worker exits 0.

Protocol discipline (critical for grading):
  * stdout carries ONLY protocol signals: one ``READY`` line up front, then one
    result-JSON line per task. Any logs, warnings, or CuPy banner output MUST
    go to stderr — the grader parses stdout line-by-line.
  * The result line is printed only AFTER the output files are fully written,
    because the grader treats reading the result line as the timing end.
  * Initialisation (importing CuPy, warming the CUDA context) happens before
    ``READY`` and is excluded from the per-task timing window.

The inference core reuses :mod:`tools.infer` (CuPy GPU backend, batch loop,
output manifest) so numerics are identical to the one-shot ``infer.py``.
"""

from __future__ import annotations

import json
import os
import sys

# Make the C3 package importable no matter the caller's cwd.
_C3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3_ROOT not in sys.path:
    sys.path.insert(0, _C3_ROOT)

import numpy as np  # noqa: E402

# The protocol pipe the grader reads. ``_PROTOCOL_OUT`` is the real stdout the
# grader parses; any stray ``print()`` from backend code is diverted to stderr
# so stdout stays protocol-clean (READY / result lines only).
_PROTOCOL_OUT = sys.stdout


def _emit(line: str) -> None:
    """Write one protocol line to the real stdout (grader reads this)."""
    _PROTOCOL_OUT.write(line + "\n")
    _PROTOCOL_OUT.flush()


# Reuse the battle-tested inference path from infer.py (CuPy backend, batching,
# output writing) — same numerics, just driven by the worker loop instead of CLI.
# Imported AFTER _PROTOCOL_OUT is captured.
from tools.infer import _load_inputs, _make_backend, _ORT_TYPE_TO_NP  # noqa: E402


# ---------------------------------------------------------------------------
def _log(*args):
    """All diagnostics go to stderr; stdout is reserved for protocol signals."""
    print(*args, file=sys.stderr, flush=True)


def _write_outputs(output_dir, output_names, results):
    """Write manifest.json + <name>.npy per output tensor (same format as infer.py)."""
    os.makedirs(output_dir, exist_ok=True)
    out_tensors = []
    for name, arr in results.items():
        fname = f"{name}.npy"
        np.save(os.path.join(output_dir, fname), arr.astype(np.float32))
        out_tensors.append(
            {"name": name, "file": fname, "dtype": "float32", "shape": list(arr.shape)}
        )
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"tensors": out_tensors}, f, indent=2, ensure_ascii=False)


def _run_one_task(task: dict) -> str:
    """Execute one task dict; return the result-JSON line (stdout-bound).

    While the backend runs, ``sys.stdout`` is redirected to stderr so any stray
    ``print()`` inside the inference path (CuPy banner, ``[infer] backend:``
    log) cannot corrupt the protocol channel. The real stdout is restored
    before we emit the result line.
    """
    onnx_path = task["onnx"]
    input_dir = task["input"]
    output_dir = task["output"]
    batch_size = int(task.get("batch_size", 256) or 256)

    saved_stdout = sys.stdout
    sys.stdout = sys.stderr  # divert any backend print() to stderr
    try:
        inputs, n = _load_inputs(input_dir)
        # Backend is created per task: the protocol times "load model + infer",
        # and each task may use a different model (grader reuses the worker
        # across models by restarting it, but defensively we support either).
        backend = _make_backend(onnx_path, backend_name="cupy")

        # Weight-streaming models (e.g. BigFormer 19GB > 17GB GPU): respect the
        # requested batch size — the pool's freed weight blocks are not always
        # reclaimed fast enough for larger batches to fit, so a conservative
        # batch keeps memory bounded while still producing correct results.
        streaming = getattr(getattr(backend, "rt", None), "streaming", False)
        bs = max(1, batch_size)
        collected = {name: [] for name in backend.output_names}
        for start in range(0, n, bs):
            end = min(start + bs, n)
            feed = {name: arr[start:end] for name, arr in inputs.items()}
            out = backend.run(feed)
            for name in backend.output_names:
                collected[name].append(np.asarray(out[name], dtype=np.float32))

        results = {name: np.concatenate(parts, axis=0)
                   for name, parts in collected.items()}
        # Files must be on disk BEFORE the result line is emitted (timing end).
        _write_outputs(output_dir, backend.output_names, results)
        sys.stdout = saved_stdout
        return json.dumps({"status": "ok", "samples": n})
    except Exception as exc:
        sys.stdout = saved_stdout
        _log(f"[worker] task error: {type(exc).__name__}: {exc}")
        return json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    # ---- one-time init (excluded from per-task timing) ----
    # Divert stdout→stderr during init so CuPy/import banners stay off the
    # protocol channel; only the READY line goes to the real stdout.
    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        import cupy as _cp  # noqa: F401  — force the import / CUDA init now
        _cp.cuda.Device(0).compute_capability  # touch device to init context
        _log("[worker] CuPy initialised")
    except Exception as exc:
        _log(f"[worker] CuPy init warning: {exc} (will retry per task)")
    sys.stdout = saved_stdout

    # Signal readiness — exactly one line on the real stdout.
    _emit("READY")

    # ---- task loop ----
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError:
            _log(f"[worker] malformed task line: {line[:80]}")
            _emit(json.dumps({"status": "error", "error": "malformed JSON"}))
            continue

        if task.get("cmd") == "exit":
            _log("[worker] exit received, shutting down")
            return 0

        _emit(_run_one_task(task))

    # stdin closed without an explicit exit — treat as clean shutdown.
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        _log(f"[worker] fatal: {type(exc).__name__}: {exc}")
        sys.exit(1)
