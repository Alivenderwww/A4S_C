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
import threading
import time

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


# Backend cache keyed by ONNX path, so warmup+timed tasks reuse one load.
_BACKENDS: dict = {}


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
        # Cache the backend by model path. The grader sends warmup + timed tasks
        # for the SAME model to one worker, and loading BigFormer's 19 GB of
        # weights from disk takes ~30 s. Loading once (first task) and reusing it
        # keeps that cost out of the timed runs -- the point of a persistent
        # worker. A different model just misses the cache and loads fresh.
        backend = _BACKENDS.get(onnx_path)
        if backend is None:
            backend = _make_backend(onnx_path, backend_name="cupy")
            _BACKENDS[onnx_path] = backend

        # Memory-planned chunking (see _infer_all): the chunk size is derived
        # from this backend's measured per-sample GPU footprint and the free
        # memory -- not a fixed cap and not trial-and-error. The requested
        # batch_size is only a hint; internal chunking is transparent (outputs
        # are concatenated in order). One cached backend is reused for every
        # chunk, so BigFormer's 19 GB streams once per large chunk instead of
        # once per micro-batch. Any leftover OOM shrinks the chunk on the SAME
        # backend (no recursion, no recreate) so nothing leaks.
        collected = _infer_all(backend, inputs, n)

        results = {name: np.concatenate(parts, axis=0)
                   for name, parts in collected.items()}
        _write_outputs(output_dir, backend.output_names, results)
        sys.stdout = saved_stdout
        return json.dumps({"status": "ok", "samples": n})
    except Exception as exc:
        sys.stdout = saved_stdout
        _log(f"[worker] task error: {type(exc).__name__}: {exc}")
        return json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def _measure_peak(backend, inputs, b, cp):
    """Run ONE forward at batch ``b``; return (peak_used_bytes, seconds).

    Peak is the true device footprint via ``memGetInfo`` sampling (CUDA context +
    cuBLAS workspace + live tensors) -- what the grader's NVML metric sees. The
    CuPy pool high-water (``total_bytes``) under-reports it ~4x because streaming
    frees blocks mid-forward and the pool counter never captures the simultaneous
    peak, so it must not be used for the memory model.
    """
    total = cp.cuda.runtime.memGetInfo()[1]
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    cp.cuda.Device(0).synchronize()
    st = {"min_free": cp.cuda.runtime.memGetInfo()[0], "stop": False}

    def _sample():
        cp.cuda.Device(0).use()
        while not st["stop"]:
            f = cp.cuda.runtime.memGetInfo()[0]
            if f < st["min_free"]:
                st["min_free"] = f
            time.sleep(0.001)

    th = threading.Thread(target=_sample, daemon=True)
    th.start()
    t0 = time.time()
    backend.run({k: v[:b] for k, v in inputs.items()})
    cp.cuda.Device(0).synchronize()
    dt = time.time() - t0
    st["stop"] = True
    th.join()
    pool.free_all_blocks()
    return (total - st["min_free"]), dt


def _safe_chunk(backend, inputs, n):
    """Largest #samples per forward, from DYNAMIC measurement -- no hard-coded
    footprint / memory-budget / chunk-cap constants.

    Peak GPU use is MEASURED with ``memGetInfo`` (true device footprint, matching
    the grader's NVML metric; the CuPy pool high-water under-reports it ~4x). From
    that, the chunk is chosen per model class:

    * **Streaming** (BigFormer -- weights re-stream every forward, so fewer/larger
      chunks mean less H2D and thus less time): probe two small batches, fit
      ``peak = fixed + per_sample * b``, then take the LARGEST chunk that still
      fits free memory while reserving the measured fixed cost as headroom.
      Time-first -- maximise the chunk (for N <= that, it's a single pass).
    * **Eager** (ResNet -- weights resident, so time is ~flat vs chunk and a
      bigger chunk only costs memory): grow the batch while throughput keeps
      improving AND it fits, stopping at the knee. Uses "did samples/s improve"
      as a parameter-free criterion, so it keeps peak memory low at no time cost.

    Cached on the backend; only the first (warm-up) task pays the probes. The
    iterative OOM backstop in :func:`_infer_all` covers any residual under-fit.
    """
    cached = getattr(backend, "_chunk", None)
    if cached is not None:
        return min(cached, n)
    try:
        import cupy as cp
    except Exception:
        backend._chunk = n
        return n

    streaming = getattr(getattr(backend, "rt", None), "streaming", False)
    try:
        if streaming:
            # Two small probes (kept small so each slow streamed forward is cheap;
            # the fitted per-sample/fixed are measured, not assumed).
            b1 = min(n, 16)
            b2 = min(n, 64)
            p1, _ = _measure_peak(backend, inputs, b1, cp)
            if b2 > b1:
                p2, _ = _measure_peak(backend, inputs, b2, cp)
                per = max(1.0, (p2 - p1) / (b2 - b1))       # bytes / sample
                fixed = max(0.0, p1 - per * b1)              # fixed footprint
            else:
                per = max(1.0, p1 / b1)
                fixed = 0.0
            free = cp.cuda.runtime.memGetInfo()[0]
            # Reserve the measured fixed cost as headroom (data-driven, not a
            # magic fraction): peak(chunk) = fixed + per*chunk <= free - fixed.
            chunk = int((free - 2.0 * fixed) / per)
            chunk = max(1, min(n, chunk))
            _log(f"[worker] chunk={chunk} streaming: per-sample {per/1e6:.2f}MB, "
                 f"fixed {fixed/1e9:.2f}GB, free {free/1e9:.1f}GB")
        else:
            # Eager throughput knee: double the batch while samples/s improves and
            # the measured peak still fits (reserve the smallest probe's peak, ~the
            # fixed cost, as headroom). Forwards are cheap for eager models.
            b = min(n, 32)
            peak, t = _measure_peak(backend, inputs, b, cp)
            headroom = peak
            free = cp.cuda.runtime.memGetInfo()[0]
            sps = b / max(t, 1e-6)
            best = b
            while b * 2 <= n:
                nb = b * 2
                peak, t = _measure_peak(backend, inputs, nb, cp)
                if peak > free - headroom:       # would not fit with headroom
                    break
                nsps = nb / max(t, 1e-6)
                if nsps <= sps:                  # throughput saturated: no time gain
                    break
                best, b, sps = nb, nb, nsps
            chunk = max(1, min(n, best))
            _log(f"[worker] chunk={chunk} eager: throughput-knee, free {free/1e9:.1f}GB")
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        cp.get_default_memory_pool().free_all_blocks()
        backend._chunk = min(n, 8)   # even a probe didn't fit -> stay tiny
        return backend._chunk

    backend._chunk = chunk
    return chunk


def _infer_all(backend, inputs, n):
    """Run the whole sample set through ONE reused backend in memory-planned
    chunks (see _safe_chunk). An unexpected OOM just frees the pool, halves the
    chunk, and retries the SAME range on the SAME backend -- iterative, no
    recursion and no backend recreate, so no GPU memory leaks between attempts.
    """
    try:
        import cupy as cp
        oom = (MemoryError, cp.cuda.memory.OutOfMemoryError)
        pool = cp.get_default_memory_pool()
    except Exception:
        cp = None
        oom = (MemoryError,)
        pool = None

    chunk = _safe_chunk(backend, inputs, n)
    collected = {name: [] for name in backend.output_names}
    start = 0
    while start < n:
        end = min(start + chunk, n)
        feed = {k: v[start:end] for k, v in inputs.items()}
        try:
            out = backend.run(feed)
        except oom:
            if pool is not None:
                pool.free_all_blocks()
            if chunk <= 1:
                raise  # a single sample still OOMs -- genuinely out of memory
            chunk = max(1, chunk // 2)
            backend._chunk = chunk
            _log(f"[worker] OOM at chunk={end - start}; shrink to {chunk} and retry")
            continue
        for name in backend.output_names:
            collected[name].append(np.asarray(out[name], dtype=np.float32))
        start = end
    return collected


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
