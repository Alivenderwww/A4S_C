#!/usr/bin/env python3
"""Test harness for C2 Runtime contract tests.

Reuses the grader's ``Runtime``, ``HostBuffer``, ``Dim3``, ``DeviceInfo``,
and helper constants so that contract tests never duplicate ctypes ABI bindings.

Usage (Linux/WSL with a built ``libaec.so``)::

    cd C2
    python3 -m pytest tests/test_runtime_contract.py -v

Usage (Windows — syntax/import check only)::

    cd C2
    python3 -c "import tests.runtime_harness"

All tests that require a live shared library must be gated by::

    from tests.runtime_harness import runtime, skip_if_no_library
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
import unittest
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Import grader helpers — these contain no ctypes side effects until
# ``Runtime()`` is instantiated, so they are safe to import on any platform.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_C2_ROOT = _HERE.parent
sys.path.insert(0, str(_C2_ROOT))

from grader.public_grade import (  # noqa: E402
    Runtime,
    HostBuffer,
    Dim3,
    DeviceInfo,
    DeviceCompletion,
    DeviceKernelInfo,
    RuntimeStats,
    VectorAddArgs,
    # Error codes
    SUCCESS,
    INVALID_ARGUMENT,
    OUT_OF_MEMORY,
    INVALID_HANDLE,
    INVALID_ADDRESS,
    NOT_READY,
    NOT_SUPPORTED,
    DEVICE_ERROR,
    ISA_TRAP,
    # Copy direction
    H2D,
    D2H,
    # Data types
    DTYPE_FP4,
    DTYPE_FP8_E4M3,
    DTYPE_FP8_E5M2,
    DTYPE_FP16,
    DTYPE_BF16,
    DTYPE_FP32,
    DTYPE_FP64,
    DTYPE_INT4,
    DTYPE_INT8,
    DTYPE_INT32,
    # Kernel IDs
    KERNEL_VECTOR_ADD,
    KERNEL_GEMM_NAIVE,
    KERNEL_GEMM_TILED,
    KERNEL_GEMM_VECTORIZED,
    # Float encoding helpers (for NaN probe etc.)
    _encode_float_values,
    _decode_float_values,
    _mini_encode,
    _mini_decode,
    _pack_nibbles,
    _pack_integers,
    _float_gemm_oracle,
    _integer_gemm_oracle,
    _f32,
    # Registration
    require as grader_require,
)

# Re-export so test modules need only ``from tests.runtime_harness import ...``.
__all__ = [
    "Runtime", "HostBuffer", "Dim3", "DeviceInfo",
    "DeviceCompletion", "DeviceKernelInfo", "RuntimeStats", "VectorAddArgs",
    "SUCCESS", "INVALID_ARGUMENT", "OUT_OF_MEMORY", "INVALID_HANDLE",
    "INVALID_ADDRESS", "NOT_READY", "NOT_SUPPORTED", "DEVICE_ERROR", "ISA_TRAP",
    "H2D", "D2H",
    "DTYPE_FP4", "DTYPE_FP8_E4M3", "DTYPE_FP8_E5M2", "DTYPE_FP16",
    "DTYPE_BF16", "DTYPE_FP32", "DTYPE_FP64", "DTYPE_INT4", "DTYPE_INT8",
    "DTYPE_INT32",
    "KERNEL_VECTOR_ADD", "KERNEL_GEMM_NAIVE", "KERNEL_GEMM_TILED",
    "KERNEL_GEMM_VECTORIZED",
    "_encode_float_values", "_decode_float_values",
    "_mini_encode", "_mini_decode", "_pack_nibbles", "_pack_integers",
    "_float_gemm_oracle", "_integer_gemm_oracle", "_f32",
    "grader_require", "require",
    "runtime", "skip_if_no_library",
    "LIBRARY_PATH", "HERE", "C2_ROOT",
    "isolated_subprocess",
    "_parse_isolated_output",
    "reset_runtime",
    "ERROR_NAMES",
    # Shim support
    "shim_library_path", "skip_if_no_shim", "compile_shim",
    "is_linux_with_gxx",
]

# Error name map for readable assertions.
ERROR_NAMES = {
    SUCCESS: "AEC_SUCCESS",
    INVALID_ARGUMENT: "AEC_ERROR_INVALID_ARGUMENT",
    OUT_OF_MEMORY: "AEC_ERROR_OUT_OF_MEMORY",
    INVALID_HANDLE: "AEC_ERROR_INVALID_HANDLE",
    INVALID_ADDRESS: "AEC_ERROR_INVALID_ADDRESS",
    NOT_READY: "AEC_ERROR_NOT_READY",
    NOT_SUPPORTED: "AEC_ERROR_NOT_SUPPORTED",
    DEVICE_ERROR: "AEC_ERROR_DEVICE",
    ISA_TRAP: "AEC_ERROR_ISA_TRAP",
}


def require(condition: object, detail: object = "contract assertion failed") -> None:
    """Checked assertion (always enabled, even under ``python -O``)."""
    if not condition:
        raise AssertionError(str(detail))


# ---------------------------------------------------------------------------
# Skip exception (usable outside pytest)
# ---------------------------------------------------------------------------

class SkipTest(unittest.SkipTest):
    """Raised by tests that should be skipped (not failed).

    Inherits from ``unittest.SkipTest`` so that pytest recognises it natively
    (no conftest conversion needed).  The standalone runner
    (``run_hidden_style.py``) catches this and reports ``SKIP`` instead of
    ``FAIL``.
    """


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------
HERE = _HERE
C2_ROOT = _C2_ROOT
LIBRARY_PATH = _C2_ROOT / "libaec.so"

# ---------------------------------------------------------------------------
# Singleton runtime (lazy — only created when first accessed).
# ---------------------------------------------------------------------------
_runtime: Runtime | None = None
_runtime_load_error: str | None = None


def _create_runtime() -> Runtime:
    global _runtime, _runtime_load_error
    if _runtime is not None:
        return _runtime
    if _runtime_load_error is not None:
        raise RuntimeError(_runtime_load_error)
    if not LIBRARY_PATH.is_file():
        _runtime_load_error = (
            f"libaec.so not found at {LIBRARY_PATH}. "
            "Build it on Linux/WSL first (make -j2)."
        )
        raise RuntimeError(_runtime_load_error)
    try:
        _runtime = Runtime(LIBRARY_PATH)
    except Exception as exc:
        _runtime_load_error = (
            f"Failed to load Runtime from {LIBRARY_PATH}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        raise RuntimeError(_runtime_load_error) from exc
    return _runtime


def runtime() -> Runtime:
    """Return the singleton :class:`Runtime` instance.

    Raises :class:`RuntimeError` with a descriptive message if the library
    cannot be loaded (wrong platform, missing build, missing device lib).
    """
    return _create_runtime()


def skip_if_no_library() -> None:
    """Call early in a test to skip when ``libaec.so`` is unavailable.

    Always raises :class:`SkipTest` (never ``pytest.skip()`` directly).
    ``SkipTest`` inherits from ``unittest.SkipTest`` so pytest handles it
    natively without any conftest conversion.
    """
    if not LIBRARY_PATH.is_file():
        _skip_or_raise(f"no {LIBRARY_PATH} — build on Linux/WSL first")
    try:
        _create_runtime()
    except RuntimeError as exc:
        _skip_or_raise(str(exc))


def reset_runtime() -> None:
    """Reset the runtime singleton for test isolation.

    Delegates to the official ``Runtime.reset()`` which checks the
    ``aecDeviceReset`` return value (raises on non-zero) and then clears
    the last error via ``aecGetLastError``.  Safe to call whether or not
    the library has been loaded.

    Raises on failure — the caller (runner or pytest fixture) is expected to
    mark the current test as FAIL and skip execution.
    """
    if _runtime is None:
        return
    _runtime.reset()


def _skip_or_raise(msg: str) -> None:
    """Always raise :class:`SkipTest`.

    ``SkipTest`` inherits from ``unittest.SkipTest`` so pytest handles it
    natively; the standalone runner catches it and reports ``SKIP``.
    """
    raise SkipTest(msg) from None


# ---------------------------------------------------------------------------
# Shim compilation (LD_PRELOAD blocker for deterministic concurrency tests)
# ---------------------------------------------------------------------------
_SHIM_CPP = HERE / "blocking_submit_shim.cpp"
_SHIM_SO = HERE / "blocking_submit_shim.so"


def is_linux_with_gxx() -> bool:
    """Check whether the current platform is Linux and ``g++`` is available."""
    if sys.platform != "linux":
        return False
    try:
        subprocess.run(["g++", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def compile_shim() -> Path | None:
    """Compile ``blocking_submit_shim.cpp`` into ``blocking_submit_shim.so``.

    Returns the ``.so`` path on success, ``None`` if the platform or compiler
    is unavailable (caller should ``skip_if_no_shim()``).
    """
    if not is_linux_with_gxx():
        return None
    if _SHIM_SO.is_file():
        return _SHIM_SO  # already built
    try:
        proc = subprocess.run(
            [
                "g++", "-shared", "-fPIC",
                "-o", str(_SHIM_SO),
                str(_SHIM_CPP),
                "-ldl", "-pthread",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            _log_shim_fail(proc.stdout, proc.stderr)
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None
    return _SHIM_SO if _SHIM_SO.is_file() else None


def _log_shim_fail(stdout: str, stderr: str) -> None:
    """Log compilation failure to stderr (avoids polluting PASS/FAIL protocol)."""
    import sys
    print(f"  NOTE  shim compilation failed: returncode!=0", file=sys.stderr)
    if stdout.strip():
        for line in stdout.strip().splitlines():
            print(f"  NOTE  shim stdout: {line}", file=sys.stderr)
    if stderr.strip():
        for line in stderr.strip().splitlines():
            print(f"  NOTE  shim stderr: {line}", file=sys.stderr)


def shim_library_path() -> Path:
    """Return the expected shim ``.so`` path (may not exist; call ``compile_shim()`` first)."""
    return _SHIM_SO


def skip_if_no_shim() -> None:
    """Skip the test when the blocking-submit shim cannot be compiled (non-Linux / no g++).

    Always raises :class:`SkipTest` (never ``pytest.skip()`` directly).
    """
    so = compile_shim()
    if so is None:
        raise SkipTest(
            "blocking_submit_shim.so unavailable — "
            "requires Linux + g++.  SKIP (not FAIL)."
        )


# ---------------------------------------------------------------------------
# Subprocess isolation for dangerous-pointer tests
# ---------------------------------------------------------------------------


def _parse_isolated_output(stdout: str, stderr: str, returncode: int) -> str:
    """Parse the protocol line from an isolated subprocess invocation.

    The last non-empty stdout line must be ``PASS`` or ``FAIL:<reason>``.
    Everything before it is diagnostic (e.g. ``NOTE`` lines) and must NOT
    cause a false failure.
    """
    lines = [l for l in stdout.splitlines() if l.strip()]
    last_line = lines[-1].strip() if lines else ""
    if returncode == 0 and last_line == "PASS":
        return "PASS"
    if last_line.startswith("FAIL:"):
        return last_line
    if returncode != 0:
        detail = (stdout.strip()[:100] + " | " + stderr.strip()[:100]).strip()[:200]
        return f"FAIL:exit={returncode}: {detail}"
    return f"FAIL:unexpected output: {stdout.strip()[:200]}"


def isolated_subprocess(
    func: Callable[..., Any],
    *args: Any,
    timeout: float = 10.0,
    ld_preload: str | Path | None = None,
    **kwargs: Any,
) -> str:
    """Run *func* in a subprocess so a crash/abort cannot take down the suite.

    The function must be importable from a module under ``C2/tests/``.

    When *ld_preload* is given (path to a ``.so``), ``LD_PRELOAD`` is set in
    the subprocess environment so the library intercepts ``aecDeviceSubmit``.

    Returns ``"PASS"`` or ``"FAIL:<reason>"``.

    Usage::

        from tests.runtime_harness import isolated_subprocess

        def _dangerous_free():
            rt = runtime()
            rt.lib.aecFree(0xDEADBEEF)

        result = isolated_subprocess(_dangerous_free)
        assert result == "FAIL:...", f"expected failure, got {result}"
    """
    # Locate the function module
    mod = sys.modules.get(func.__module__)
    if mod is None:
        return f"FAIL:cannot locate module {func.__module__}"

    # Build a module path relative to tests/
    mod_file = getattr(mod, "__file__", None)
    if mod_file is None:
        # The function was defined in an interactive/__main__ context;
        # fall back to the module name directly.
        module_dot_path = func.__module__
    else:
        rel = Path(mod_file).resolve().relative_to(HERE)
        module_dot_path = str(rel.with_suffix("")).replace(os.sep, ".")

    import json
    payload = json.dumps({"args": args, "kwargs": kwargs})
    # Flatten to positional for the entry point
    flat_args = json.dumps(list(args)) if args else "[]"
    # Actually pass the function name and the JSON of args
    # Use the entry point protocol: "<module_path>.<func_name>" and JSON args
    # We need to also pass kwargs properly
    # Let's use a simpler approach: pickle-like via JSON
    func_ref = f"{module_dot_path}.{func.__name__}"

    # Build LD_PRELOAD env addition if requested
    ld_preload_env = {}
    if ld_preload is not None:
        ld_preload_env["LD_PRELOAD"] = str(ld_preload)

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(HERE / "runtime_harness.py"),
                "--subprocess",
                func_ref,
                json.dumps(list(args)),
                json.dumps(kwargs),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ,
                 "AEC_DEVICE_LIBRARY": os.environ.get("AEC_DEVICE_LIBRARY", ""),
                 "PYTHONHASHSEED": "0",
                 "PYTHONPATH": str(C2_ROOT),
                 **(ld_preload_env),}
        )
    except subprocess.TimeoutExpired:
        return "FAIL:subprocess timed out"
    except FileNotFoundError as exc:
        return f"FAIL:cannot launch subprocess: {exc}"

    stdout = proc.stdout        # keep raw for diagnostic logging
    stderr = proc.stderr.strip()

    return _parse_isolated_output(stdout, stderr, proc.returncode)


# ---------------------------------------------------------------------------
# Self-test for protocol-line parsing logic
# ---------------------------------------------------------------------------

def _test_parse_isolated_output() -> None:
    """Self-test for ``_parse_isolated_output`` (no subprocess needed)."""
    cases: list[tuple[tuple[str, str, int], str]] = [
        # (stdout, stderr, returncode) -> expected
        (("PASS", "", 0), "PASS"),
        (("NOTE x\nPASS", "", 0), "PASS"),
        (("NOTE a\nNOTE b\nPASS", "", 0), "PASS"),
        (("PASS\nNOTE after\nPASS", "", 0), "PASS"),
        (("  PASS  ", "", 0), "PASS"),
        (("NOTE x\n  PASS  ", "", 0), "PASS"),
        (("FAIL:something broke", "", 1), "FAIL:something broke"),
        (("NOTE x\nFAIL:invalid argument", "", 1), "FAIL:invalid argument"),
        (("NOTE x\n  FAIL:spaces  ", "", 1), "FAIL:spaces"),
        (("NOTE x", "", 0), "FAIL:unexpected output: NOTE x"),
        (("", "", 0), "FAIL:unexpected output: "),
        (("some stdout", "segfault", -11), "FAIL:exit=-11: some stdout | segfault"),
    ]
    for (stdout, stderr, rc), expected in cases:
        result = _parse_isolated_output(stdout, stderr, rc)
        if result != expected:
            msg = (f"FAIL: _parse_isolated_output({stdout=!r}, {stderr=!r}, {rc=})\n"
                   f"       got {result!r}, expected {expected!r}")
            print(msg)
            sys.exit(1)
    print("PASS")


# ---------------------------------------------------------------------------
# Self-test for reset_runtime error propagation
# ---------------------------------------------------------------------------

def _test_reset_runtime_rejects_nonzero() -> None:
    """Verify that ``reset_runtime()`` raises when the device reset returns
    a non-zero status code (e.g. 7).  Pure-Python test — no library needed.
    """
    # Mocks a Runtime whose device reset returns 7 (failure)
    class _MockDevice:
        def aecDeviceReset(self) -> int:
            return 7

    class _MockRuntime:
        def __init__(self) -> None:
            self.device = _MockDevice()
        def reset(self) -> None:
            if self.device.aecDeviceReset() != 0:
                raise AssertionError("reference device reset failed")

    global _runtime
    saved = _runtime
    try:
        _runtime = _MockRuntime()  # type: ignore[assignment]
        reset_runtime()
        print("FAIL: reset_runtime should have raised AssertionError")
        sys.exit(1)
    except AssertionError:
        print("PASS")
    finally:
        _runtime = saved


# ---------------------------------------------------------------------------
# CLI entry point for subprocess isolation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        _test_parse_isolated_output()
        _test_reset_runtime_rejects_nonzero()
        sys.exit(0)
    elif len(sys.argv) >= 4 and sys.argv[1] == "--subprocess":
        func_ref = sys.argv[2]
        args_json = sys.argv[3]
        kwargs_json = sys.argv[4] if len(sys.argv) > 4 else "{}"
        import json
        try:
            parts = func_ref.split(".")
            # The module is something like "tests.test_runtime_contract"
            module_name = ".".join(parts[:-1])
            func_name = parts[-1]
            import importlib
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            args = json.loads(args_json) if args_json.strip() else []
            kwargs = json.loads(kwargs_json) if kwargs_json.strip() else {}
            func(*args, **kwargs)
            print("PASS")
        except Exception as exc:
            msg = str(exc).replace("\n", " | ")
            print(f"FAIL:{msg}")
            sys.exit(1)
    else:
        print("runtime_harness.py -- subprocess entry point for isolation tests",
              file=sys.stderr)
        sys.exit(1)
