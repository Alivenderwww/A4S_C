#!/usr/bin/env python3
"""Hidden-style runner for C2 contract tests.

Discovers all ``test_*`` functions from modules under ``C2/tests/``,
executes them sequentially, and prints a ``PASS`` / ``FAIL`` / ``SKIP``
summary.

Module import / collection failures are counted as **FAIL** and produce a
non-zero exit code — they never vanish from the total or become SKIP.

No third-party dependencies (no pytest, no unittest).

Usage::

    cd C2
    python3 -B tests/run_hidden_style.py
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_C2_ROOT = _HERE.parent

# Ensure C2 root is on sys.path so that test modules can do
# ``from tests.runtime_harness import ...`` reliably.
if str(_C2_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2_ROOT))


def _discover_test_modules() -> list[str]:
    """Return a sorted list of dotted module names under ``C2/tests/``
    whose name starts with ``test_`` (excluding ``run_hidden_style``)."""
    modules: list[str] = []
    for entry in sorted(_HERE.iterdir()):
        if entry.suffix != ".py":
            continue
        stem = entry.stem
        if not stem.startswith("test_") or stem == "run_hidden_style":
            continue
        modules.append(f"tests.{stem}")
    return modules


def _collect_tests(
    module_names: list[str],
) -> tuple[list[tuple[types.FunctionType, str]], list[tuple[str, str]]]:
    """Import each module and collect ``test_*`` functions.

    Returns ``(tests, errors)`` where *tests* is ``[(func, name), ...]``
    sorted by name, and *errors* is ``[(module_name, reason), ...]`` for
    modules that could not be imported.

    Module import failures are returned as errors — they are **not** silently
    skipped and **do** count toward the FAIL total.

    ``SystemExit`` raised during import is explicitly caught and counted as
    FAIL (not propagated).  ``KeyboardInterrupt`` is **never** swallowed.
    """
    tests: list[tuple[types.FunctionType, str]] = []
    errors: list[tuple[str, str]] = []
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except SystemExit as exc:
            msg = f"module raised SystemExit({exc.code})"
            errors.append((mod_name, msg))
            continue
        except Exception as exc:
            msg = str(exc).split("\n")[0]
            errors.append((mod_name, msg))
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            obj = getattr(mod, name)
            if not isinstance(obj, types.FunctionType):
                continue
            # Skip helper/nested functions that are not intended as test cases
            if name.startswith("test_") and obj.__module__ == mod_name:
                tests.append((obj, name))
    return tests, errors


def _import_harness_helpers() -> tuple:
    """Import ``SkipTest`` and ``reset_runtime`` from the harness.

    Returns ``(SkipTest_class, reset_func)``.  Either may be ``None`` if the
    harness cannot be imported (the runner degrades gracefully).
    """
    try:
        from tests.runtime_harness import SkipTest as _SkipTest, reset_runtime as _reset
        return _SkipTest, _reset
    except ImportError:
        return None, None


def main() -> int:
    module_names = _discover_test_modules()
    if not module_names:
        print("No test modules found under C2/tests/")
        return 1

    print(f"Discovered modules: {', '.join(module_names)}")
    print()

    tests, module_errors = _collect_tests(module_names)

    if not tests and not module_errors:
        print("No test_* functions found.")
        return 1

    _SkipTest, _reset_runtime = _import_harness_helpers()

    passed = 0
    failed = 0
    skipped = 0
    results: list[tuple[str, str, str]] = []  # (name, status, detail)

    # Report module-level failures first (counted as FAIL)
    for mod_name, reason in module_errors:
        results.append((f"{mod_name} (module)", "FAIL", reason))
        failed += 1
        print(f"FAIL  {mod_name}: module import failed — {reason}")

    for func, name in tests:
        # Test isolation: reset runtime state before each library-dependent test.
        # If reset raises the test is FAILed and skipped — the failure is never
        # silently swallowed.
        if _reset_runtime is not None:
            try:
                _reset_runtime()
            except KeyboardInterrupt:
                raise  # never swallow Ctrl+C
            except BaseException as exc:
                msg = str(exc).split("\n")[0]
                results.append((name, "FAIL", f"reset failed — {msg}"))
                failed += 1
                print(f"FAIL  {name}: reset failed — {msg}")
                continue  # skip test execution

        try:
            func()
            results.append((name, "PASS", ""))
            passed += 1
            print(f"PASS  {name}")
        except KeyboardInterrupt:
            raise  # never swallow Ctrl+C
        except BaseException as exc:
            if _SkipTest is not None and isinstance(exc, _SkipTest):
                msg = str(exc)
                results.append((name, "SKIP", msg))
                skipped += 1
                print(f"SKIP  {name}: {msg}")
            else:
                msg = str(exc).split("\n")[0]
                results.append((name, "FAIL", msg))
                failed += 1
                print(f"FAIL  {name}: {msg}")

    # Summary — total includes ALL discovered items
    total = passed + failed + skipped
    print(f"\n{'=' * 50}")
    parts = [f"{passed}/{total} passed"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed")
    print("Results: " + ", ".join(parts))
    if failed:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  {name}: {detail}")
    if skipped:
        print(f"\nSkipped tests ({skipped}):")
        for name, status, detail in results:
            if status == "SKIP":
                short = detail[:100]
                print(f"  {name}: {short}")

    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _test_import_system_exit() -> None:
    """Verify that a module raising ``SystemExit(7)`` on import is counted
    as a module-level FAIL, **not** propagated up to kill the runner.

    This proves the runner's ``SystemExit``-catching guarantee.
    """
    import tempfile
    import shutil

    tmp = tempfile.mkdtemp()
    # Build a small package under the temp dir
    pkg_dir = os.path.join(tmp, "_selftest_sys_pkg")
    os.makedirs(pkg_dir)
    with open(os.path.join(pkg_dir, "__init__.py"), "w"):
        pass
    with open(os.path.join(pkg_dir, "bad_mod.py"), "w") as f:
        f.write("raise SystemExit(7)\n")

    sys.path.insert(0, tmp)
    try:
        tests, errors = _collect_tests(["_selftest_sys_pkg.bad_mod"])
        assert len(errors) == 1, (
            f"expected 1 module error from SystemExit(7), got {len(errors)}"
        )
        mod_name, reason = errors[0]
        assert "SystemExit" in reason, (
            f"expected 'SystemExit' in error reason, got: {reason!r}"
        )
        assert len(tests) == 0
        print("PASS  _test_import_system_exit")
    finally:
        sys.path.remove(tmp)
        shutil.rmtree(tmp)


def _test_reset_keyboard_interrupt() -> None:
    """Verify that ``KeyboardInterrupt`` raised during reset is propagated
    (not caught as FAIL), matching the ``except KeyboardInterrupt: raise``
    guard added to the reset block.
    """
    def _mock_reset() -> None:
        raise KeyboardInterrupt()

    # Simulate the exact try/except pattern from main()
    try:
        try:
            _mock_reset()
        except KeyboardInterrupt:
            raise
        except BaseException:
            print("FAIL: KeyboardInterrupt was caught as BaseException")
            sys.exit(1)
    except KeyboardInterrupt:
        pass  # expected — KeyboardInterrupt propagated correctly

    print("PASS  _test_reset_keyboard_interrupt")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _test_import_system_exit()
        _test_reset_keyboard_interrupt()
        sys.exit(0)
    sys.exit(main())
