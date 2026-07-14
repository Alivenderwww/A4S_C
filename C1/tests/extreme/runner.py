"""Extreme-test runner: discovery, execution, comparison, and result classification."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tests.extreme.cases import ExtremeCase, OutputExpectation, load_case_matrix
from tests.extreme.backends import CompileResult, ExecutionResult


# ════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing actual simulator output against expected values."""

    matched: bool
    first_mismatch_index: Optional[int] = None
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    max_abs_diff: Optional[float] = None


@dataclass(frozen=True)
class Classification:
    """Single-case classification verdict."""

    verdict: str  # PASS, FAIL, XFAIL, XPASS


# ════════════════════════════════════════════════════════════════════════
# Case discovery
# ════════════════════════════════════════════════════════════════════════


def discover_cases(root: Path, suite: str) -> List[ExtremeCase]:
    """Discover extreme-test cases for *suite*, failing if none are found."""
    cases = load_case_matrix(root, suite)
    if not cases:
        raise RuntimeError("zero extreme cases discovered for suite %s" % suite)
    return cases


# ════════════════════════════════════════════════════════════════════════
# Opt-level filtering
# ════════════════════════════════════════════════════════════════════════


def filter_opt_levels(cases: List[ExtremeCase], opt: str) -> List[ExtremeCase]:
    """Filter cases to keep only *opt*.

    When *opt* is ``"all"``, every case's original ``opt_levels`` tuple
    is preserved.  Otherwise, only cases that include *opt* are kept, and
    each is replaced with ``opt_levels=(opt,)``.
    """
    if opt == "all":
        return list(cases)

    result: List[ExtremeCase] = []
    for case in cases:
        if opt in case.opt_levels:
            result.append(replace(case, opt_levels=(opt,)))
    return result


# ════════════════════════════════════════════════════════════════════════
# Output comparison
# ════════════════════════════════════════════════════════════════════════


def _np_dtype(dtype_str: str) -> np.dtype:
    """Convert a dtype string (e.g. ``'<u4'``, ``'<f4'``) to a numpy dtype."""
    return np.dtype(dtype_str)


def compare_output(actual: bytes, exp: OutputExpectation) -> ComparisonResult:
    """Compare actual simulator output bytes against an *OutputExpectation*.

    Checks, in order:
    - actual byte length sufficient for shape × element size;
    - NaN / Inf in either expected or actual → mismatch;
    - tolerance comparison using ``exp.rtol`` / ``exp.atol``;
    - exact integer comparison when tolerances are zero.

    Returns a :class:`ComparisonResult` with mismatch metadata on failure.
    """
    dt = _np_dtype(exp.dtype)
    elem_size = dt.itemsize
    expected_numel = int(np.prod(exp.shape))
    expected_bytes = expected_numel * elem_size

    # Exact-length check.
    if len(actual) != expected_bytes:
        return ComparisonResult(matched=False)

    arr_actual = np.frombuffer(actual[:expected_bytes], dtype=dt).reshape(exp.shape)
    arr_expected = np.array(exp.expected, dtype=dt).reshape(exp.shape)

    # Finite-value checks (NaN / Inf in either side → mismatch).
    if not np.all(np.isfinite(arr_expected)):
        return ComparisonResult(matched=False)
    if not np.all(np.isfinite(arr_actual)):
        return ComparisonResult(matched=False)

    # Elementwise comparison with tolerances.
    flat_actual = arr_actual.ravel()
    flat_expected = arr_expected.ravel()

    # For integer types, compare exactly.
    if np.issubdtype(dt, np.integer):
        diff = flat_actual != flat_expected
        if np.any(diff):
            idx = int(np.argmax(diff))
            return ComparisonResult(
                matched=False,
                first_mismatch_index=idx,
                expected_value=float(flat_expected[idx]),
                actual_value=float(flat_actual[idx]),
                max_abs_diff=float(
                    np.max(np.abs(
                        flat_actual.astype(np.float64)
                        - flat_expected.astype(np.float64)
                    ))
                ),
            )
        return ComparisonResult(matched=True)

    # Float comparison with tolerances.
    abs_diff = np.abs(flat_actual.astype(np.float64) - flat_expected.astype(np.float64))
    tol = exp.atol + exp.rtol * np.abs(flat_expected.astype(np.float64))
    out_of_tol = abs_diff > tol
    if np.any(out_of_tol):
        idx = int(np.argmax(out_of_tol))
        return ComparisonResult(
            matched=False,
            first_mismatch_index=idx,
            expected_value=float(flat_expected[idx]),
            actual_value=float(flat_actual[idx]),
            max_abs_diff=float(np.max(abs_diff)),
        )
    return ComparisonResult(matched=True)


# ════════════════════════════════════════════════════════════════════════
# Classification
# ════════════════════════════════════════════════════════════════════════


def classify_case(
    suite: str,
    case: ExtremeCase,
    comp: ComparisonResult,
    observed_phase: str = "compare",
    observed_detail: str = "",
) -> Classification:
    """Classify a single case result into PASS / FAIL / XFAIL / XPASS.

    Contract:
        matched → PASS;  mismatched → FAIL.

    Frontier:
        If ``expected_failure`` is None (no resolved backend expectation):
            matched → PASS;  mismatched → FAIL.
        If an expected failure is set:
            matched → XPASS (regardless of observed phase).
            mismatched and ``observed_phase == expected_failure_phase`` → XFAIL.
            mismatched and ``observed_phase != expected_failure_phase`` → FAIL.

    *observed_phase* is compared against the case's ``expected_failure_phase``
    to determine whether a failure matches the registered expectation.
    """
    matched = comp.matched

    if suite == "contract":
        return Classification(verdict="PASS" if matched else "FAIL")

    if suite == "frontier":
        ef = case.expected_failure
        ef_phase = case.expected_failure_phase
        if ef is not None:
            # Backend-specific expectation resolved.
            if matched:
                return Classification(verdict="XPASS")
            # Mismatched: check phase.
            if observed_phase == ef_phase:
                return Classification(verdict="XFAIL")
            return Classification(verdict="FAIL")
        else:
            # No backend expectation for this case.
            return Classification(verdict="PASS" if matched else "FAIL")

    return Classification(verdict="FAIL")


# ════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════


def summarize_results(classifications: List[Classification]) -> int:
    """Return exit code for a list of classification results.

    Returns ``0`` if all are PASS or XFAIL only; ``1`` if any FAIL or XPASS.
    """
    for c in classifications:
        if c.verdict in ("FAIL", "XPASS"):
            return 1
    return 0


# ════════════════════════════════════════════════════════════════════════
# Artifact writer
# ════════════════════════════════════════════════════════════════════════


def _sanitize_path_component(name: str) -> str:
    """Replace characters that are unsafe on common filesystems."""
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _find_disassembly_tool(name: str, bin_dir: Path) -> List[str]:
    """Yield candidate paths for a disassembly tool.

    Checks *bin_dir* first (project-local build artifact), then falls
    back to a bare-name ``shutil.which`` lookup so that tools on PATH
    are also found.  Returns a list of candidate strings (empty when
    nothing is resolvable).
    """
    candidates: List[str] = []
    local = bin_dir / name
    if local.is_file():
        candidates.append(str(local))
    resolved = shutil.which(name)
    if resolved is not None and resolved not in candidates:
        candidates.append(resolved)
    # Always include the bare name as a last-resort candidate in case
    # the subprocess environment has PATH set differently.
    if not candidates or candidates[-1] != name:
        candidates.append(name)
    return candidates


class ArtifactWriter:
    """Writes per-case artifact files to a structured directory tree.

    Layout::

        {root}/{suite}/{sanitized_case_name}/{opt}/{backend}/
            case.ptx
            compile.stdout.txt
            compile.stderr.txt
            compile_report.json
            program.aecbin
            program.asm
            result.json
            expected.bin
            actual.bin
    """

    def __init__(self, root: Path, suite: str) -> None:
        self._root = root
        self._suite = suite

    def write_all(
        self,
        cases: List[ExtremeCase],
        compile_results: List[CompileResult],
        exec_results: List[ExecutionResult],
        comparisons: List[ComparisonResult],
        classifications: List[Classification],
        opt: str = "O2",
        backend: str = "local",
    ) -> List[Path]:
        """Write artifacts for a batch of results.

        Returns the list of file paths that were successfully created.
        Raises ``OSError`` on any write failure so the caller can convert
        the verdict to FAIL.
        """
        written: List[Path] = []

        for case, cr, er, comp, cls_ in zip(
            cases, compile_results, exec_results, comparisons, classifications
        ):
            sanitized = _sanitize_path_component(case.name)
            case_dir = self._root / self._suite / sanitized / opt / backend
            case_dir.mkdir(parents=True, exist_ok=True)

            # case.ptx
            self._write_or_raise(case_dir / "case.ptx", case.ptx.encode("utf-8"), written)

            # compile stdout / stderr
            self._write_or_raise(case_dir / "compile.stdout.txt", cr.stdout.encode("utf-8"), written)
            self._write_or_raise(case_dir / "compile.stderr.txt", cr.stderr.encode("utf-8"), written)

            # compile_report.json
            if cr.report is not None:
                data = json.dumps(cr.report, indent=2).encode("utf-8")
                self._write_or_raise(case_dir / "compile_report.json", data, written)

            # program.aecbin (copy)
            if cr.aecbin is not None and cr.aecbin.is_file():
                shutil.copy2(cr.aecbin, case_dir / "program.aecbin")
                written.append(case_dir / "program.aecbin")

            # program.asm (disassembly)
            self._write_disassembly(case_dir, cr.aecbin, written)

            # result.json
            result_data = {
                "case": case.name,
                "suite": case.suite,
                "opt": opt,
                "backend": backend,
                "verdict": cls_.verdict,
                "compile_returncode": cr.returncode,
                "exec_returncode": er.returncode,
                "exec_status": er.status,
                "exec_cycles": er.cycles,
                "exec_detail": er.detail,
                "comparison_matched": comp.matched,
            }
            if not comp.matched:
                result_data["first_mismatch_index"] = comp.first_mismatch_index
                result_data["expected_value"] = comp.expected_value
                result_data["actual_value"] = comp.actual_value
                result_data["max_abs_diff"] = comp.max_abs_diff
            text = json.dumps(result_data, indent=2).encode("utf-8")
            self._write_or_raise(case_dir / "result.json", text, written)

            # expected.bin — pack the expected values from OutputExpectation
            out = case.output
            dt = np.dtype(out.dtype)
            expected_arr = np.array(out.expected, dtype=dt)
            self._write_or_raise(case_dir / "expected.bin", expected_arr.tobytes(), written)

            # actual.bin
            self._write_or_raise(case_dir / "actual.bin", er.output, written)

        return written

    @staticmethod
    def _write(path: Path, data: bytes, written: List[Path]) -> None:
        """Write *data* to *path*, appending to *written* on success."""
        try:
            path.write_bytes(data)
            written.append(path)
        except OSError:
            pass

    @staticmethod
    def _write_or_raise(path: Path, data: bytes, written: List[Path]) -> None:
        """Write *data* to *path* or raise ``OSError``.

        Required artifact writes use this for fail-on-failure semantics.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written.append(path)

    @staticmethod
    def _write_disassembly(case_dir: Path, aecbin: Optional[Path], written: List[Path]) -> None:
        """Attempt to write a disassembly of *aecbin*.

        Uses :func:`_find_disassembly_tool` to locate ``aec-objdump``,
        then falls back to ``aec-cc --objdump`` and ``llvm-objdump``.
        Writes a placeholder on total failure.
        """
        if aecbin is None or not aecbin.is_file():
            ArtifactWriter._write_or_raise(case_dir / "program.asm", b"(no binary)\n", written)
            return

        # Project bin dir is C1/bin/ relative to this file (C1/tests/extreme/).
        _script_dir = Path(__file__).resolve().parent
        _c1_root = _script_dir.parent.parent          # C1/
        _proj_bin = _c1_root / "bin"

        # Build candidate command-lists — prefer project-local aec-objdump.
        candidates: List[Tuple[str, ...]] = []
        for path in _find_disassembly_tool("aec-objdump", _proj_bin):
            candidates.append((path, str(aecbin)))
        candidates.append(("aec-cc", "--objdump", str(aecbin)))
        for path in _find_disassembly_tool("llvm-objdump", _proj_bin):
            candidates.append((path, "-d", str(aecbin)))

        for cmd in candidates:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    ArtifactWriter._write_or_raise(
                        case_dir / "program.asm",
                        result.stdout.encode("utf-8") if isinstance(result.stdout, str) else result.stdout,
                        written,
                    )
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue

        ArtifactWriter._write_or_raise(
            case_dir / "program.asm",
            b"(no disassembly tool available)\n",
            written,
        )
