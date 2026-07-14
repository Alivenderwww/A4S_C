"""Backend abstractions: compile, validate, and execute via the local simulator.

Provides the data types and helpers that the extreme-test runner uses to
invoke the compiler, validate its output, and run the resulting binary
through the local AEC functional simulator.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Ensure the repo root (A4S_C) is on sys.path for cross-package imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.extreme.cases import ExtremeCase


# ════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CompileResult:
    """Result of compiling a single PTX case through aec-cc."""

    returncode: int
    stdout: str
    stderr: str
    aecbin: Optional[Path] = None
    report: Optional[dict] = None


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a compiled .aecbin on the local simulator."""

    returncode: int
    status: str  # "pass" or "fail"
    output: bytes
    cycles: Optional[int] = None
    detail: str = ""


# ════════════════════════════════════════════════════════════════════════
# Compiler discovery
# ════════════════════════════════════════════════════════════════════════


def select_compiler(root: Path) -> Path:
    """Find the aec-cc compiler binary with priority: .exe → bin/aec-cc → compiler/aec-cc.

    Raises:
        FileNotFoundError: When no candidate exists.
    """
    candidates = [
        root / "bin" / "aec-cc.exe",
        root / "bin" / "aec-cc",
        root / "compiler" / "aec-cc",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no compiler found at any of: {', '.join(str(c) for c in candidates)}"
    )


# ════════════════════════════════════════════════════════════════════════
# Build compile command
# ════════════════════════════════════════════════════════════════════════


def build_compile_command(
    compiler: Path,
    ptx_path: Path,
    opt: str,
    aecbin_path: Path,
    report_path: Path,
) -> List[str]:
    """Build the aec-cc command-line for a PTX input at optimisation level *opt*.

    Real ``aec-cc --help`` shows::

        ./bin/aec-cc --help
        Usage: aec-cc input.ptx [-O0|-O2|-O3] [-o out] [--report file]

    Returns a list of strings suitable for ``subprocess.run``.
    """
    return [
        str(compiler),
        str(ptx_path),
        f"-{opt}",
        "-o", str(aecbin_path),
        "--report", str(report_path),
    ]


# ════════════════════════════════════════════════════════════════════════
# Compile validation
# ════════════════════════════════════════════════════════════════════════


def validate_compile(result: CompileResult, opt: str) -> None:
    """Validate a compile result.  Raises ``ValueError`` on any failure.

    A zero process exit is still treated as failure unless:
    - the .aecbin file exists, is non-empty, and is 16-byte aligned;
    - ``report`` is a dict with ``status='ok'``, ``opt_level`` matching
      *opt*, a non-negative integer ``num_aec_instructions``, and
      ``num_aec_instructions * 16 == len(actual_aecbin_bytes)``.
    """
    # --- aecbin existence & shape ---
    if result.aecbin is None or not result.aecbin.is_file():
        raise ValueError("aecbin not found")
    aecbin_bytes = result.aecbin.read_bytes()
    if len(aecbin_bytes) == 0:
        raise ValueError("aecbin is empty")
    if len(aecbin_bytes) % 16 != 0:
        raise ValueError(
            f"aecbin size {len(aecbin_bytes)} is not 16-byte aligned"
        )

    # --- report existence & shape ---
    if result.report is None:
        raise ValueError("report is missing")
    report = result.report
    if not isinstance(report, dict):
        raise ValueError("report is not a dict")

    if report.get("status") != "ok":
        raise ValueError(
            f"report status is '{report.get('status')}', expected 'ok'"
        )
    if report.get("opt_level") != opt:
        raise ValueError(
            f"report opt_level '{report.get('opt_level')}' does not match expected '{opt}'"
        )

    inst_count = report.get("num_aec_instructions")
    if inst_count is None or not isinstance(inst_count, int) or inst_count < 0:
        raise ValueError(f"invalid num_aec_instructions: {inst_count}")

    if inst_count * 16 != len(aecbin_bytes):
        raise ValueError(
            f"num_aec_instructions {inst_count} * 16 = {inst_count * 16} "
            f"!= aecbin size {len(aecbin_bytes)}"
        )


# ════════════════════════════════════════════════════════════════════════
# Compile one case
# ════════════════════════════════════════════════════════════════════════


def compile_case(
    compiler: Path,
    case: ExtremeCase,
    opt: str,
    work_dir: Path,
    timeout: int = 60,
) -> CompileResult:
    """Compile *case* at optimisation level *opt* using *compiler*.

    Writes ``case.ptx`` to *work_dir*, invokes ``aec-cc`` with
    ``-O0``/``-O2``/``-O3`` and ``--report``, and returns a
    :class:`CompileResult` with the parsed report dict and path to the
    generated ``.aecbin``.
    """
    ptx_path = work_dir / "case.ptx"
    ptx_path.write_text(case.ptx, encoding="utf-8")

    aecbin_path = work_dir / "program.aecbin"
    report_path = work_dir / "compile_report.json"

    cmd = build_compile_command(compiler, ptx_path, opt, aecbin_path, report_path)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        so = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "")
        se = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "")
        return CompileResult(
            returncode=-1,
            stdout=so,
            stderr=se,
            aecbin=None,
            report=None,
        )
    except FileNotFoundError:
        return CompileResult(
            returncode=-1,
            stdout="",
            stderr=f"compiler not found: {compiler}",
            aecbin=None,
            report=None,
        )

    # Load report if it exists and is valid JSON.
    report: Optional[dict] = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    aecbin = aecbin_path if aecbin_path.is_file() else None

    return CompileResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        aecbin=aecbin,
        report=report,
    )


# ════════════════════════════════════════════════════════════════════════
# Local simulator execution
# ════════════════════════════════════════════════════════════════════════


def execute_local(
    case: ExtremeCase,
    aecbin_path: Path,
    timeout: int = 30,
) -> ExecutionResult:
    """Execute a compiled ``.aecbin`` on the local simulator.

    Imports ``aec_sim`` from ``C1/sim/`` and calls
    ``simulate(…, strict=True)`` with the case's grid, block, pmem, and
    gmem.  Catches simulator exceptions and returns the output byte range
    specified by ``case.output`` (offset + dtype × shape).
    """
    import numpy as np
    # C1/sim/ is not a regular package (no __init__.py), so add it to
    # sys.path to enable absolute imports of aec_sim and its dependency
    # aec_decode.
    _SIM_DIR = Path(__file__).resolve().parent.parent.parent / "sim"
    if str(_SIM_DIR) not in sys.path:
        sys.path.insert(0, str(_SIM_DIR))
    from aec_sim import ExecError, simulate  # type: ignore[import-untyped]

    try:
        gmem, total_cycles, _warps = simulate(
            aecbin_path,
            grid=case.grid,
            block=case.block,
            param_block=case.pmem,
            gmem_init=case.gmem,
            strict=True,
        )
    except ExecError as e:
        return ExecutionResult(
            returncode=1,
            status="fail",
            output=b"",
            detail=f"simulation error: {e}",
        )
    except Exception as e:
        return ExecutionResult(
            returncode=1,
            status="fail",
            output=b"",
            detail=f"unexpected error: {e}",
        )

    # Extract the output byte range.
    out = case.output
    dt = np.dtype(out.dtype)
    elem_size = dt.itemsize
    out_size = int(np.prod(out.shape)) * elem_size
    output = bytes(gmem[out.offset: out.offset + out_size])

    return ExecutionResult(
        returncode=0,
        status="pass",
        output=output,
        cycles=total_cycles,
        detail="",
    )


# ════════════════════════════════════════════════════════════════════════
# CModel binary discovery
# ════════════════════════════════════════════════════════════════════════


def select_cmodel(root: Path) -> Path:
    """Find the aec-precise CModel release binary.

    Searches ``public/aec-cmodel-release/bin/aec-precise-linux-x86_64``
    under *root*.  Raises ``FileNotFoundError`` when no candidate exists.

    Args:
        root: Repository root directory (``A4S_C``).

    Returns:
        Absolute path to the CModel binary.

    Raises:
        FileNotFoundError: No release binary found at expected location.
    """
    candidate = (
        root / "public" / "aec-cmodel-release" / "bin" / "aec-precise-linux-x86_64"
    )
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"no CModel binary found at {candidate} — "
        "ensure public/aec-cmodel-release/bin/aec-precise-linux-x86_64 exists"
    )


# ════════════════════════════════════════════════════════════════════════
# Build CModel command
# ════════════════════════════════════════════════════════════════════════


def build_cmodel_command(
    cmodel: Path,
    aecbin: Path,
    ninstr: int,
    grid: tuple,
    block: tuple,
    pmem_path: Path,
    gmem_path: Path,
    dump_path: Path,
    dump_offset: int,
    dump_bytes: int,
) -> list:
    """Build the aec-precise command-line for a compiled .aecbin.

    Follows the flag order from ``C1/sim/cmodel/conformance.py``::

        --program <aecbin>
        --instructions <ninstr>
        --grid x,y,z
        --block x,y,z
        --load pmem:0:<pmem_path>
        --load gmem:0:<gmem_path>
        --dump <offset>:<bytes>:<dump_path>

    Args:
        cmodel: Path to the aec-precise binary.
        aecbin: Path to the compiled .aecbin file.
        ninstr: Number of AEC instructions (= file_size / 16).
        grid: Tuple ``(x, y, z)`` launch grid dimensions.
        block: Tuple ``(x, y, z)`` thread block dimensions.
        pmem_path: Path to the PMEM binary file.
        gmem_path: Path to the GMEM binary file.
        dump_path: Path for the output dump file.
        dump_offset: Byte offset in GMEM to start dumping.
        dump_bytes: Number of bytes to dump.

    Returns:
        List of command-line argument strings.
    """
    return [
        str(cmodel),
        "--program", str(aecbin),
        "--instructions", str(ninstr),
        "--grid", f"{grid[0]},{grid[1]},{grid[2]}",
        "--block", f"{block[0]},{block[1]},{block[2]}",
        "--load", f"pmem:0:{pmem_path}",
        "--load", f"gmem:0:{gmem_path}",
        "--dump", f"{dump_offset}:{dump_bytes}:{dump_path}",
    ]


# ════════════════════════════════════════════════════════════════════════
# CModel execution (fail-closed)
# ════════════════════════════════════════════════════════════════════════


def execute_cmodel(
    cmodel: Path,
    aecbin: Path,
    case: "ExtremeCase",
    work_dir: Path,
    timeout: int = 120,
) -> ExecutionResult:
    """Execute a compiled .aecbin on the official aec-precise CModel.

    Writes ``pmem.bin`` and ``gmem.bin`` to *work_dir*, invokes the CModel
    with the correct flags, and parses the JSON result.  All failure modes
    (missing binary, non-zero exit, malformed JSON, wrong status, missing
    or wrong-length dump) return ``ExecutionResult(status="fail")`` — never
    raise.

    Args:
        cmodel: Path to the aec-precise binary.
        aecbin: Path to the compiled .aecbin.
        case: The :class:`ExtremeCase` being executed.
        work_dir: Temporary working directory for input/output files.
        timeout: Subprocess timeout in seconds (default 120).

    Returns:
        :class:`ExecutionResult` with ``status="pass"`` and ``output`` /
        ``cycles`` on success, or ``status="fail"`` on any error.
    """
    import numpy as np

    out = case.output
    dt = np.dtype(out.dtype)
    elem_size = dt.itemsize
    output_size = int(np.prod(out.shape)) * elem_size

    # Write input files
    pmem_path = work_dir / "pmem.bin"
    gmem_path = work_dir / "gmem.bin"
    dump_path = work_dir / "actual.bin"

    pmem_path.write_bytes(case.pmem)
    gmem_path.write_bytes(case.gmem)

    # Compute instruction count from the binary
    try:
        aecbin_bytes = aecbin.read_bytes()
    except OSError:
        return ExecutionResult(
            returncode=-1, status="fail", output=b"",
            detail="cannot read aecbin file",
        )
    ninstr = len(aecbin_bytes) // 16

    cmd = build_cmodel_command(
        cmodel=cmodel,
        aecbin=aecbin,
        ninstr=ninstr,
        grid=case.grid,
        block=case.block,
        pmem_path=pmem_path,
        gmem_path=gmem_path,
        dump_path=dump_path,
        dump_offset=out.offset,
        dump_bytes=output_size,
    )

    # --- subprocess ---
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return ExecutionResult(
            returncode=-1, status="fail", output=b"",
            detail=f"cmodel not found: {cmodel}",
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            returncode=-1, status="fail", output=b"",
            detail="cmodel timed out",
        )

    # Non-zero exit
    if proc.returncode != 0:
        stdout_detail = proc.stdout.strip()[:240]
        stderr_detail = proc.stderr.strip()[:240]
        return ExecutionResult(
            returncode=proc.returncode, status="fail", output=b"",
            detail=(
                f"cmodel exited {proc.returncode}: "
                f"stdout={stdout_detail!r} stderr={stderr_detail!r}"
            ),
        )

    # Parse JSON
    try:
        result_json = json.loads(proc.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return ExecutionResult(
            returncode=proc.returncode, status="fail", output=b"",
            detail="cmodel stdout is not valid JSON",
        )

    if not isinstance(result_json, dict):
        return ExecutionResult(
            returncode=proc.returncode, status="fail", output=b"",
            detail="cmodel stdout JSON is not a dict",
        )

    # Check status
    if result_json.get("status") != "done":
        return ExecutionResult(
            returncode=proc.returncode, status="fail", output=b"",
            detail=f"cmodel status is '{result_json.get('status')}', expected 'done'",
        )

    # Check dump file exists
    if not dump_path.is_file():
        return ExecutionResult(
            returncode=proc.returncode, status="fail", output=b"",
            detail="cmodel dump file not found",
        )

    output = dump_path.read_bytes()

    # Check output length
    if len(output) != output_size:
        return ExecutionResult(
            returncode=proc.returncode, status="fail", output=b"",
            detail=f"dump file size {len(output)} != expected {output_size}",
        )

    # Extract steps/cycles
    steps = result_json.get("steps")
    if steps is None:
        steps = result_json.get("cycles", 0)

    return ExecutionResult(
        returncode=0,
        status="pass",
        output=output,
        cycles=int(steps),
        detail="",
    )


# ════════════════════════════════════════════════════════════════════════
# Backend dispatch
# ════════════════════════════════════════════════════════════════════════


def execute_by_backend(
    backend: str,
    case: "ExtremeCase",
    aecbin_path: Path,
    cmodel: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> ExecutionResult:
    """Dispatch execution to the appropriate backend.

    Args:
        backend: ``"local"`` or ``"cmodel"``.
        case: The :class:`ExtremeCase` to execute.
        aecbin_path: Path to the compiled ``.aecbin``.
        cmodel: Path to the CModel binary (required for ``"cmodel"``).
        work_dir: Working directory (required for ``"cmodel"``).

    Returns:
        :class:`ExecutionResult` from the chosen backend.

    Raises:
        ValueError: If *backend* is unknown.
    """
    if backend == "local":
        return execute_local(case, aecbin_path)
    elif backend == "cmodel":
        if cmodel is None:
            raise ValueError("cmodel path is required for 'cmodel' backend")
        if work_dir is None:
            raise ValueError("work_dir is required for 'cmodel' backend")
        return execute_cmodel(cmodel, aecbin_path, case, work_dir)
    raise ValueError(f"unknown backend: '{backend}' — must be 'local' or 'cmodel'")
