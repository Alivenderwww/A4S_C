#!/usr/bin/env python3
"""CLI entry point for the extreme correctness test suite.

Usage examples::

    python tests/extreme/run_extreme.py --suite contract --backend local --opt all
    python tests/extreme/run_extreme.py --suite frontier --backend local --opt O2
    python tests/extreme/run_extreme.py --list
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure the C1 root (containing tests/) is findable when running as a script.
_C1_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_C1_ROOT) not in sys.path:
    sys.path.insert(0, str(_C1_ROOT))

from tests.extreme.backends import (
    CompileResult,
    ExecutionResult,
    compile_case,
    execute_by_backend,
    select_cmodel,
    select_compiler,
    validate_compile,
)
from tests.extreme.cases import (
    ExtremeCase,
    RegistryError,
    apply_registry_to_cases,
    load_pressure_registry,
)
from tests.extreme.runner import (
    ArtifactWriter,
    Classification,
    ComparisonResult,
    classify_case,
    compare_output,
    discover_cases,
    filter_opt_levels,
    summarize_results,
)


def _repo_root() -> Path:
    """Resolve the repository root (A4S_C) from this file's location."""
    return _C1_ROOT.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run extreme correctness tests for the A4S C1 compiler."
    )
    parser.add_argument(
        "--suite",
        choices=["contract", "frontier"],
        help="Test suite to execute (contract = must-pass, frontier = known limits).",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "cmodel"],
        help="Execution backend (local simulator or official CModel).",
    )
    parser.add_argument(
        "--opt",
        choices=["O0", "O2", "O3", "all"],
        help="Optimisation level to apply.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic case generation.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Directory to write per-case artifact files.",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "strict"],
        default=None,
        help="Profile selection: fast (default for local) or strict (default for cmodel).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List discovered cases without executing them.",
    )
    return parser


def get_profile_default(backend: str) -> str:
    """Return the default profile for a given backend.

    ``"local"`` → ``"fast"``; ``"cmodel"`` → ``"strict"``.
    """
    if backend == "cmodel":
        return "strict"
    return "fast"


def _run_case_opt(
    compiler: Path,
    case: ExtremeCase,
    opt: str,
    artifact_writer: Optional[ArtifactWriter],
    artifacts_root: Optional[Path],
    work_dir: Path,
    backend: str = "local",
    cmodel: Optional[Path] = None,
) -> Tuple[CompileResult, ExecutionResult, ComparisonResult, Classification]:
    """Run a single case at a single opt level. Returns all results."""
    # ── Compile ────────────────────────────────────────────────────────
    cr = compile_case(compiler, case, opt, work_dir)

    # Validate compile — failure is fatal for this combination.
    try:
        validate_compile(cr, opt)
    except ValueError:
        # Compile did not produce a valid binary. Return a failure
        # classification early.
        er = ExecutionResult(returncode=-1, status="fail", output=b"", detail="compile validation failed")
        comp = ComparisonResult(matched=False)
        cls_ = classify_case(case.suite, case, comp, observed_phase="compile", observed_detail="compile validation failed")
        return cr, er, comp, cls_

    # ── Execute ────────────────────────────────────────────────────────
    assert cr.aecbin is not None  # guaranteed by validate_compile
    er = execute_by_backend(backend, case, cr.aecbin, cmodel=cmodel, work_dir=work_dir)

    observed_phase: str = "compare"
    observed_detail: str = ""

    # ── Compare ────────────────────────────────────────────────────────
    if er.status == "pass":
        comp = compare_output(er.output, case.output)
    else:
        comp = ComparisonResult(matched=False)
        observed_phase = "execute"
        observed_detail = er.detail or "execution failed"

    # ── Classify ───────────────────────────────────────────────────────
    cls_ = classify_case(case.suite, case, comp, observed_phase=observed_phase, observed_detail=observed_detail)

    # ── Artifacts ──────────────────────────────────────────────────────
    if artifact_writer is not None and artifacts_root is not None:
        try:
            artifact_writer.write_all(
                [case], [cr], [er], [comp], [cls_],
                opt=opt, backend=backend,
            )
        except OSError:
            # Artifact write failure → convert verdict to FAIL.
            cls_ = Classification(verdict="FAIL")
        except Exception:
            # Any other artifact exception → FAIL as well.
            cls_ = Classification(verdict="FAIL")

    return cr, er, comp, cls_


def main(argv: Optional[List[str]] = None) -> int:
    """Parse args, discover cases, and run the suite. Returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    root = _repo_root()

    # --list does not require --suite or --backend
    if args.list_only:
        suite = args.suite or "contract"
        try:
            cases = discover_cases(root, suite)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for c in cases:
            print(f"{c.name}  suite={c.suite}  opt={','.join(c.opt_levels)}")
        return 0

    # Validate required flags.
    errors: List[str] = []
    if not args.suite:
        errors.append("--suite is required (contract|frontier)")
    if not args.backend:
        errors.append("--backend is required (local|cmodel)")
    if not args.opt:
        errors.append("--opt is required (O0|O2|O3|all)")

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1

    # Discover cases — every exception prints a concise stderr and returns 1.
    try:
        cases = discover_cases(root, args.suite)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Locate compiler – the compiler lives under C1/, not the repo root.
    try:
        compiler = select_compiler(_C1_ROOT)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Locate CModel binary if backend is cmodel.
    cmodel_path: Optional[Path] = None
    if args.backend == "cmodel":
        try:
            cmodel_path = select_cmodel(root)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    # Apply profile.
    profile = args.profile if args.profile is not None else get_profile_default(args.backend)
    if profile == "strict":
        from tests.extreme.cases import select_strict_profile
        cases = select_strict_profile(cases)
        print(f"strict profile: {sum(len(c.opt_levels) for c in cases)} case×opt combination(s)")

    # Apply expected-failure registry overlay.
    try:
        registry = load_pressure_registry(root)
        cases = apply_registry_to_cases(cases, registry, backend=args.backend)
    except RegistryError as exc:
        print(f"error: registry: {exc}", file=sys.stderr)
        return 1

    # Filter by opt level.
    cases = filter_opt_levels(cases, args.opt)

    # Expand case×opt combinations.
    expanded: List[Tuple[ExtremeCase, str]] = []
    for case in cases:
        for opt in case.opt_levels:
            expanded.append((case, opt))
    total = len(expanded)
    if total == 0:
        print("error: no case×opt combinations to run", file=sys.stderr)
        return 1

    # Set up artifacts writer if requested.
    artifact_writer: Optional[ArtifactWriter] = None
    artifacts_root: Optional[Path] = None
    if args.artifacts is not None:
        ar = args.artifacts.resolve()
        artifact_writer = ArtifactWriter(ar, args.suite)
        artifacts_root = ar
    else:
        artifacts_root = None

    # ── Execution loop ─────────────────────────────────────────────────
    classifications: List[Classification] = []
    seed_info = f" seed={args.seed}" if args.seed is not None else ""
    print(f"running {total} case(s) suite={args.suite} backend={args.backend} "
          f"opt={args.opt}{seed_info}")

    for idx, (case, opt) in enumerate(expanded):
        label = f"[{idx + 1}/{total}] {case.name} opt={opt}"

        _cr: Optional[CompileResult] = None
        _er: Optional[ExecutionResult] = None
        comp: ComparisonResult = ComparisonResult(matched=False)
        cls_: Classification = Classification(verdict="FAIL")

        # Work directory for this case+opt combination.
        with tempfile.TemporaryDirectory(prefix="extreme_") as tmp:
            work_dir = Path(tmp)
            try:
                _cr, _er, comp, cls_ = _run_case_opt(
                    compiler, case, opt, artifact_writer, artifacts_root, work_dir,
                    backend=args.backend, cmodel=cmodel_path,
                )
            except Exception as exc:
                # Fail-closed: any exception → FAIL
                cls_ = Classification(verdict="FAIL")
                comp = ComparisonResult(matched=False)
                print(f"error: {label} — {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        classifications.append(cls_)
        # Include failure phase/detail for diagnosability
        phase_detail = ""
        if cls_.verdict in ("FAIL", "XFAIL") and _er is not None and not comp.matched:
            if _er.detail:
                phase_detail = f"  [{_er.detail}]"
        print(f"{cls_.verdict}{phase_detail}  {label}")
        sys.stdout.flush()

    # ── Summary ────────────────────────────────────────────────────────
    exit_code = summarize_results(classifications)
    verdict_counts = _count_verdicts(classifications)
    print(f"\nsummary: {verdict_counts}")
    print(f"exit: {'PASS' if exit_code == 0 else 'FAIL'} ({exit_code})")
    return exit_code


def _count_verdicts(classifications: List[Classification]) -> str:
    counts: dict = {}
    for c in classifications:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    return " ".join(parts) if parts else "(empty)"


if __name__ == "__main__":
    sys.exit(main())
