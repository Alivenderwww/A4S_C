#!/usr/bin/env python3
"""Run one executable public requirement case."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ('R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R201', 'R202', 'R203', 'R204', 'R301', 'R302', 'R303', 'R304', 'R401', 'R402')


def run(requirement: str, submission: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="aec-case-") as temporary:
        report = Path(temporary) / "report.json"
        command = [sys.executable, str(ROOT / "grader" / "public_grade.py"),
                   "--submission", str(submission.resolve()), "--profile", "public",
                   "--requirement", requirement, "--json-out", str(report), "--quiet"]
        result = subprocess.run(command, cwd=ROOT, text=True, check=False)
        if result.returncode != 0 or not report.is_file():
            print(f"FAIL {requirement}: grader exit={result.returncode}")
            return 1
        payload = json.loads(report.read_text(encoding="utf-8"))
        outcome = payload["requirements"][requirement]
        marker = "PASS" if outcome["passed"] else "FAIL"
        print(f"{marker} {requirement}: {outcome['earned']}/{outcome['possible']} — "
              f"{outcome['detail']}")
        return 0 if outcome["passed"] else 1


def main(default_requirement: str | None = None) -> int:
    parser = argparse.ArgumentParser()
    if default_requirement is None:
        parser.add_argument("requirement", choices=REQUIREMENTS)
    parser.add_argument("--submission", type=Path, default=ROOT)
    args = parser.parse_args()
    requirement = default_requirement or args.requirement
    return run(requirement, args.submission)


if __name__ == "__main__":
    raise SystemExit(main())
