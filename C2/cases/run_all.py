#!/usr/bin/env python3
"""Run all public requirement cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from run_case import REQUIREMENTS, ROOT, run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=ROOT)
    args = parser.parse_args()
    failures = sum(run(requirement, args.submission) for requirement in REQUIREMENTS)
    print(f"{len(REQUIREMENTS) - failures}/{len(REQUIREMENTS)} cases passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
