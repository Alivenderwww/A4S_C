#!/usr/bin/env python3
"""R402 oracle analysis: find counterexamples and estimate hidden performance.

Loads scan_results.json (from device_oracle_scan.py), splits into explore/holdout
by fixed seed, then:
  1. Explore: find cases where "highest legal variant" != fastest legal variant.
  2. Holdout: compute grader-style fraction for current and improved strategy.

Usage (from C2 root, after scan_results.json exists):
    python3 -B tests/test_r402_oracle.py
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C2 = os.path.dirname(HERE)
for p in (C2, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from runtime_harness import SkipTest, require  # noqa: E402

SCAN_RESULTS = os.path.join(C2, "scan_results.json")
SEED = 42
HOLDOUT_RATIO = 0.30

VARIANT_ORDER = {"naive": 1, "tiled": 2, "vectorized": 3}


def _load_results():
    if not os.path.exists(SCAN_RESULTS):
        raise SkipTest("no scan_results.json — run device_oracle_scan.py first")
    with open(SCAN_RESULTS) as f:
        return json.load(f)


def _split(results):
    rng = random.Random(SEED)
    indices = list(range(len(results)))
    rng.shuffle(indices)
    cut = int(len(indices) * (1 - HOLDOUT_RATIO))
    explore = [results[i] for i in sorted(indices[:cut])]
    holdout = [results[i] for i in sorted(indices[cut:])]
    return explore, holdout


def _current_strategy_choice(case):
    """kernel_agent.py without diagnostic_cycles: highest legal variant."""
    legal = [(v, info) for v, info in case["variants"].items()
             if isinstance(info, dict) and "cycles" in info]
    if not legal:
        return None, None
    legal.sort(key=lambda x: VARIANT_ORDER[x[0]], reverse=True)
    return legal[0]


def _best_choice(case):
    """Truly fastest legal variant."""
    legal = [(v, info) for v, info in case["variants"].items()
             if isinstance(info, dict) and "cycles" in info]
    if not legal:
        return None, None
    legal.sort(key=lambda x: x[1]["cycles"])
    return legal[0]


def _grader_fraction(case, chosen_info):
    """grader-style: (naive_cycles/chosen_cycles - 1) / 0.5, clamped [0,1]."""
    naive = case["variants"].get("naive", {})
    if not isinstance(naive, dict) or "cycles" not in naive:
        return 0.0
    if "cycles" not in chosen_info:
        return 0.0
    baseline = naive["cycles"]
    candidate = chosen_info["cycles"]
    if candidate <= 0:
        return 0.0
    return max(0.0, min(1.0, (baseline / candidate - 1.0) / 0.5))


def _analyze_explore(explore):
    counterexamples = []
    for i, case in enumerate(explore):
        cur_name, cur_info = _current_strategy_choice(case)
        best_name, best_info = _best_choice(case)
        if cur_name is None or best_name is None:
            continue
        if cur_name != best_name:
            counterexamples.append({
                "index": i,
                "dtype": case["dtype"],
                "m": case["m"], "n": case["n"], "k": case["k"],
                "alignment": case["alignment"],
                "workspace": case["workspace"],
                "current": cur_name,
                "current_cycles": cur_info["cycles"],
                "best": best_name,
                "best_cycles": best_info["cycles"],
            })
    return counterexamples


def _score_holdout(holdout, strategy_fn):
    fractions = []
    for case in holdout:
        name, info = strategy_fn(case)
        if name is None or info is None:
            fractions.append(0.0)
        else:
            fractions.append(_grader_fraction(case, info))
    return sum(fractions) / len(fractions) if fractions else 0.0


def test_explore_counterexamples():
    results = _load_results()
    explore, holdout = _split(results)
    ces = _analyze_explore(explore)
    if ces:
        print(f"\n  EXPLORE: {len(ces)} counterexamples found:")
        for ce in ces[:20]:
            print(f"    {ce['dtype']} {ce['m']}x{ce['n']}x{ce['k']} "
                  f"align={ce['alignment']} ws={ce['workspace']}: "
                  f"current={ce['current']}({ce['current_cycles']}) "
                  f"best={ce['best']}({ce['best_cycles']})")
        if len(ces) > 20:
            print(f"    ... and {len(ces) - 20} more")
    else:
        print("\n  EXPLORE: no counterexamples — strategy already optimal")


def test_holdout_current_score():
    results = _load_results()
    explore, holdout = _split(results)
    score = _score_holdout(holdout, _current_strategy_choice)
    print(f"\n  HOLDOUT (current strategy): avg fraction = {score:.4f}")
    require(score >= 0.0, "holdout score negative")


def test_holdout_optimal_score():
    results = _load_results()
    explore, holdout = _split(results)
    score = _score_holdout(holdout, _best_choice)
    print(f"\n  HOLDOUT (optimal oracle): avg fraction = {score:.4f}")


def _run_all():
    for name, fn in sorted(
        (n, f) for n, f in globals().items()
        if n.startswith("test_") and callable(f)
    ):
        try:
            fn()
            print(f"PASS  {name}")
        except SkipTest as e:
            print(f"SKIP  {name}: {e}")
        except Exception as e:
            print(f"FAIL  {name}: {e}")


if __name__ == "__main__":
    _run_all()
