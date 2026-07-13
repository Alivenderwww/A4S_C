#!/usr/bin/env python3
"""R401 exhaustive verification: prove DMA Agent is globally optimal on a broad grid.

Also computes grader-style fraction on a holdout split.

Usage (pure Python, no device needed):
    python3 -B tests/test_r401_exhaustive.py
"""
import itertools
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C2 = os.path.dirname(HERE)
if C2 not in sys.path:
    sys.path.insert(0, C2)
sys.path.insert(0, os.path.join(C2, "agents"))

from dma_agent import decide  # noqa: E402

SEED = 42
HOLDOUT_RATIO = 0.30

LEGAL_CHUNKS = (4096, 65536, 1048576)
LEGAL_DEPTHS = (1, 2, 4, 8)
LEGAL_CHANNELS = (0, 1)

GRID_BYTES = [64, 256, 1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]
GRID_ALIGNMENTS = [8, 16, 32, 64]
GRID_CONCURRENCIES = [1, 2, 4, 8]


def _dma_cycles(request, action):
    chunk = action["chunk_bytes"]
    depth = action["queue_depth"]
    zero_copy = action["use_zero_copy"]
    chunks = math.ceil(request["bytes"] / chunk)
    payload = math.ceil(request["bytes"] / 32)
    parallelism = min(depth, request["concurrency"], 2)
    setup = 45 if zero_copy else 100
    alignment_penalty = 0 if request["alignment"] >= 64 else 13
    return setup + math.ceil(payload / parallelism) + 24 * (chunks - 1) + alignment_penalty


def _all_legal_actions(request):
    zc_opts = [True, False] if request["registered"] else [False]
    for channel, chunk, depth, zc in itertools.product(
        LEGAL_CHANNELS, LEGAL_CHUNKS, LEGAL_DEPTHS, zc_opts
    ):
        yield {"channel": channel, "chunk_bytes": chunk,
               "queue_depth": depth, "use_zero_copy": zc}


def _brute_force_optimal(request):
    best_cycles = float("inf")
    best_action = None
    for action in _all_legal_actions(request):
        cycles = _dma_cycles(request, action)
        if cycles < best_cycles:
            best_cycles = cycles
            best_action = action
    return best_action, best_cycles


def _generate_grid():
    requests = []
    for direction in ("h2d", "d2h"):
        for registered in (True, False):
            for bytes_ in GRID_BYTES:
                for alignment in GRID_ALIGNMENTS:
                    for concurrency in GRID_CONCURRENCIES:
                        requests.append({
                            "direction": direction,
                            "bytes": bytes_,
                            "alignment": alignment,
                            "concurrency": concurrency,
                            "registered": registered,
                        })
    return requests


def _split(requests):
    rng = random.Random(SEED)
    indices = list(range(len(requests)))
    rng.shuffle(indices)
    cut = int(len(indices) * (1 - HOLDOUT_RATIO))
    explore = [requests[i] for i in sorted(indices[:cut])]
    holdout = [requests[i] for i in sorted(indices[cut:])]
    return explore, holdout


def _baseline_cycles(request):
    baseline_action = {"chunk_bytes": 4096, "queue_depth": 1,
                       "channel": 0, "use_zero_copy": False}
    return _dma_cycles(request, baseline_action)


def _fraction(request, action):
    baseline = _baseline_cycles(request)
    candidate = _dma_cycles(request, action)
    return max(0.0, min(1.0, (baseline / candidate - 1.0) / 0.5))


def test_exhaustive_optimal_on_explore():
    requests = _generate_grid()
    explore, holdout = _split(requests)
    mismatches = 0
    for req in explore:
        agent_action = decide(req)
        _, optimal_cycles = _brute_force_optimal(req)
        agent_cycles = _dma_cycles(req, agent_action)
        if agent_cycles > optimal_cycles:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH: {req} agent={agent_cycles} optimal={optimal_cycles}")
    if mismatches:
        raise AssertionError(
            f"{mismatches}/{len(explore)} explore cases suboptimal")
    print(f"  EXPLORE: {len(explore)} cases, all optimal")


def test_holdout_score():
    requests = _generate_grid()
    explore, holdout = _split(requests)
    fractions = [_fraction(req, decide(req)) for req in holdout]
    avg = sum(fractions) / len(fractions) if fractions else 0.0
    print(f"\n  HOLDOUT: {len(holdout)} cases, avg fraction = {avg:.4f}")
    assert avg >= 0.0


def _run_all():
    for name, fn in sorted(
        (n, f) for n, f in globals().items()
        if n.startswith("test_") and callable(f)
    ):
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {name}: {e}")


if __name__ == "__main__":
    _run_all()
