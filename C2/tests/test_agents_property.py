#!/usr/bin/env python3
"""Property tests for the C2 Excellent-level agents.

The hidden performance score (R401/R402, 6 pts each) is only as safe as the
agents' policy is on inputs we have NOT seen. These tests prove the policy is
optimal / conformant across a dense sweep of the whole *legal* input space, so a
hidden edge case cannot quietly make an agent pick a slower (or illegal) action.

DMA (R401): the virtual-cycle cost model is fully documented (doc 05 sec 3), so
we brute-force the minimum-cost legal action for each request and assert the
agent's action costs exactly that. Kernel (R402): cycles come from image
interpretation (no host formula), so we assert *policy conformance* against the
spec's tier rules -- the agent must pick a LEGAL candidate, and among legal ones
the min-diagnostic-cycles candidate when cycles are given, else the highest
capability tier (vectorized > tiled > naive, i.e. max variant).

Also runs a protocol smoke test via subprocess: each agent must emit exactly one
JSON object on stdout with only the allowed keys and no extra output.

Run:  python3 tests/test_agents_property.py        (exit 0 = all pass)
"""
import itertools
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C2 = os.path.dirname(HERE)
AGENTS = os.path.join(C2, "agents")
sys.path.insert(0, AGENTS)
import dma_agent          # noqa: E402
import kernel_agent       # noqa: E402

# --------------------------------------------------------------------------
# DMA cost model (doc 05 sec 3) -- the single source of truth for R401 optimality
# --------------------------------------------------------------------------
LEGAL_CHUNKS = (4096, 65536, 1048576)
LEGAL_QDS = (1, 2, 4, 8)
LEGAL_CHANNELS = (0, 1)


def dma_cost(nbytes, alignment, zero_copy, chunk, qd, concurrency):
    setup = 45 if zero_copy else 100
    parallelism = min(qd, concurrency, 2)
    chunks = math.ceil(nbytes / chunk)
    align_penalty = 13 if alignment < 64 else 0
    return (setup + math.ceil(math.ceil(nbytes / 32) / parallelism)
            + 24 * (chunks - 1) + align_penalty)


def dma_brute_min(nbytes, alignment, registered, concurrency):
    """Minimum achievable cost over ALL legal actions (channel is cost-free)."""
    zc_opts = (True, False) if registered else (False,)
    best = None
    for zc, chunk, qd in itertools.product(zc_opts, LEGAL_CHUNKS, LEGAL_QDS):
        c = dma_cost(nbytes, alignment, zc, chunk, qd, concurrency)
        best = c if best is None else min(best, c)
    return best


def test_dma_optimal_over_space():
    byte_sizes = [1, 31, 32, 33, 63, 64, 127, 128, 255, 256, 1023, 1024,
                  4095, 4096, 4097, 65535, 65536, 65537, 1048575, 1048576,
                  1048577, 2097152, 4194304, 16777216, 100_000_000]
    aligns = [1, 2, 8, 15, 16, 31, 32, 63, 64, 65, 128, 256, 4096]
    directions = ("h2d", "d2h")
    fails, total = 0, 0
    for nbytes in byte_sizes:
        for align in aligns:
            for reg in (True, False):
                for conc in (1, 2, 3, 4, 8):
                    for direction in directions:
                        total += 1
                        req = {"case_id": 0, "direction": direction, "bytes": nbytes,
                               "alignment": align, "registered": reg, "concurrency": conc}
                        out = dma_agent.decide(req)
                        # --- legality ---
                        assert out["channel"] in LEGAL_CHANNELS, out
                        assert out["chunk_bytes"] in LEGAL_CHUNKS, out
                        assert out["queue_depth"] in LEGAL_QDS, out
                        assert set(out) == {"channel", "chunk_bytes", "queue_depth",
                                            "use_zero_copy"}, out
                        if out["use_zero_copy"]:
                            assert reg, ("zero-copy on unregistered range", req, out)
                        # --- optimality: agent cost == brute-force minimum ---
                        agent_c = dma_cost(nbytes, align, out["use_zero_copy"],
                                           out["chunk_bytes"], out["queue_depth"], conc)
                        min_c = dma_brute_min(nbytes, align, reg, conc)
                        if agent_c != min_c:
                            fails += 1
                            if fails <= 8:
                                print(f"  [DMA SUBOPTIMAL] bytes={nbytes} align={align} "
                                      f"reg={reg} conc={conc}: agent={agent_c} min={min_c} "
                                      f"action={out}")
    ok = fails == 0
    print(f"DMA optimality:  {total} cases -> {'ALL OPTIMAL' if ok else f'{fails} SUBOPTIMAL'}")
    return ok


# --------------------------------------------------------------------------
# Kernel policy conformance (doc 05 sec 4)
# --------------------------------------------------------------------------
def kernel_legal(c, m, n, k, alignment, workspace):
    if c["workspace"] > workspace:
        return False
    if c["alignment"] > alignment:
        return False
    d = c["divisibility"]
    return not (d > 1 and (m % d or n % d or k % d))


def make_candidates(with_cycles, cyc):
    """naive (always legal), tiled (div 4), vectorized (div 8, needs align>=16)."""
    cands = [
        {"id": "naive", "semantic_kernel_id": 1, "image_id": 1, "variant": 1,
         "workspace": 0, "alignment": 1, "divisibility": 1},
        {"id": "tiled", "semantic_kernel_id": 2, "image_id": 2, "variant": 2,
         "workspace": 4096, "alignment": 8, "divisibility": 4},
        {"id": "vec", "semantic_kernel_id": 3, "image_id": 3, "variant": 3,
         "workspace": 8192, "alignment": 16, "divisibility": 8},
    ]
    if with_cycles:
        for c in cands:
            c["diagnostic_cycles"] = cyc[c["id"]]
    return cands


def test_kernel_policy_over_space():
    dims = [1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32, 64, 100, 128, 127, 256]
    aligns = [1, 8, 15, 16, 32, 128]
    workspaces = [0, 4096, 8192, 65536]
    # diagnostic-cycle scenarios, incl. a NON-monotonic one (tiled cheaper than vec)
    cyc_scenarios = [None,
                     {"naive": 1000, "tiled": 500, "vec": 250},   # monotonic: vec best
                     {"naive": 1000, "tiled": 200, "vec": 400}]   # tiled beats vec
    fails, total = 0, 0
    for m, n, k in [(d, d, d) for d in dims] + [(8, 4, 2), (16, 8, 4), (32, 16, 7)]:
        for align in aligns:
            for ws in workspaces:
                for cyc in cyc_scenarios:
                    total += 1
                    cands = make_candidates(cyc is not None, cyc or {})
                    req = {"case_id": 0, "dtype": "f16", "m": m, "n": n, "k": k,
                           "alignment": align, "workspace": ws, "candidates": cands}
                    out = kernel_agent.decide(req)
                    assert set(out) == {"kernel_id"}, out
                    chosen_id = out["kernel_id"]
                    ids = {c["id"] for c in cands}
                    assert chosen_id in ids, (chosen_id, ids)
                    chosen = next(c for c in cands if c["id"] == chosen_id)
                    # independent expected pick
                    legal = [c for c in cands if kernel_legal(c, m, n, k, align, ws)]
                    assert legal, "naive must always be legal"     # sanity
                    # chosen must itself be legal
                    if not kernel_legal(chosen, m, n, k, align, ws):
                        fails += 1
                        if fails <= 8:
                            print(f"  [KERNEL ILLEGAL] m={m} n={n} k={k} align={align} "
                                  f"ws={ws} cyc={cyc} -> {chosen_id}")
                        continue
                    if cyc is not None:
                        want = min(legal, key=lambda c: c["diagnostic_cycles"])
                        if chosen["diagnostic_cycles"] != want["diagnostic_cycles"]:
                            fails += 1
                            if fails <= 8:
                                print(f"  [KERNEL NOT MIN-CYCLE] m={m} n={n} k={k} "
                                      f"align={align} ws={ws} -> {chosen_id} "
                                      f"({chosen['diagnostic_cycles']}) want {want['id']}")
                    else:
                        want = max(legal, key=lambda c: c["variant"])
                        if chosen["variant"] != want["variant"]:
                            fails += 1
                            if fails <= 8:
                                print(f"  [KERNEL NOT TOP-TIER] m={m} n={n} k={k} "
                                      f"align={align} ws={ws} -> {chosen_id} "
                                      f"(var {chosen['variant']}) want {want['id']}")
    ok = fails == 0
    print(f"Kernel policy:   {total} cases -> {'ALL CONFORMANT' if ok else f'{fails} VIOLATIONS'}")
    return ok


# --------------------------------------------------------------------------
# Protocol smoke test: exactly one JSON object on stdout, allowed keys only
# --------------------------------------------------------------------------
def _run_agent(script, request):
    p = subprocess.run([sys.executable, os.path.join(AGENTS, script)],
                       input=json.dumps(request), capture_output=True, text=True, timeout=5)
    return p


def test_protocol():
    ok = True
    dma_reqs = [{"case_id": 1, "direction": "h2d", "bytes": 4096, "alignment": 64,
                 "registered": True, "concurrency": 2},
                {"case_id": 2, "direction": "d2h", "bytes": 100000000, "alignment": 1,
                 "registered": False, "concurrency": 1}]
    for r in dma_reqs:
        p = _run_agent("dma_agent.py", r)
        try:
            obj = json.loads(p.stdout)
            assert p.returncode == 0 and p.stderr == "", (p.returncode, p.stderr[:200])
            assert set(obj) == {"channel", "chunk_bytes", "queue_depth", "use_zero_copy"}
            assert len(p.stdout.encode()) < 65536
        except Exception as e:
            ok = False
            print(f"  [DMA PROTOCOL] {r} -> rc={p.returncode} out={p.stdout!r} err={p.stderr[:120]!r} ({e})")
    kern_req = {"case_id": 1, "dtype": "f16", "m": 128, "n": 128, "k": 128,
                "alignment": 16, "workspace": 8192,
                "candidates": make_candidates(False, {})}
    p = _run_agent("kernel_agent.py", kern_req)
    try:
        obj = json.loads(p.stdout)
        assert p.returncode == 0 and p.stderr == "", (p.returncode, p.stderr[:200])
        assert set(obj) == {"kernel_id"}
        assert obj["kernel_id"] == "vec"      # all tiers legal here -> fastest
    except Exception as e:
        ok = False
        print(f"  [KERNEL PROTOCOL] -> rc={p.returncode} out={p.stdout!r} err={p.stderr[:120]!r} ({e})")
    print(f"Protocol smoke:  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [test_dma_optimal_over_space(),
               test_kernel_policy_over_space(),
               test_protocol()]
    print("=" * 48)
    if all(results):
        print("ALL AGENT PROPERTY TESTS PASSED")
        sys.exit(0)
    print("SOME AGENT PROPERTY TESTS FAILED")
    sys.exit(1)
