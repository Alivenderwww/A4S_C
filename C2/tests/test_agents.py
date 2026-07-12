"""Unit tests for the two C2 agents' decide() functions (pure logic, no device)."""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
from dma_agent import decide as dma_decide
from kernel_agent import decide as kernel_decide


# --- DMA cycle formula (mirrors grader _dma_cycles, doc 05 sec 3) ---
def _ceil(a, b):
    return (a + b - 1) // b


def _dma_cycles(request, action):
    chunk = action["chunk_bytes"]
    depth = action["queue_depth"]
    zero_copy = action["use_zero_copy"]
    chunks = _ceil(request["bytes"], chunk)
    payload = _ceil(request["bytes"], 32)
    parallelism = min(depth, request["concurrency"], 2)
    setup = 45 if zero_copy else 100
    penalty = 0 if request["alignment"] >= 64 else 13
    return setup + _ceil(payload, parallelism) + 24 * (chunks - 1) + penalty


def _all_legal_actions(request):
    zc_opts = [True, False] if request["registered"] else [False]
    for ch, ck, dp, zc in itertools.product((0, 1),
                                            (4096, 65536, 1048576),
                                            (1, 2, 4, 8), zc_opts):
        yield {"channel": ch, "chunk_bytes": ck, "queue_depth": dp, "use_zero_copy": zc}


DMA_REQUESTS = [
    {"bytes": 1024,   "alignment": 64,  "registered": False, "concurrency": 1},
    {"bytes": 65536,  "alignment": 64,  "registered": True,  "concurrency": 4},
    {"bytes": 1048576,"alignment": 16,  "registered": False, "concurrency": 2},
    {"bytes": 4096,   "alignment": 32,  "registered": True,  "concurrency": 1},
    {"bytes": 200000, "alignment": 128, "registered": False, "concurrency": 8},
]


def test_dma_output_legal_and_keys_exact():
    for req in DMA_REQUESTS:
        action = dma_decide(req)
        assert set(action) == {"channel", "chunk_bytes", "queue_depth", "use_zero_copy"}
        assert action["channel"] in (0, 1)
        assert action["chunk_bytes"] in (4096, 65536, 1048576)
        assert action["queue_depth"] in (1, 2, 4, 8)
        assert isinstance(action["use_zero_copy"], bool)
        assert not (action["use_zero_copy"] and not req["registered"])


def test_dma_optimal_against_exhaustive_search():
    """decide() must tie the best legal action on every representative request."""
    for req in DMA_REQUESTS:
        action = dma_decide(req)
        best = min(_dma_cycles(req, a) for a in _all_legal_actions(req))
        assert _dma_cycles(req, action) == best, (req, action, best)


# --- Kernel agent ---
def _kernel_candidates():
    return [
        {"id": "naive", "semantic_kernel_id": 10, "image_id": 0, "variant": 1,
         "workspace": 0, "alignment": 1, "divisibility": 1},
        {"id": "tiled", "semantic_kernel_id": 11, "image_id": 0, "variant": 2,
         "workspace": 4096, "alignment": 1, "divisibility": 4},
        {"id": "vectorized", "semantic_kernel_id": 12, "image_id": 0, "variant": 3,
         "workspace": 8192, "alignment": 16, "divisibility": 8},
    ]


def _krequest(m, n, k, alignment, workspace, inject=None):
    cands = _kernel_candidates()
    if inject is not None:
        cands = [dict(c, diagnostic_cycles=inject[c["id"]]) if c["id"] in inject else dict(c)
                 for c in cands]
    return {"case_id": 0, "dtype": "fp32", "m": m, "n": n, "k": k,
            "alignment": alignment, "workspace": workspace, "candidates": cands}


def test_kernel_diagnostic_picks_min_cycles():
    # vec legal (32/64/16 all %8==0); injected cycles: vec smallest
    req = _krequest(32, 64, 16, 64, 8192,
                    inject={"naive": 1115, "tiled": 731, "vectorized": 603})
    assert kernel_decide(req)["kernel_id"] == "vectorized"


def test_kernel_diagnostic_ignores_illegal_candidate():
    # vec illegal (20%8!=0); even with a fake tiny cycle it must not be picked
    req = _krequest(20, 12, 28, 16, 4096,
                    inject={"naive": 543, "tiled": 351, "vectorized": 1})
    assert kernel_decide(req)["kernel_id"] == "tiled"


def test_kernel_hidden_picks_highest_legal_variant():
    # no diagnostic_cycles; vec legal (64/64/64 %8==0, align16>=16, ws8192>=8192)
    req = _krequest(64, 64, 64, 16, 8192)
    assert kernel_decide(req)["kernel_id"] == "vectorized"


def test_kernel_hidden_falls_back_to_tiled_when_vec_illegal():
    # 36%8!=0 but 36%4==0; tiled legal
    req = _krequest(36, 36, 36, 16, 8192)
    assert kernel_decide(req)["kernel_id"] == "tiled"


def test_kernel_hidden_only_naive_legal():
    # 7%4!=0 -> tiled illegal; align8<16 -> vec illegal
    req = _krequest(7, 9, 5, 8, 0)
    assert kernel_decide(req)["kernel_id"] == "naive"


def test_kernel_workspace_filters_tiled_and_vec():
    # vec needs ws8192, tiled needs ws4096; only 100 available -> only naive
    req = _krequest(64, 64, 64, 16, 100)
    assert kernel_decide(req)["kernel_id"] == "naive"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)


if __name__ == "__main__":
    _run_all()
    print("OK")
