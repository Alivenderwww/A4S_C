"""Unit tests for the two C2 agents' decide() functions (pure logic, no device)."""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
from dma_agent import decide as dma_decide


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


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)


if __name__ == "__main__":
    _run_all()
    print("OK")
