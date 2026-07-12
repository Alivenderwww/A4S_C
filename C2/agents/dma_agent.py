#!/usr/bin/env python3
"""DMA policy agent: minimizes the AEC virtual-cycle formula (doc 05 sec 3).

Formula:
  cycles = setup + ceil(ceil(bytes/32)/parallelism) + 24*(ceil(bytes/chunk)-1) + align_penalty
  setup = 45 if use_zero_copy else 100
  parallelism = min(queue_depth, concurrency, 2)
  align_penalty = 13 if alignment < 64 else 0

Each term is minimized independently. channel is not in the formula.
"""
import json
import sys


def decide(request):
    """Return the cycle-optimal legal DMA action for the request."""
    registered = bool(request["registered"])
    concurrency = int(request["concurrency"])
    return {
        "channel": 0,                                  # not in cycle formula
        "chunk_bytes": 1048576,                        # minimizes 24*(chunks-1); largest legal chunk
        "queue_depth": 2 if concurrency >= 2 else 1,   # parallelism = min(depth, conc, 2); 2 is the cap
        "use_zero_copy": registered,                   # setup 45 vs 100; legal only when registered
    }


if __name__ == "__main__":
    request = json.load(sys.stdin)
    json.dump(decide(request), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
