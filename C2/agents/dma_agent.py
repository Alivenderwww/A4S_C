#!/usr/bin/env python3
"""Legal baseline DMA policy stub."""

import json
import sys


json.load(sys.stdin)
json.dump({"channel": 0, "chunk_bytes": 4096, "queue_depth": 1,
           "use_zero_copy": False}, sys.stdout, sort_keys=True)
sys.stdout.write("\n")
