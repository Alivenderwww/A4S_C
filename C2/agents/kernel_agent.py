#!/usr/bin/env python3
"""Legal baseline AEC-image policy stub."""

import json
import sys


request = json.load(sys.stdin)
candidate_ids = {item["id"] for item in request["candidates"]}
if "naive" not in candidate_ids:
    raise SystemExit(2)
json.dump({"kernel_id": "naive"}, sys.stdout, sort_keys=True)
sys.stdout.write("\n")
