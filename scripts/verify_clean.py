#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify a cleaned dataset: identifier args should be present in the query."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from edge_slm.pipeline.prepare import validate_example

path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/prepared/tool_use_train.jsonl")
ID_FIELDS = {"video_url", "subtitle_file", "project_id", "scheduled_time"}

n = bad_validate = mismatch = 0
samples = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    n += 1
    row = json.loads(line)
    ok, msg = validate_example(row)
    if not ok:
        bad_validate += 1
        continue
    msgs = row["messages"]
    q = next((m["content"] for m in msgs if m["role"] == "user"), "")
    call = json.loads(msgs[-1]["content"])
    for f in ID_FIELDS:
        v = call.get("arguments", {}).get(f)
        if isinstance(v, str) and v and v not in q:
            mismatch += 1
            if len(samples) < 8:
                samples.append((f, v, q[:60]))

print(f"rows={n} invalid={bad_validate} remaining_id_mismatch={mismatch}")
for f, v, q in samples:
    print(f"  [{f}] {v!r} NOT IN  {q!r}")
