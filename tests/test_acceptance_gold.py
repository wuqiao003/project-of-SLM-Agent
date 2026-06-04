"""Tests for gold dataset and acceptance metrics."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from edge_slm.evaluation.metrics import compute_metrics
from edge_slm.pipeline.gold_data import generate_simple_gold_dataset
from edge_slm.pipeline.prepare import validate_example
from edge_slm.pipeline.acceptance import parse_tool_call_from_text


def test_gold_generation(tmp_path):
    out = tmp_path / "gold.jsonl"
    path = generate_simple_gold_dataset(out, include_variants=False)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= len(__import__("edge_slm.pipeline.gold_data", fromlist=["GOLD_CASES"]).GOLD_CASES)

    for line in lines:
        row = json.loads(line)
        ok, msg = validate_example(row)
        assert ok, msg
        assert "expected" in row
        assert row["expected"]["name"]


def test_perfect_predictions_score_100(tmp_path):
    out = tmp_path / "gold.jsonl"
    generate_simple_gold_dataset(out, include_variants=False)
    refs = []
    preds = []
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        refs.append(row["expected"])
        preds.append(row["expected"])

    m = compute_metrics(preds, refs)
    assert m.exact_match_accuracy == 1.0
    assert m.tool_selection_accuracy == 1.0


def test_parse_legacy_tool_format():
    text = '{"tool": "parse_video", "params": {"video_url": "https://a.com/x.mp4"}}'
    parsed = parse_tool_call_from_text(text)
    assert parsed["name"] == "parse_video"
    assert parsed["arguments"]["video_url"] == "https://a.com/x.mp4"
