"""
Dataset validation, cleaning, splitting, and gold-data merge.
"""

import json
import random
from pathlib import Path
from typing import Any

from edge_slm.data.schema import get_tool_by_name


def validate_example(example: dict) -> tuple[bool, str]:
    """Validate a training or gold example."""
    if "messages" not in example:
        return False, "missing messages"

    messages = example["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        return False, "last message must be assistant"

    try:
        content = messages[-1]["content"]
        # Strip optional thinking block
        if "<thinking>" in content:
            content = content.split("</thinking>")[-1].strip()
        data = json.loads(content.strip())
    except json.JSONDecodeError as e:
        return False, f"invalid assistant json: {e}"

    if "name" not in data or "arguments" not in data:
        return False, "assistant json must have name and arguments"

    tool = get_tool_by_name(data["name"])
    if tool is None:
        return False, f"unknown tool: {data['name']}"

    if not isinstance(data["arguments"], dict):
        return False, "arguments must be object"

    return True, "ok"


def _dedupe_key(example: dict) -> str:
    user = next((m["content"] for m in example["messages"] if m["role"] == "user"), "")
    assistant = example["messages"][-1]["content"]
    return f"{user.strip()}::{assistant.strip()}"


def prepare_dataset_from_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    gold_path: str | Path | None = None,
    gold_oversample: int = 3,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Load JSONL, validate, dedupe, optionally merge gold oversamples, split, save.

    Returns paths: train, val, test
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    rejected = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ok, _ = validate_example(ex)
            if ok:
                examples.append(ex)
            else:
                rejected += 1

    # Dedupe
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        key = _dedupe_key(ex)
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    examples = unique

    # Merge gold oversamples into train only (later split keeps gold in train)
    if gold_path:
        gold_path = Path(gold_path)
        if gold_path.exists():
            with open(gold_path, encoding="utf-8") as f:
                gold_rows = [json.loads(line) for line in f if line.strip()]
            for _ in range(gold_oversample):
                examples.extend(gold_rows)

    random.seed(seed)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    paths = {}
    for name, rows in splits.items():
        out = output_dir / f"tool_use_{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[name] = out

    stats_path = output_dir / "prepare_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(input_path),
                "rejected": rejected,
                "total_after_dedupe": n,
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
                "gold_oversample": gold_oversample if gold_path else 0,
            },
            f,
            indent=2,
        )

    return paths
