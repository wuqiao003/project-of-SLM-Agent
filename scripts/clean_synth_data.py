#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean old synthetic tool-use data so that arguments are grounded in the query.

Core problem: the original generator filled string identifiers (video_url,
subtitle_file, project_id, scheduled_time, ...) from a random pool that is
independent of the user query. That teaches the model to hallucinate arguments.

This script re-grounds those identifiers: when the query contains a concrete
URL / file path / id / datetime, we overwrite the corresponding argument with
the value actually present in the query. Rows whose key identifier cannot be
grounded are dropped (configurable), because they are actively harmful.

Enum-ish fields (source_language, analysis_type, output_format, ...) are left
as-is; they are validated elsewhere and are not the source of the URL drift.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A URL or a unix-style file path ending in a known media/subtitle extension.
URL_RE = re.compile(
    r"https?://[^\s\u4e00-\u9fff，。；、,;]+"
    r"|/[^\s\u4e00-\u9fff，。；、,;]+\.(?:srt|vtt|ass|mp4|mov|webm|mkv|avi)",
    re.IGNORECASE,
)

# bare-id tokens like proj_001, sub_anime_456, final_cut_2024, file_001
ID_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")

# ISO 8601 datetime
DT_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?")

# Fields that must be grounded in the query text.
URL_FIELDS = {"video_url"}
FILE_FIELDS = {"subtitle_file"}
ID_FIELDS = {"project_id"}
DT_FIELDS = {"scheduled_time"}

# Language keyword -> ISO code. Used to re-ground language args from the query.
LANG_KEYWORDS = {
    "zh": ["中文", "中字", "汉语", "简体", "繁体", "普通话", "chinese", "mandarin"],
    "en": ["英文", "英语", "english"],
    "ja": ["日文", "日语", "日本", "japanese"],
    "ko": ["韩文", "韩语", "韩国", "korean"],
    "es": ["西班牙", "spanish", "espanol", "español"],
    "fr": ["法文", "法语", "french"],
    "de": ["德文", "德语", "german"],
    "pt": ["葡萄牙", "portuguese"],
    "th": ["泰文", "泰语", "thai"],
    "ru": ["俄文", "俄语", "russian"],
    "it": ["意大利", "italian"],
    "ar": ["阿拉伯", "arabic"],
}


def detect_languages(text: str) -> list[str]:
    """Return language codes mentioned in text, ordered by first appearance."""
    low = text.lower()
    hits: list[tuple[int, str]] = []
    for code, kws in LANG_KEYWORDS.items():
        pos = min((low.find(k.lower()) for k in kws if low.find(k.lower()) >= 0), default=-1)
        if pos >= 0:
            hits.append((pos, code))
    hits.sort()
    return [c for _, c in hits]


def ground_languages(query: str, name: str, args: dict) -> dict:
    """
    Re-ground language args from the query using positional heuristics.

    Only mutates a field when the query actually mentions language(s); when the
    query is silent we keep the existing value (no guessing, no extra drops).
    """
    langs = detect_languages(query)
    if not langs:
        return args
    new = dict(args)

    if name == "generate_subtitles" and "source_language" in new:
        new["source_language"] = langs[0]
    elif name == "generate_dubbing" and "target_language" in new:
        new["target_language"] = langs[-1]
    elif name == "translate_subtitles":
        if "target_language" in new:
            new["target_language"] = langs[-1]
        if "source_language" in new:
            # source only knowable when two distinct languages are stated
            if len(langs) >= 2 and langs[0] != langs[-1]:
                new["source_language"] = langs[0]
            else:
                new["source_language"] = "auto"
    elif name == "list_voices" and "language" in new:
        new["language"] = langs[0]

    return new


def _first(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    return m.group(0) if m else None


def _all(pattern: re.Pattern, text: str) -> list[str]:
    return pattern.findall(text)


def ground_arguments(query: str, name: str, args: dict) -> tuple[dict, list[str]]:
    """
    Re-ground identifier args from the query.

    Returns (new_args, problems). problems is non-empty when a required-style
    identifier present in args cannot be found in the query.
    """
    new_args = dict(args)
    problems: list[str] = []

    url_in_q = _first(URL_RE, query)

    for field in URL_FIELDS:
        if field in new_args and isinstance(new_args[field], str) and new_args[field]:
            if new_args[field] in query:
                continue
            if url_in_q:
                new_args[field] = url_in_q
            else:
                problems.append(f"{field}: no url in query")

    for field in FILE_FIELDS:
        if field in new_args and isinstance(new_args[field], str) and new_args[field]:
            val = new_args[field]
            if val in query:
                continue
            # try a path-like or id-like token from query
            cand = _first(URL_RE, query)
            if cand is None:
                # subtitle file might be referenced as an id (sub_xxx)
                ids = [t for t in _all(ID_RE, query)]
                cand = ids[0] if ids else None
            if cand:
                new_args[field] = cand
            else:
                problems.append(f"{field}: no file/id in query")

    for field in ID_FIELDS:
        if field in new_args and isinstance(new_args[field], str) and new_args[field]:
            if new_args[field] in query:
                continue
            ids = _all(ID_RE, query)
            if ids:
                new_args[field] = ids[0]
            else:
                problems.append(f"{field}: no id in query")

    for field in DT_FIELDS:
        if field in new_args and isinstance(new_args[field], str) and new_args[field]:
            if new_args[field] in query:
                continue
            dt = _first(DT_RE, query)
            if dt:
                new_args[field] = dt
            else:
                problems.append(f"{field}: no datetime in query")

    return new_args, problems


def clean_file(
    in_path: Path,
    out_path: Path,
    drop_ungroundable: bool = True,
) -> dict:
    stats = {
        "total": 0,
        "unparsable": 0,
        "modified": 0,
        "dropped": 0,
        "kept": 0,
        "fields_fixed": {},
    }
    out_rows = []

    for line in open(in_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        stats["total"] += 1
        row = json.loads(line)
        msgs = row.get("messages", [])
        query = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        try:
            call = json.loads(msgs[-1]["content"])
            name = call["name"]
            args = call.get("arguments", {})
        except (json.JSONDecodeError, KeyError, IndexError):
            stats["unparsable"] += 1
            if not drop_ungroundable:
                out_rows.append(row)
                stats["kept"] += 1
            else:
                stats["dropped"] += 1
            continue

        new_args, problems = ground_arguments(query, name, args)
        new_args = ground_languages(query, name, new_args)

        if problems and drop_ungroundable:
            stats["dropped"] += 1
            continue

        if new_args != args:
            stats["modified"] += 1
            for f in new_args:
                if args.get(f) != new_args.get(f):
                    stats["fields_fixed"][f] = stats["fields_fixed"].get(f, 0) + 1
            call["arguments"] = new_args
            msgs[-1]["content"] = json.dumps(call, ensure_ascii=False)

        out_rows.append(row)
        stats["kept"] += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/prepared/tool_use_train.jsonl")
    ap.add_argument("--out", dest="out", default="data/prepared/tool_use_train.clean.jsonl")
    ap.add_argument(
        "--keep-ungroundable",
        action="store_true",
        help="keep rows whose identifiers cannot be grounded (default: drop)",
    )
    args = ap.parse_args()

    in_path = ROOT / args.inp
    out_path = ROOT / args.out
    stats = clean_file(in_path, out_path, drop_ungroundable=not args.keep_ungroundable)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nclean output -> {out_path}")


if __name__ == "__main__":
    main()
