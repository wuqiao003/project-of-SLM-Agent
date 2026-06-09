#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate parse_video vs analyze_content contrastive samples.

parse_video  -> metadata intent (duration / resolution / format / fps / size)
analyze_content -> semantic intent (topics / summary / sentiment / key points)

These two tools are the dominant confusion pair in evaluation, so we sharpen
the decision boundary with explicit cue words on both sides.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM = "You are a helpful AI assistant. Respond with JSON tool calls when appropriate."

VIDEO_URLS = [
    "https://example.com/video1.mp4",
    "https://storage.example.com/uploads/meeting_2024.mp4",
    "/data/videos/tutorial.mp4",
    "https://cdn.example.com/content/lecture_01.mp4",
    "https://media.example.org/clip.mp4",
    "/videos/presentation.mp4",
    "https://youtube.com/watch?v=abc123",
    "https://bilibili.com/video/BV1xx411",
    "https://ted.com/talk999.mp4",
    "https://drama.com/ep07.mp4",
    "https://company.com/promo2024.mp4",
    "https://app.com/intro_v2.mp4",
]

# parse_video: metadata-oriented phrasings (cue: duration/resolution/format/...)
PARSE_TEMPLATES = [
    "解析视频 {url}",
    "看看 {url} 的时长和分辨率",
    "{url} 这个视频多长、什么分辨率？",
    "帮我获取 {url} 的基本信息（时长、帧率）",
    "{url} 的格式和编码是什么，帮我解析下",
    "我想知道 {url} 的文件大小和时长",
    "先解析一下 {url} 的元信息",
    "查一下 {url} 的分辨率和码率",
    "Parse the video {url} and show its metadata",
    "Get duration and resolution of {url}",
    "这个视频 {url} 的基本参数帮我读出来",
    "录了个视频 {url}，先看看时长多少",
]

# analyze_content: semantic-oriented phrasings (cue: topics/summary/sentiment/...)
ANALYZE_TEMPLATES = [
    ("分析一下视频 {url} 的内容", "all"),
    ("帮我总结 {url} 这个视频讲了什么", "summary"),
    ("提取 {url} 的主题和关键点", "topics"),
    ("{url} 这个视频的情感倾向是怎样的", "sentiment"),
    ("看看 {url} 里面到底讲了些什么内容", "summary"),
    ("帮我分析 {url} 的核心要点和主题", "topics"),
    ("{url} 观众反馈情绪如何，分析下内容情感", "sentiment"),
    ("Analyze the content of {url}", "all"),
    ("Summarize what {url} is about", "summary"),
    ("我想了解 {url} 讲了哪些主题", "topics"),
    ("这个视频 {url} 想表达什么，帮我总结", "summary"),
    ("分析 {url} 的内容结构和关键时刻", "all"),
]


def _row(query: str, name: str, arguments: dict) -> dict:
    call = {"name": name, "arguments": arguments}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(call, ensure_ascii=False)},
        ]
    }


def generate(n_each: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    rows = []

    for _ in range(n_each):
        url = random.choice(VIDEO_URLS)
        tmpl = random.choice(PARSE_TEMPLATES)
        rows.append(_row(tmpl.format(url=url), "parse_video", {"video_url": url}))

    for _ in range(n_each):
        url = random.choice(VIDEO_URLS)
        tmpl, atype = random.choice(ANALYZE_TEMPLATES)
        rows.append(
            _row(
                tmpl.format(url=url),
                "analyze_content",
                {"video_url": url, "analysis_type": atype},
            )
        )

    random.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-each", type=int, default=250, help="samples per tool")
    ap.add_argument("--out", type=str, default="data/contrastive_parse_analyze.jsonl")
    ap.add_argument(
        "--append-to",
        type=str,
        default="data/prepared/tool_use_train.jsonl",
        help="training file to append into (set empty to skip)",
    )
    args = ap.parse_args()

    rows = generate(args.n_each)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} contrastive samples -> {out}")

    if args.append_to:
        target = ROOT / args.append_to
        before = sum(1 for _ in open(target, encoding="utf-8")) if target.exists() else 0
        with open(target, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        after = sum(1 for _ in open(target, encoding="utf-8"))
        print(f"appended to {target}: {before} -> {after}")


if __name__ == "__main__":
    main()
