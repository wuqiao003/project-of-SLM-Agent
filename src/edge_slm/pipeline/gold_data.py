"""
Generate high-quality simple-task gold examples for acceptance testing.
"""

import json
from pathlib import Path

from edge_slm.data.schema import LIGHT_ON_TOOLS, get_tool_by_name
from edge_slm.pipeline.prompts import build_tool_use_messages


# Per-tool simple queries with explicit arguments in the user text.
GOLD_CASES: list[dict] = [
    {
        "query": "解析视频 https://example.com/demo.mp4",
        "name": "parse_video",
        "arguments": {"video_url": "https://example.com/demo.mp4"},
    },
    {
        "query": "分析 https://cdn.test.com/v1.mp4 这个视频",
        "name": "parse_video",
        "arguments": {"video_url": "https://cdn.test.com/v1.mp4"},
    },
    {
        "query": "给 https://example.com/lecture.mp4 生成中文字幕",
        "name": "generate_subtitles",
        "arguments": {"video_url": "https://example.com/lecture.mp4", "source_language": "zh"},
    },
    {
        "query": "为视频 https://test.com/movie.mp4 创建英文字幕",
        "name": "generate_subtitles",
        "arguments": {"video_url": "https://test.com/movie.mp4", "source_language": "en"},
    },
    {
        "query": "把字幕文件 /subs/meeting.srt 翻译成日文",
        "name": "translate_subtitles",
        "arguments": {
            "subtitle_file": "/subs/meeting.srt",
            "source_language": "zh",
            "target_language": "ja",
        },
    },
    {
        "query": "翻译字幕 /data/captions.vtt 从英文到中文",
        "name": "translate_subtitles",
        "arguments": {
            "subtitle_file": "/data/captions.vtt",
            "source_language": "en",
            "target_language": "zh",
        },
    },
    {
        "query": "给 https://example.com/promo.mp4 配上中文配音，字幕在 /subs/promo.srt",
        "name": "generate_dubbing",
        "arguments": {
            "video_url": "https://example.com/promo.mp4",
            "subtitle_file": "/subs/promo.srt",
            "voice_id": "voice_zh_female_01",
            "target_language": "zh",
        },
    },
    {
        "query": "分析视频 https://example.com/v.mp4 的内容，要全部维度",
        "name": "analyze_content",
        "arguments": {"video_url": "https://example.com/v.mp4", "analysis_type": "all"},
    },
    {
        "query": "安排在 2026-06-10T10:00:00 执行字幕任务",
        "name": "schedule_task",
        "arguments": {
            "task_type": "subtitle",
            "task_params": {"video_url": "https://example.com/scheduled.mp4"},
            "scheduled_time": "2026-06-10T10:00:00",
        },
    },
    {
        "query": "导出项目 proj_2024 为 mp4 1080p",
        "name": "export_project",
        "arguments": {
            "project_id": "proj_2024",
            "output_format": "mp4",
            "quality": "1080p",
        },
    },
    {
        "query": "列出中文女声配音",
        "name": "list_voices",
        "arguments": {"language": "zh", "gender": "female"},
    },
]


def _expand_variants(base_cases: list[dict]) -> list[dict]:
    """Add phrasing variants for higher coverage."""
    variants = []
    suffixes = [
        ("", {}),
        ("，谢谢", {}),
        (" Please help.", {}),
    ]
    for case in base_cases:
        for suffix, _ in suffixes:
            row = dict(case)
            row["query"] = case["query"] + suffix
            variants.append(row)
    return variants


def generate_simple_gold_dataset(
    output_path: str | Path,
    include_variants: bool = True,
) -> Path:
    """
    Write acceptance gold set as JSONL with messages + expected tool call.

    Each line:
      messages, expected: {name, arguments}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = list(GOLD_CASES)
    if include_variants:
        cases = _expand_variants(cases)

    lines = []
    for case in cases:
        tool = get_tool_by_name(case["name"])
        if tool is None:
            continue
        call = {"name": case["name"], "arguments": case["arguments"]}
        example = {
            "messages": build_tool_use_messages(
                case["query"],
                json.dumps(call, ensure_ascii=False),
            ),
            "expected": call,
            "complexity": "simple",
            "source": "gold",
        }
        lines.append(example)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path
