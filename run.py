#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Must run before other imports on Windows (trl reads UTF-8 .jinja files)."""
import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

"""
Simple runner script for Edge SLM Agent.
Use this instead of the 'edge-slm' command if it's not found.

Usage:
    python run.py distill --num-samples 100
    python run.py train data/sample_train.jsonl
    python run.py train data/prepared/tool_use_train.jsonl --low-load --no-use-unsloth --step-delay 0.8
    python run.py serve outputs/model
    python run.py infer outputs/model "帮我分析视频"
    python run.py demo
    python run.py land
    python run.py generate-gold
    python run.py prepare-data data/raw/synthetic_train.jsonl
    python run.py accept outputs/model
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from edge_slm.cli import app

if __name__ == "__main__":
    # Add 'demo' command for quick testing
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("Running inference demo...")
        exec(open("scripts/inference_demo.py").read())
    else:
        app()
