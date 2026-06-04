#!/usr/bin/env python
"""
Bootstrap Edge SLM Agent for local / cluster training.

Usage:
    python scripts/land_project.py              # full bootstrap
    python scripts/land_project.py --step env   # environment check only
    python scripts/land_project.py --step data  # generate gold + synthetic data
    python scripts/land_project.py --step test  # run pytest subset
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def check_env() -> bool:
    from rich.console import Console

    console = Console()
    console.print("[bold]1. Environment check[/bold]")

    ok = True
    try:
        import torch

        cuda = torch.cuda.is_available()
        console.print(f"  PyTorch: {torch.__version__}, CUDA: {cuda}")
        if cuda:
            console.print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")
        ok = False

    for pkg, label in [
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("datasets", "Datasets"),
        ("typer", "Typer CLI"),
    ]:
        try:
            __import__(pkg)
            console.print(f"  {label}: OK")
        except ImportError:
            console.print(f"  [yellow]{label}: missing[/yellow]")
            if pkg in ("transformers", "peft", "datasets"):
                ok = False

    return ok


def step_data(samples: int = 2000) -> None:
    from rich.console import Console
    from edge_slm.pipeline.gold_data import generate_simple_gold_dataset
    from edge_slm.pipeline.prepare import prepare_dataset_from_jsonl

    console = Console()
    console.print("[bold]2. Generate datasets[/bold]")

    gold_path = ROOT / "data/acceptance/simple_gold.jsonl"
    generate_simple_gold_dataset(gold_path, include_variants=True)
    console.print(f"  Gold acceptance set: {gold_path}")

    # Synthetic train pool (no API)
    sys.path.insert(0, str(ROOT))
    from generate_and_train import generate_dataset

    raw_path = ROOT / "data/raw/synthetic_train.jsonl"
    generate_dataset(samples, str(raw_path))

    prepared_dir = ROOT / "data/prepared"
    paths = prepare_dataset_from_jsonl(
        raw_path,
        prepared_dir,
        gold_path=gold_path,
        gold_oversample=3,
    )
    for name, p in paths.items():
        console.print(f"  {name}: {p}")

    # Minimal sample for README quick start
    sample_path = ROOT / "data/sample_train.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(paths["train"], encoding="utf-8") as src, open(sample_path, "w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= 200:
                break
            dst.write(line)
    console.print(f"  Quick-start sample (200 lines): {sample_path}")


def step_test() -> int:
    from rich.console import Console

    console = Console()
    console.print("[bold]3. Run tests[/bold]")
    cmd = [sys.executable, "-m", "pytest", "tests/test_acceptance_gold.py", "tests/test_router.py", "-q"]
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def print_next_steps() -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    text = """
[bold]Next steps[/bold]

[cyan]Local distill (Ollama 14B):[/cyan]
  python run_distill_2000.py
  python run.py distill --local --num-samples 5000 --backend ollama

[cyan]Train (Windows uses standard QLoRA):[/cyan]
  python run.py train data/prepared/tool_use_train.jsonl --output-dir outputs/tool_use_v1 --epochs 3

[cyan]Or one-shot synthetic + train:[/cyan]
  python generate_and_train.py --samples 2000 --epochs 3

[cyan]Export & accept:[/cyan]
  python run.py export outputs/tool_use_v1/final_adapter outputs/tool_use_v1_merged --format merged
  python run.py accept outputs/tool_use_v1_merged --config configs/acceptance.json

[cyan]Benchmark:[/cyan]
  python run.py benchmark outputs/tool_use_v1_merged --test-data data/prepared/tool_use_test.jsonl
"""
    console.print(Panel(text, title="Landing complete", border_style="green"))


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Edge SLM Agent")
    parser.add_argument(
        "--step",
        choices=["all", "env", "data", "test"],
        default="all",
    )
    parser.add_argument("--samples", type=int, default=2000, help="Synthetic training samples")
    args = parser.parse_args()

    if args.step in ("all", "env"):
        if not check_env():
            print("Fix dependencies: pip install -r requirements.txt")
            sys.exit(1)

    if args.step in ("all", "data"):
        step_data(args.samples)

    if args.step in ("all", "test"):
        code = step_test()
        if code != 0:
            sys.exit(code)

    if args.step == "all":
        print_next_steps()


if __name__ == "__main__":
    main()
