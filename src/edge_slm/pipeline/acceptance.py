"""
Acceptance gate runner for release-quality checks on simple tasks.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from edge_slm.evaluation.metrics import ToolUseMetrics, compute_metrics
from edge_slm.pipeline.prompts import build_inference_prompt


@dataclass
class AcceptanceThresholds:
    tool_selection_min: float = 0.98
    exact_match_min: float = 0.95
    argument_accuracy_min: float = 0.95
    json_validity_min: float = 0.99
    simple_exact_match_min: float = 1.0  # first N simple cases in evaluate script


@dataclass
class AcceptanceConfig:
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)
    gold_data_path: str = "data/acceptance/simple_gold.jsonl"
    held_out_test_path: Optional[str] = None


@dataclass
class AcceptanceReport:
    passed: bool
    metrics: ToolUseMetrics
    failures: list[str] = field(default_factory=list)
    dataset: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "dataset": self.dataset,
            "failures": self.failures,
            "metrics": {
                "tool_selection_accuracy": self.metrics.tool_selection_accuracy,
                "argument_accuracy": self.metrics.argument_accuracy,
                "exact_match_accuracy": self.metrics.exact_match_accuracy,
                "json_validity_rate": self.metrics.json_validity_rate,
                "schema_compliance_rate": self.metrics.schema_compliance_rate,
            },
        }


def load_acceptance_config(path: str | Path) -> AcceptanceConfig:
    path = Path(path)
    if not path.exists():
        return AcceptanceConfig()

    with open(path, encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                raw = yaml.safe_load(f) or {}
            except ImportError:
                raise ImportError("Install PyYAML for YAML configs: pip install pyyaml")
        else:
            raw = json.load(f)

    th = raw.get("thresholds", {})
    thresholds = AcceptanceThresholds(
        tool_selection_min=th.get("tool_selection_min", 0.98),
        exact_match_min=th.get("exact_match_min", 0.95),
        argument_accuracy_min=th.get("argument_accuracy_min", 0.95),
        json_validity_min=th.get("json_validity_min", 0.99),
        simple_exact_match_min=th.get("simple_exact_match_min", 1.0),
    )
    return AcceptanceConfig(
        thresholds=thresholds,
        gold_data_path=raw.get("gold_data_path", "data/acceptance/simple_gold.jsonl"),
        held_out_test_path=raw.get("held_out_test_path"),
    )


def _load_gold_references(path: Path) -> tuple[list[dict], list[dict]]:
    """Load gold file: each line has expected + optional messages."""
    predictions_placeholder = []
    references = []
    samples = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            references.append(row["expected"])
            samples.append(row)

    return samples, references


def parse_tool_call_from_text(text: str) -> dict:
    """Parse model output; supports name/arguments and legacy tool/params."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return {}

    data = json.loads(text[start:end])

    if "name" in data and "arguments" in data:
        return data
    if "tool" in data:
        return {
            "name": data["tool"],
            "arguments": data.get("params") or data.get("arguments") or {},
        }
    return {}


def run_acceptance_on_predictions(
    predictions: list[dict],
    references: list[dict],
    config: AcceptanceConfig,
    dataset_name: str,
    latencies: Optional[list[float]] = None,
) -> AcceptanceReport:
    metrics = compute_metrics(predictions, references, latencies)
    failures = []
    th = config.thresholds

    if metrics.tool_selection_accuracy < th.tool_selection_min:
        failures.append(
            f"tool_selection {metrics.tool_selection_accuracy:.1%} < {th.tool_selection_min:.1%}"
        )
    if metrics.exact_match_accuracy < th.exact_match_min:
        failures.append(
            f"exact_match {metrics.exact_match_accuracy:.1%} < {th.exact_match_min:.1%}"
        )
    if metrics.argument_accuracy < th.argument_accuracy_min:
        failures.append(
            f"argument_accuracy {metrics.argument_accuracy:.1%} < {th.argument_accuracy_min:.1%}"
        )
    if metrics.json_validity_rate < th.json_validity_min:
        failures.append(
            f"json_validity {metrics.json_validity_rate:.1%} < {th.json_validity_min:.1%}"
        )

    return AcceptanceReport(
        passed=len(failures) == 0,
        metrics=metrics,
        failures=failures,
        dataset=dataset_name,
    )


def run_acceptance_gate(
    model_path: str,
    config_path: str = "configs/acceptance.json",
    *,
    use_structured_decoding: bool = True,
) -> AcceptanceReport:
    """
    Run acceptance on gold set using the project's inference engine.
    """
    from edge_slm.inference import create_engine
    from edge_slm.data.schema import LIGHT_ON_TOOLS

    config = load_acceptance_config(config_path)
    gold_path = Path(config.gold_data_path)
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold data not found: {gold_path}. Run: python scripts/land_project.py --step data"
        )

    samples, references = _load_gold_references(gold_path)
    tools = [t.to_openai_format() for t in LIGHT_ON_TOOLS]

    engine = create_engine(model_path, use_structured_decoding=use_structured_decoding)
    engine.load_model()

    predictions = []
    latencies = []

    for sample in samples:
        query = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        prompt = build_inference_prompt(query)

        result = engine.generate(prompt, tools=tools)
        latencies.append(result.latency_ms)

        if result.is_valid and result.parsed:
            predictions.append(result.parsed)
        else:
            try:
                predictions.append(parse_tool_call_from_text(result.text or ""))
            except json.JSONDecodeError:
                predictions.append({})

    report = run_acceptance_on_predictions(
        predictions, references, config, dataset_name="simple_gold", latencies=latencies
    )
    return report


def save_acceptance_report(report: AcceptanceReport, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "acceptance_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    return out
