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


def _load_model_for_eval(model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load a LoRA adapter (4bit base + PEFT) or a full/merged model on limited VRAM."""
    import torch
    from pathlib import Path as _P
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    is_adapter = (_P(model_path) / "adapter_config.json").exists()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    if is_adapter:
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_acceptance_gate(
    model_path: str,
    config_path: str = "configs/acceptance.json",
    *,
    use_structured_decoding: bool = True,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    max_samples: Optional[int] = None,
) -> AcceptanceReport:
    """
    Run acceptance on the gold set.

    Loads a LoRA adapter (base 4bit + PEFT) or merged model, and reuses the gold
    messages (same system/user prompt as training) for faithful evaluation.
    """
    import time
    import torch

    config = load_acceptance_config(config_path)
    gold_path = Path(config.gold_data_path)
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold data not found: {gold_path}. Run: python scripts/land_project.py --step data"
        )

    samples, references = _load_gold_references(gold_path)
    if max_samples:
        samples = samples[:max_samples]
        references = references[:max_samples]

    model, tokenizer = _load_model_for_eval(model_path, base_model=base_model)

    predictions = []
    latencies = []

    for sample in samples:
        # Use the same messages as training (system + user), drop the gold assistant turn
        msgs = [m for m in sample["messages"] if m["role"] in ("system", "user")]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latencies.append((time.time() - start) * 1000)

        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        try:
            predictions.append(parse_tool_call_from_text(gen))
        except (json.JSONDecodeError, ValueError):
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
