"""
End-to-end pipeline: data preparation, training helpers, acceptance gates.
"""

from edge_slm.pipeline.prompts import build_tool_use_messages, build_inference_prompt
from edge_slm.pipeline.gold_data import generate_simple_gold_dataset
from edge_slm.pipeline.prepare import prepare_dataset_from_jsonl, validate_example
from edge_slm.pipeline.acceptance import run_acceptance_gate, AcceptanceConfig, AcceptanceReport

__all__ = [
    "build_tool_use_messages",
    "build_inference_prompt",
    "generate_simple_gold_dataset",
    "prepare_dataset_from_jsonl",
    "validate_example",
    "run_acceptance_gate",
    "AcceptanceConfig",
    "AcceptanceReport",
]
