"""
Edge-side SLM Fine-tuning & Structured Inference Optimization for Agents
=========================================================================

A lightweight framework for deploying fine-tuned small language models (SLMs)
on consumer-grade GPUs for agent tool-calling and intent recognition.

Core Components:
- data: Dataset construction and GPT-4 distillation
- finetune: QLoRA fine-tuning with Unsloth
- inference: Structured inference with grammar constraints
- agent: Routing and orchestration layer
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from edge_slm.config import EdgeSLMConfig

__all__ = ["EdgeSLMConfig", "__version__"]
