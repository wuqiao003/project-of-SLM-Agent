"""
Training profiles for different host load budgets.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_slm.finetune.trainer import TrainingConfig


def low_load_training_config(base: "TrainingConfig") -> "TrainingConfig":
    """
    Gentle profile: less VRAM burst, fewer CPU workers, optional step delay.

    Slower but keeps GPU/CPU from pegging at 100% continuously.
    """
    return replace(
        base,
        batch_size=1,
        gradient_accumulation_steps=16,
        max_seq_length=1024,
        lora_r=min(base.lora_r, 32),
        lora_alpha=min(base.lora_alpha, 64),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        step_delay_seconds=base.step_delay_seconds if base.step_delay_seconds > 0 else 0.5,
        logging_steps=5,
        save_steps=200,
        eval_steps=200,
        report_to="none",
    )


def apply_host_thread_limits(num_threads: int = 4) -> None:
    """Limit CPU threads used by PyTorch / OpenMP (reduces CPU fan spike)."""
    import os

    n = str(max(1, num_threads))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(int(n))
    except ImportError:
        pass
