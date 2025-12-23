"""
Dataset module for loading and processing Tool-Use training data.
Provides PyTorch Dataset and HuggingFace Dataset compatibility.
"""

import json
import random
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

from edge_slm.data.schema import ToolUseExample, ToolDefinition, ToolCall, ToolCategory


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    max_seq_length: int = 2048
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    shuffle: bool = True
    seed: int = 42


class ToolUseDataset:
    """
    Dataset class for Tool-Use training data.
    Handles loading, processing, and formatting for different training frameworks.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[DatasetConfig] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config or DatasetConfig()
        self.examples: list[dict] = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from JSONL or JSON file."""
        if self.data_path.suffix == ".jsonl":
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.examples = [json.loads(line) for line in f if line.strip()]
        elif self.data_path.suffix == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.examples = data
                else:
                    self.examples = [data]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        if self.config.shuffle:
            random.seed(self.config.seed)
            random.shuffle(self.examples)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]
    
    def format_for_training(
        self,
        example: dict,
        chat_template: Optional[str] = None,
    ) -> str:
        """
        Format a single example for training.
        
        Args:
            example: Training example with 'messages' field
            chat_template: Optional custom chat template
        
        Returns:
            Formatted string ready for tokenization
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for formatting")
        
        messages = example.get("messages", [])
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        
        # Fallback: simple formatting
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|{role}|>\n{content}\n"
        
        return formatted
    
    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        return Dataset.from_list(self.examples)
    
    def split(self) -> DatasetDict:
        """
        Split dataset into train/val/test sets.
        
        Returns:
            DatasetDict with 'train', 'validation', 'test' splits
        """
        n = len(self.examples)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        return DatasetDict({
            "train": Dataset.from_list(self.examples[:train_end]),
            "validation": Dataset.from_list(self.examples[train_end:val_end]),
            "test": Dataset.from_list(self.examples[val_end:]),
        })
    
    def prepare_for_sft(
        self,
        add_eos: bool = True,
    ) -> Dataset:
        """
        Prepare dataset for Supervised Fine-Tuning (SFT).
        
        Returns:
            Dataset with 'text' field containing formatted conversations
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for SFT preparation")
        
        def format_example(example):
            text = self.format_for_training(example)
            if add_eos and self.tokenizer.eos_token:
                text += self.tokenizer.eos_token
            return {"text": text}
        
        dataset = self.to_hf_dataset()
        return dataset.map(format_example)


def create_dataset(
    data_path: Union[str, Path],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_seq_length: int = 2048,
    split: bool = True,
) -> Union[Dataset, DatasetDict]:
    """
    Convenience function to create a dataset from a file.
    
    Args:
        data_path: Path to JSONL or JSON file
        tokenizer: Optional tokenizer for formatting
        max_seq_length: Maximum sequence length
        split: Whether to split into train/val/test
    
    Returns:
        Dataset or DatasetDict
    """
    config = DatasetConfig(max_seq_length=max_seq_length)
    dataset = ToolUseDataset(data_path, tokenizer, config)
    
    if split:
        return dataset.split()
    return dataset.to_hf_dataset()


def create_chat_format_dataset(
    examples: list[ToolUseExample],
    include_reasoning: bool = False,
) -> Dataset:
    """
    Create a HuggingFace Dataset from ToolUseExample objects.
    
    Args:
        examples: List of ToolUseExample objects
        include_reasoning: Whether to include chain-of-thought reasoning
    
    Returns:
        HuggingFace Dataset
    """
    data = [
        example.to_training_format(include_reasoning=include_reasoning)
        for example in examples
    ]
    return Dataset.from_list(data)


# =============================================================================
# Prompt Templates for Different Models
# =============================================================================

QWEN_TOOL_PROMPT = """<|im_start|>system
You are a helpful AI assistant with access to the following tools. Use them to help the user.

Tools:
{tools}

When you need to use a tool, respond with a JSON object in this exact format:
{{"name": "tool_name", "arguments": {{"param1": "value1", ...}}}}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

PHI_TOOL_PROMPT = """<|system|>
You are a helpful AI assistant. You have access to the following tools:

{tools}

To use a tool, respond with JSON: {{"name": "tool_name", "arguments": {{...}}}}
<|end|>
<|user|>
{query}<|end|>
<|assistant|>
"""


def get_prompt_template(model_type: str) -> str:
    """Get the appropriate prompt template for a model type."""
    templates = {
        "qwen": QWEN_TOOL_PROMPT,
        "phi": PHI_TOOL_PROMPT,
    }
    return templates.get(model_type, QWEN_TOOL_PROMPT)


def format_tools_for_prompt(tools: list[ToolDefinition]) -> str:
    """Format tool definitions for inclusion in prompts."""
    formatted = []
    for tool in tools:
        params = ", ".join([
            f"{p.name}: {p.type}" + ("" if p.required else " (optional)")
            for p in tool.parameters
        ])
        formatted.append(f"- {tool.name}({params}): {tool.description}")
    return "\n".join(formatted)
