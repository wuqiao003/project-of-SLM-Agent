#!/usr/bin/env python
"""
Example training script for Edge SLM Agent.

This script demonstrates the complete training pipeline:
1. Generate synthetic training data (or use GPT-4 distillation)
2. Fine-tune Qwen2.5-3B with QLoRA
3. Evaluate the model
4. Export for deployment

Optimized for RTX 3060 (6GB VRAM).
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rich.console import Console

console = Console()


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """Create a sample dataset for testing."""
    import json
    from edge_slm.data.schema import LIGHT_ON_TOOLS
    
    console.print("[bold blue]Creating sample dataset...[/bold blue]")
    
    # Sample training examples
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Respond with JSON tool calls."},
                {"role": "user", "content": "帮我分析视频 https://example.com/video.mp4"},
                {"role": "assistant", "content": '{"name": "parse_video", "arguments": {"video_url": "https://example.com/video.mp4"}}'}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Respond with JSON tool calls."},
                {"role": "user", "content": "给视频生成中文字幕：https://test.com/v.mp4"},
                {"role": "assistant", "content": '{"name": "generate_subtitles", "arguments": {"video_url": "https://test.com/v.mp4", "source_language": "zh"}}'}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Respond with JSON tool calls."},
                {"role": "user", "content": "把字幕翻译成日文：/subs/video.srt"},
                {"role": "assistant", "content": '{"name": "translate_subtitles", "arguments": {"subtitle_file": "/subs/video.srt", "source_language": "zh", "target_language": "ja"}}'}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Respond with JSON tool calls."},
                {"role": "user", "content": "分析视频内容：https://cdn.example.com/lecture.mp4"},
                {"role": "assistant", "content": '{"name": "analyze_content", "arguments": {"video_url": "https://cdn.example.com/lecture.mp4", "analysis_type": "all"}}'}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Respond with JSON tool calls."},
                {"role": "user", "content": "为视频配上英文配音：https://example.com/chinese_video.mp4"},
                {"role": "assistant", "content": '{"name": "generate_dubbing", "arguments": {"video_url": "https://example.com/chinese_video.mp4", "target_language": "en"}}'}
            ]
        },
    ]
    
    # Expand dataset
    expanded = []
    while len(expanded) < num_samples:
        expanded.extend(examples)
    expanded = expanded[:num_samples]
    
    # Save as JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in expanded:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    console.print(f"[green]Created {len(expanded)} training examples at {output_path}[/green]")
    return output_path


def train_model(
    data_path: str,
    output_dir: str = "outputs/tool_use_model",
    use_unsloth: bool = True,
):
    """Train the model with QLoRA."""
    console.print("[bold blue]Starting model training...[/bold blue]")
    
    if use_unsloth:
        try:
            from edge_slm.finetune.unsloth_trainer import UnslothTrainer, UnslothConfig
            
            config = UnslothConfig(
                model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                output_dir=output_dir,
                num_epochs=3,
                batch_size=2,
                gradient_accumulation_steps=8,
                lora_r=32,
                max_seq_length=1536,
            )
            
            trainer = UnslothTrainer(config)
            trainer.load_model()
            
            from edge_slm.data import create_dataset
            dataset = create_dataset(data_path, split=True)
            
            trainer.train(
                train_dataset=dataset["train"],
                eval_dataset=dataset.get("validation"),
            )
            
            # Export for vLLM
            trainer.export_to_vllm(os.path.join(output_dir, "vllm_model"))
            
            console.print(f"[green]Training complete! Model saved to {output_dir}[/green]")
            return output_dir
            
        except ImportError:
            console.print("[yellow]Unsloth not available, using standard trainer[/yellow]")
    
    # Fallback to standard trainer
    from edge_slm.finetune.trainer import SLMTrainer, TrainingConfig
    from edge_slm.data import create_dataset
    
    config = TrainingConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir=output_dir,
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=8,
        lora_r=32,
        max_seq_length=1536,
    )
    
    trainer = SLMTrainer(config)
    trainer.load_model()
    
    dataset = create_dataset(data_path, trainer.tokenizer, split=True)
    
    def format_for_sft(example):
        messages = example.get("messages", [])
        text = trainer.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    
    train_data = dataset["train"].map(format_for_sft)
    eval_data = dataset["validation"].map(format_for_sft) if "validation" in dataset else None
    
    trainer.train(train_data, eval_data)
    
    console.print(f"[green]Training complete! Model saved to {output_dir}[/green]")
    return output_dir


def evaluate_model(model_path: str):
    """Evaluate the trained model."""
    import asyncio
    from edge_slm.evaluation.benchmark import run_benchmark
    
    console.print("[bold blue]Evaluating model...[/bold blue]")
    
    results = asyncio.run(run_benchmark(
        local_model_path=model_path,
        num_samples=50,
        compare_cloud=False,
    ))
    
    return results


def main():
    """Main training pipeline."""
    console.print("[bold magenta]Edge SLM Agent Training Pipeline[/bold magenta]")
    console.print("=" * 50)
    
    # Step 1: Create sample dataset
    data_path = create_sample_dataset(
        "data/sample_train.jsonl",
        num_samples=100,
    )
    
    # Step 2: Train model
    # Note: Uncomment to actually train (requires GPU)
    # model_path = train_model(str(data_path))
    
    console.print("\n[yellow]Note: Actual training requires GPU and is commented out.[/yellow]")
    console.print("[yellow]To train, uncomment the train_model() call in main()[/yellow]")
    
    # Step 3: Evaluate (requires trained model)
    # results = evaluate_model(model_path)
    
    console.print("\n[bold green]Pipeline setup complete![/bold green]")
    console.print("\nTo run the full pipeline:")
    console.print("  1. Ensure you have a GPU with 6GB+ VRAM")
    console.print("  2. Install dependencies: pip install -e .[unsloth]")
    console.print("  3. Run: python scripts/train_example.py")
    console.print("\nOr use the CLI:")
    console.print("  edge-slm train data/sample_train.jsonl --output-dir outputs/model")


if __name__ == "__main__":
    main()
