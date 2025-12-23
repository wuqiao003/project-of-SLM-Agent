"""
Unsloth-accelerated QLoRA Trainer.

Unsloth provides 2-5x faster training with 60% less memory usage.
Optimized specifically for consumer-grade GPUs like RTX 3060.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import logging

import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class UnslothConfig:
    """Configuration for Unsloth-accelerated training."""
    
    # Model
    model_name: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 32  # Lower rank for memory efficiency
    lora_alpha: int = 64
    lora_dropout: float = 0.0  # Unsloth recommends 0 dropout
    target_modules: list = None  # Auto-detected by Unsloth
    
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 2  # Small batch for 6GB VRAM
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_steps: int = 10
    weight_decay: float = 0.01
    
    # Optimization
    use_gradient_checkpointing: bool = True
    random_state: int = 42
    
    # Output
    output_dir: str = "outputs/unsloth"
    logging_steps: int = 10
    save_steps: int = 100
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Qwen
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


class UnslothTrainer:
    """
    Unsloth-accelerated trainer for maximum efficiency on consumer GPUs.
    
    Features:
    - 2-5x faster training than standard HuggingFace
    - 60% less memory usage
    - Native 4-bit quantization support
    - Optimized for RTX 3060 (6GB)
    """
    
    def __init__(self, config: UnslothConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._unsloth_available = self._check_unsloth()
    
    def _check_unsloth(self) -> bool:
        """Check if Unsloth is available."""
        try:
            from unsloth import FastLanguageModel
            return True
        except ImportError:
            logger.warning(
                "Unsloth not installed. Install with: "
                "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
            )
            return False
    
    def load_model(self) -> None:
        """Load model using Unsloth's optimized loader."""
        if not self._unsloth_available:
            raise RuntimeError("Unsloth is not available")
        
        from unsloth import FastLanguageModel
        
        logger.info(f"Loading model with Unsloth: {self.config.model_name}")
        
        # Load 4-bit quantized model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        # Apply LoRA with Unsloth optimizations
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=self.config.random_state,
            use_rslora=False,  # Rank-stabilized LoRA
            loftq_config=None,
        )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model loaded with Unsloth optimizations")
        self._print_memory_stats()
    
    def _print_memory_stats(self) -> None:
        """Print GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def prepare_dataset(
        self,
        dataset: Dataset,
        formatting_func: Optional[callable] = None,
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            dataset: HuggingFace Dataset with 'messages' field
            formatting_func: Optional custom formatting function
        
        Returns:
            Formatted dataset
        """
        if formatting_func is None:
            # Default formatting using chat template
            def format_example(example):
                messages = example.get("messages", [])
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return {"text": text}
            
            formatting_func = format_example
        
        return dataset.map(formatting_func)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Run training with Unsloth acceleration.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        if self.model is None:
            self.load_model()
        
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        # Prepare datasets
        train_data = self.prepare_dataset(train_dataset)
        eval_data = self.prepare_dataset(eval_dataset) if eval_dataset else None
        
        # Training arguments optimized for RTX 3060
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            seed=self.config.random_state,
            # Memory optimization
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            max_grad_norm=1.0,
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            packing=False,
        )
        
        # Train
        logger.info("Starting Unsloth-accelerated training...")
        self._print_memory_stats()
        
        trainer.train()
        
        # Save
        self.save_model()
        
        logger.info(f"Training complete. Model saved to {self.config.output_dir}")
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the LoRA adapter."""
        save_path = path or os.path.join(self.config.output_dir, "lora_adapter")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"LoRA adapter saved to {save_path}")
    
    def save_merged_model(
        self,
        output_path: str,
        quantization: Optional[str] = None,
    ) -> None:
        """
        Merge LoRA and save in various formats.
        
        Args:
            output_path: Output directory
            quantization: Optional quantization format ('q4_k_m', 'q8_0', etc.)
        """
        from unsloth import FastLanguageModel
        
        if quantization:
            # Save as GGUF for llama.cpp
            logger.info(f"Saving as GGUF with {quantization} quantization...")
            self.model.save_pretrained_gguf(
                output_path,
                self.tokenizer,
                quantization_method=quantization,
            )
        else:
            # Save as merged 16-bit model
            logger.info("Saving merged 16-bit model...")
            self.model.save_pretrained_merged(
                output_path,
                self.tokenizer,
                save_method="merged_16bit",
            )
        
        logger.info(f"Model saved to {output_path}")
    
    def export_to_vllm(self, output_path: str) -> None:
        """Export model in format optimized for vLLM inference."""
        logger.info("Exporting for vLLM...")
        
        # Save merged model
        self.save_merged_model(output_path)
        
        # vLLM can load the merged model directly
        logger.info(f"Model ready for vLLM at {output_path}")
    
    def export_to_gguf(
        self,
        output_path: str,
        quantization: str = "q4_k_m",
    ) -> None:
        """
        Export model as GGUF for llama.cpp.
        
        Args:
            output_path: Output directory
            quantization: Quantization method (q4_k_m, q5_k_m, q8_0, etc.)
        """
        self.save_merged_model(output_path, quantization=quantization)


def train_with_unsloth(
    data_path: Union[str, Path],
    output_dir: str = "outputs/unsloth_model",
    model_name: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    num_epochs: int = 3,
) -> None:
    """
    Convenience function for Unsloth training.
    
    Args:
        data_path: Path to training data (JSONL)
        output_dir: Output directory
        model_name: Unsloth model name
        num_epochs: Number of epochs
    """
    from edge_slm.data import create_dataset
    
    config = UnslothConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
    )
    
    trainer = UnslothTrainer(config)
    trainer.load_model()
    
    # Load dataset
    dataset = create_dataset(data_path, split=True)
    
    # Train
    trainer.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
    )
    
    # Export for inference
    trainer.export_to_vllm(os.path.join(output_dir, "vllm_model"))
