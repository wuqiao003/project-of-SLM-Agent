"""
Standard QLoRA Trainer using PEFT and Transformers.
Provides a fallback when Unsloth is not available.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import logging

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    
    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    
    # Optimizer
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    
    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    
    # Output
    output_dir: str = "outputs/qlora"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Misc
    seed: int = 42
    report_to: str = "wandb"


class SLMTrainer:
    """
    QLoRA Trainer for Small Language Models.
    
    Optimized for consumer-grade GPUs like RTX 3060 (6GB).
    Uses 4-bit quantization and gradient checkpointing for memory efficiency.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_quantization(self) -> BitsAndBytesConfig:
        """Configure 4-bit quantization for memory efficiency."""
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
        
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
        )
    
    def setup_lora(self) -> LoraConfig:
        """Configure LoRA adapters."""
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    
    def load_model(self) -> None:
        """Load and prepare the base model with quantization."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with quantization
        bnb_config = self.setup_quantization()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )
        
        # Apply LoRA
        lora_config = self.setup_lora()
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        logger.info("Model loaded and prepared for training")
    
    def get_training_arguments(self) -> TrainingArguments:
        """Configure training arguments."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps" if self.config.eval_steps else "no",
            load_best_model_at_end=True if self.config.eval_steps else False,
            report_to=self.config.report_to,
            seed=self.config.seed,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=1.0,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Run the fine-tuning process.
        
        Args:
            train_dataset: Training dataset with 'text' or 'messages' field
            eval_dataset: Optional evaluation dataset
        """
        if self.peft_model is None:
            self.load_model()
        
        training_args = self.get_training_arguments()
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            packing=False,  # Disable packing for tool-use data
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        self.save_model()
        
        logger.info(f"Training complete. Model saved to {self.config.output_dir}")
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the fine-tuned LoRA adapter."""
        save_path = path or os.path.join(self.config.output_dir, "final_adapter")
        
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Adapter saved to {save_path}")
    
    def merge_and_save(self, output_path: str) -> None:
        """Merge LoRA weights with base model and save."""
        logger.info("Merging LoRA weights with base model...")
        
        # Merge weights
        merged_model = self.peft_model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved to {output_path}")


def train_tool_use_model(
    data_path: Union[str, Path],
    output_dir: str = "outputs/tool_use_model",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_epochs: int = 3,
    batch_size: int = 4,
) -> None:
    """
    Convenience function to train a tool-use model.
    
    Args:
        data_path: Path to training data (JSONL format)
        output_dir: Output directory for the model
        model_name: Base model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Training batch size
    """
    from edge_slm.data import create_dataset
    
    config = TrainingConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    
    trainer = SLMTrainer(config)
    trainer.load_model()
    
    # Load and prepare dataset
    dataset = create_dataset(data_path, trainer.tokenizer, split=True)
    
    # Prepare for SFT
    def format_for_sft(example):
        messages = example.get("messages", [])
        text = trainer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    
    train_data = dataset["train"].map(format_for_sft)
    eval_data = dataset["validation"].map(format_for_sft) if "validation" in dataset else None
    
    # Train
    trainer.train(train_data, eval_data)
