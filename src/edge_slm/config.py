"""
Configuration module for Edge SLM Agent system.
Centralized configuration management for all components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import json


@dataclass
class ModelConfig:
    """Model configuration for base and fine-tuned models."""
    
    # Base model selection
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_type: Literal["qwen", "phi", "llama"] = "qwen"
    
    # Quantization settings
    quantization: Literal["none", "int4", "int8", "awq"] = "int4"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    
    # Memory optimization
    max_memory_mb: int = 5500  # For RTX 3060 6GB
    device_map: str = "auto"
    
    # Model paths
    base_model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    merged_model_path: Optional[str] = None


@dataclass
class LoRAConfig:
    """LoRA/QLoRA fine-tuning configuration."""
    
    # LoRA hyperparameters
    r: int = 64  # LoRA rank
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    
    # Target modules for different model architectures
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training settings
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
    
    # Gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = True
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


@dataclass
class InferenceConfig:
    """Inference engine configuration."""
    
    # Engine selection
    engine: Literal["vllm", "transformers", "llama_cpp"] = "vllm"
    
    # vLLM specific settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.1  # Low for structured output
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    
    # Structured decoding
    use_guided_decoding: bool = True
    guided_decoding_backend: Literal["outlines", "lm-format-enforcer"] = "outlines"
    
    # Batching
    max_batch_size: int = 8
    max_waiting_tokens: int = 20


@dataclass
class DataConfig:
    """Dataset configuration for training data generation."""
    
    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    output_path: str = "data/tool_use_dataset"
    
    # GPT-4 distillation settings
    teacher_model: str = "gpt-4-turbo-preview"
    distillation_temperature: float = 0.7
    num_samples: int = 1000
    
    # Data split
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_factor: int = 2


@dataclass
class AgentConfig:
    """Agent routing and orchestration configuration."""
    
    # Routing strategy
    routing_strategy: Literal["local_first", "cloud_first", "hybrid"] = "local_first"
    
    # Local model settings
    local_model_endpoint: str = "http://localhost:8000"
    local_timeout_ms: int = 5000
    
    # Cloud fallback settings
    cloud_model: str = "gpt-4-turbo-preview"
    cloud_timeout_ms: int = 30000
    
    # Routing thresholds
    complexity_threshold: float = 0.7  # Route to cloud if complexity > threshold
    confidence_threshold: float = 0.8  # Route to cloud if confidence < threshold
    
    # Retry settings
    max_retries: int = 2
    retry_delay_ms: int = 100


@dataclass
class EdgeSLMConfig:
    """Master configuration combining all component configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # Global settings
    project_name: str = "edge-slm-agent"
    output_dir: str = "outputs"
    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "edge-slm-agent"
    
    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "inference": self.inference.__dict__,
            "data": self.data.__dict__,
            "agent": self.agent.__dict__,
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str | Path) -> "EdgeSLMConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        config = cls()
        config.model = ModelConfig(**config_dict.get("model", {}))
        config.lora = LoRAConfig(**config_dict.get("lora", {}))
        config.inference = InferenceConfig(**config_dict.get("inference", {}))
        config.data = DataConfig(**config_dict.get("data", {}))
        config.agent = AgentConfig(**config_dict.get("agent", {}))
        
        for key in ["project_name", "output_dir", "seed", "use_wandb", "wandb_project"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config


# Predefined configurations for different hardware
CONFIGS = {
    "rtx3060_6gb": EdgeSLMConfig(
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            quantization="int4",
            max_memory_mb=5500,
        ),
        lora=LoRAConfig(
            r=32,
            batch_size=2,
            gradient_accumulation_steps=8,
            max_seq_length=1536,
        ),
        inference=InferenceConfig(
            gpu_memory_utilization=0.80,
            max_model_len=2048,
        ),
    ),
    "rtx4090_24gb": EdgeSLMConfig(
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            quantization="int4",
            max_memory_mb=22000,
        ),
        lora=LoRAConfig(
            r=64,
            batch_size=8,
            gradient_accumulation_steps=2,
            max_seq_length=4096,
        ),
        inference=InferenceConfig(
            gpu_memory_utilization=0.90,
            max_model_len=8192,
        ),
    ),
}


def get_config(preset: str = "rtx3060_6gb") -> EdgeSLMConfig:
    """Get predefined configuration by preset name."""
    if preset not in CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[preset]
