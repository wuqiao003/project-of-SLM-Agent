"""
Base Inference Engine with support for multiple backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InferenceBackend(str, Enum):
    """Supported inference backends."""
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    
    # Model
    model_path: str = ""
    backend: InferenceBackend = InferenceBackend.TRANSFORMERS
    
    # Quantization
    load_in_4bit: bool = True
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    
    # Structured decoding
    use_structured_decoding: bool = True
    
    # Memory
    max_memory_mb: int = 5500
    device: str = "cuda"


@dataclass
class GenerationResult:
    """Result from generation."""
    text: str
    parsed: Optional[Any] = None
    tokens_generated: int = 0
    latency_ms: float = 0.0
    is_valid: bool = True
    error: Optional[str] = None


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text with optional structured decoding."""
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> list[GenerationResult]:
        """Generate for multiple prompts."""
        pass
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self.tokenizer = None
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TransformersEngine(InferenceEngine):
    """Inference engine using HuggingFace Transformers."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.structured_decoder = None
    
    def load_model(self) -> None:
        """Load model with optional quantization."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info(f"Loading model: {self.config.model_path}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Initialize structured decoder
        if self.config.use_structured_decoding:
            from edge_slm.inference.structured import StructuredDecoder
            self.structured_decoder = StructuredDecoder()
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate with optional structured decoding."""
        import time
        import torch
        
        start_time = time.time()
        
        try:
            if tools and self.config.use_structured_decoding and self.structured_decoder:
                # Use structured decoding
                from edge_slm.inference.structured import GrammarConstraint, OutputFormat
                
                constraint = self.structured_decoder.create_constraint(tools=tools)
                
                result = self.structured_decoder.decode(
                    self.model,
                    self.tokenizer,
                    prompt,
                    constraint,
                    max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                )
                
                latency = (time.time() - start_time) * 1000
                
                if isinstance(result, dict):
                    return GenerationResult(
                        text=str(result),
                        parsed=result,
                        latency_ms=latency,
                        is_valid=True,
                    )
                else:
                    return GenerationResult(
                        text=result,
                        latency_ms=latency,
                        is_valid=False,
                        error="Failed to parse as JSON",
                    )
            
            # Standard generation
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    top_p=kwargs.get("top_p", self.config.top_p),
                    top_k=kwargs.get("top_k", self.config.top_k),
                    repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
                    do_sample=kwargs.get("temperature", self.config.temperature) > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            generated = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            
            latency = (time.time() - start_time) * 1000
            tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            
            return GenerationResult(
                text=generated,
                tokens_generated=tokens,
                latency_ms=latency,
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return GenerationResult(
                text="",
                latency_ms=latency,
                is_valid=False,
                error=str(e),
            )
    
    def generate_batch(
        self,
        prompts: list[str],
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> list[GenerationResult]:
        """Generate for multiple prompts."""
        return [self.generate(p, tools, **kwargs) for p in prompts]


def create_engine(
    model_path: str,
    backend: str = "transformers",
    **kwargs,
) -> InferenceEngine:
    """
    Factory function to create an inference engine.
    
    Args:
        model_path: Path to the model
        backend: Backend to use ('transformers', 'vllm', 'llama_cpp')
        **kwargs: Additional configuration
    
    Returns:
        InferenceEngine instance
    """
    config = InferenceConfig(
        model_path=model_path,
        backend=InferenceBackend(backend),
        **kwargs,
    )
    
    if backend == "vllm":
        from edge_slm.inference.vllm_engine import VLLMEngine
        return VLLMEngine(config)
    elif backend == "llama_cpp":
        raise NotImplementedError("LlamaCpp backend not yet implemented")
    else:
        return TransformersEngine(config)
