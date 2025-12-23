"""
vLLM-based Inference Engine with Native Structured Decoding Support.

vLLM provides:
- High-throughput serving with PagedAttention
- Native guided decoding support
- Efficient batching and scheduling
- Production-ready performance
"""

import json
import time
from typing import Any, Optional
import logging

from edge_slm.inference.engine import InferenceEngine, InferenceConfig, GenerationResult

logger = logging.getLogger(__name__)


class VLLMEngine(InferenceEngine):
    """
    vLLM-based inference engine optimized for production deployment.
    
    Key Features:
    - PagedAttention for efficient KV cache management
    - Native guided decoding with Outlines integration
    - Continuous batching for high throughput
    - Optimized for RTX 3060 (6GB) with proper memory configuration
    """
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.llm = None
        self.sampling_params = None
    
    def load_model(self) -> None:
        """Load model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
        
        logger.info(f"Loading model with vLLM: {self.config.model_path}")
        
        # Calculate GPU memory utilization
        # For 6GB GPU, leave ~1GB for KV cache overhead
        gpu_memory_utilization = min(0.85, self.config.max_memory_mb / 6000)
        
        # Load model
        self.llm = LLM(
            model=self.config.model_path,
            tensor_parallel_size=1,  # Single GPU
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,  # Reduced for memory efficiency
            trust_remote_code=True,
            dtype="half",  # FP16 for memory efficiency
            quantization="awq" if "awq" in self.config.model_path.lower() else None,
        )
        
        # Default sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        )
        
        logger.info("vLLM model loaded successfully")
    
    def _create_guided_params(
        self,
        tools: list[dict],
        base_params: Any,
    ) -> Any:
        """Create sampling params with guided decoding."""
        from vllm import SamplingParams
        
        # Build JSON schema for tool calls
        tool_schemas = []
        for tool in tools:
            func = tool.get("function", tool)
            tool_schemas.append({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": func["name"]},
                    "arguments": func.get("parameters", {"type": "object"}),
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            })
        
        schema = {"oneOf": tool_schemas} if len(tool_schemas) > 1 else tool_schemas[0]
        
        # Create new params with guided decoding
        return SamplingParams(
            max_tokens=base_params.max_tokens,
            temperature=base_params.temperature,
            top_p=base_params.top_p,
            top_k=base_params.top_k,
            repetition_penalty=base_params.repetition_penalty,
            # Guided decoding configuration
            guided_decoding={
                "json": schema,
                "backend": "outlines",
            } if self.config.use_structured_decoding else None,
        )
    
    def generate(
        self,
        prompt: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate with vLLM, optionally using guided decoding."""
        if self.llm is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Prepare sampling params
            from vllm import SamplingParams
            
            params = SamplingParams(
                max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                top_k=kwargs.get("top_k", self.config.top_k),
                repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
            )
            
            # Add guided decoding if tools provided
            if tools and self.config.use_structured_decoding:
                params = self._create_guided_params(tools, params)
            
            # Generate
            outputs = self.llm.generate([prompt], params)
            
            latency = (time.time() - start_time) * 1000
            
            if outputs and outputs[0].outputs:
                output = outputs[0].outputs[0]
                text = output.text
                tokens = len(output.token_ids)
                
                # Try to parse as JSON
                parsed = None
                is_valid = True
                error = None
                
                if tools:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError as e:
                        is_valid = False
                        error = f"JSON parse error: {e}"
                
                return GenerationResult(
                    text=text,
                    parsed=parsed,
                    tokens_generated=tokens,
                    latency_ms=latency,
                    is_valid=is_valid,
                    error=error,
                )
            
            return GenerationResult(
                text="",
                latency_ms=latency,
                is_valid=False,
                error="No output generated",
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Generation error: {e}")
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
        """Generate for multiple prompts with batching."""
        if self.llm is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            from vllm import SamplingParams
            
            params = SamplingParams(
                max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            
            if tools and self.config.use_structured_decoding:
                params = self._create_guided_params(tools, params)
            
            # Batch generate
            outputs = self.llm.generate(prompts, params)
            
            total_latency = (time.time() - start_time) * 1000
            per_prompt_latency = total_latency / len(prompts)
            
            results = []
            for output in outputs:
                if output.outputs:
                    text = output.outputs[0].text
                    tokens = len(output.outputs[0].token_ids)
                    
                    parsed = None
                    is_valid = True
                    error = None
                    
                    if tools:
                        try:
                            parsed = json.loads(text)
                        except json.JSONDecodeError as e:
                            is_valid = False
                            error = str(e)
                    
                    results.append(GenerationResult(
                        text=text,
                        parsed=parsed,
                        tokens_generated=tokens,
                        latency_ms=per_prompt_latency,
                        is_valid=is_valid,
                        error=error,
                    ))
                else:
                    results.append(GenerationResult(
                        text="",
                        latency_ms=per_prompt_latency,
                        is_valid=False,
                        error="No output",
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            return [
                GenerationResult(text="", is_valid=False, error=str(e))
                for _ in prompts
            ]


class VLLMServer:
    """
    vLLM Server wrapper for production deployment.
    
    Provides an OpenAI-compatible API endpoint.
    """
    
    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.kwargs = kwargs
    
    def start(self) -> None:
        """Start the vLLM server."""
        import subprocess
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", "half",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.85",
        ]
        
        if self.kwargs.get("quantization"):
            cmd.extend(["--quantization", self.kwargs["quantization"]])
        
        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        subprocess.run(cmd)


def start_vllm_server(
    model_path: str,
    port: int = 8000,
    **kwargs,
) -> None:
    """Convenience function to start vLLM server."""
    server = VLLMServer(model_path, port=port, **kwargs)
    server.start()
