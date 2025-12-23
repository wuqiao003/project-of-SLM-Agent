"""
FastAPI Service for Agent Tool-Calling.

Provides a production-ready HTTP API for the edge SLM agent.
"""

from contextlib import asynccontextmanager
from typing import Any, Optional
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class ToolDefinition(BaseModel):
    """Tool definition in OpenAI format."""
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """Request for tool calling."""
    query: str = Field(..., description="User query")
    tools: list[dict] = Field(..., description="Available tools")
    conversation_history: Optional[list[dict]] = Field(default=None, description="Previous messages")
    use_structured_decoding: bool = Field(default=True, description="Enable grammar-constrained decoding")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.1, description="Sampling temperature")


class ToolCallResponse(BaseModel):
    """Response from tool calling."""
    name: str = Field(..., description="Tool name")
    arguments: dict = Field(..., description="Tool arguments")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    source: str = Field(default="local", description="Inference source (local/cloud)")
    is_valid: bool = Field(default=True, description="Whether output is valid JSON")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_memory_mb: Optional[float] = None
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Statistics response."""
    total_requests: int
    local_requests: int
    cloud_requests: int
    avg_latency_ms: float
    success_rate: float


# =============================================================================
# Service Class
# =============================================================================

class AgentService:
    """
    Agent Service that wraps the inference engine and router.
    
    Provides:
    - HTTP API endpoints for tool calling
    - Health monitoring
    - Performance statistics
    - Graceful shutdown
    """
    
    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        use_router: bool = True,
        cloud_api_key: Optional[str] = None,
    ):
        self.model_path = model_path
        self.backend = backend
        self.use_router = use_router
        self.cloud_api_key = cloud_api_key
        
        self.engine = None
        self.router = None
        self.start_time = time.time()
        
        # Statistics
        self._request_count = 0
        self._success_count = 0
        self._total_latency = 0.0
    
    def load(self) -> None:
        """Load the inference engine and router."""
        from edge_slm.inference import create_engine
        
        logger.info(f"Loading model: {self.model_path}")
        
        self.engine = create_engine(
            self.model_path,
            backend=self.backend,
            use_structured_decoding=True,
        )
        self.engine.load_model()
        
        if self.use_router:
            from edge_slm.agent.router import AgentRouter, RoutingConfig, RoutingStrategy
            
            # Setup cloud client if API key provided
            cloud_client = None
            if self.cloud_api_key:
                from openai import AsyncOpenAI
                cloud_client = AsyncOpenAI(api_key=self.cloud_api_key)
            
            self.router = AgentRouter(
                config=RoutingConfig(strategy=RoutingStrategy.LOCAL_FIRST),
                local_engine=self.engine,
                cloud_client=cloud_client,
            )
        
        logger.info("Service loaded successfully")
    
    def unload(self) -> None:
        """Unload the model to free resources."""
        if self.engine:
            self.engine.unload_model()
        self.engine = None
        self.router = None
        logger.info("Service unloaded")
    
    async def call_tool(self, request: ToolCallRequest) -> ToolCallResponse:
        """
        Process a tool calling request.
        
        Args:
            request: Tool call request
        
        Returns:
            Tool call response with name, arguments, and metadata
        """
        self._request_count += 1
        start_time = time.time()
        
        try:
            if self.router and self.use_router:
                # Use router for intelligent routing
                result = await self.router.route_request(
                    request.query,
                    request.tools,
                    request.conversation_history,
                )
                
                if result["success"]:
                    latency = result["latency_ms"]
                    self._success_count += 1
                    self._total_latency += latency
                    
                    return ToolCallResponse(
                        name=result["result"]["name"],
                        arguments=result["result"]["arguments"],
                        latency_ms=latency,
                        source=result["source"],
                        is_valid=True,
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Tool calling failed: {result.get('error', 'Unknown error')}"
                    )
            
            # Direct inference without router
            prompt = self._build_prompt(request)
            result = self.engine.generate(
                prompt,
                tools=request.tools,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            latency = (time.time() - start_time) * 1000
            self._total_latency += latency
            
            if result.is_valid and result.parsed:
                self._success_count += 1
                return ToolCallResponse(
                    name=result.parsed["name"],
                    arguments=result.parsed["arguments"],
                    latency_ms=latency,
                    source="local",
                    is_valid=True,
                )
            
            raise HTTPException(
                status_code=500,
                detail=f"Invalid output: {result.error or result.text}"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Tool calling error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _build_prompt(self, request: ToolCallRequest) -> str:
        """Build prompt for inference."""
        tools_desc = "\n".join([
            f"- {t.get('function', t).get('name', t.get('name'))}: "
            f"{t.get('function', t).get('description', t.get('description', ''))}"
            for t in request.tools
        ])
        
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Use the following tools:

{tools_desc}

Respond with JSON: {{"name": "tool_name", "arguments": {{...}}}}
<|im_end|>
"""
        
        if request.conversation_history:
            for msg in request.conversation_history:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{request.query}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def get_health(self) -> HealthResponse:
        """Get service health status."""
        import torch
        
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
        
        return HealthResponse(
            status="healthy" if self.engine else "not_loaded",
            model_loaded=self.engine is not None,
            gpu_memory_mb=gpu_memory,
            uptime_seconds=time.time() - self.start_time,
        )
    
    def get_stats(self) -> StatsResponse:
        """Get service statistics."""
        router_stats = self.router.get_stats() if self.router else {}
        
        return StatsResponse(
            total_requests=self._request_count,
            local_requests=router_stats.get("local", {}).get("success", 0) + 
                          router_stats.get("local", {}).get("failure", 0),
            cloud_requests=router_stats.get("cloud", {}).get("success", 0) + 
                          router_stats.get("cloud", {}).get("failure", 0),
            avg_latency_ms=self._total_latency / self._request_count if self._request_count > 0 else 0,
            success_rate=self._success_count / self._request_count if self._request_count > 0 else 0,
        )


# =============================================================================
# FastAPI Application Factory
# =============================================================================

def create_app(
    model_path: str,
    backend: str = "transformers",
    cloud_api_key: Optional[str] = None,
) -> FastAPI:
    """
    Create FastAPI application with the agent service.
    
    Args:
        model_path: Path to the model
        backend: Inference backend ('transformers', 'vllm')
        cloud_api_key: Optional OpenAI API key for cloud fallback
    
    Returns:
        FastAPI application
    """
    service = AgentService(
        model_path=model_path,
        backend=backend,
        cloud_api_key=cloud_api_key,
    )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        service.load()
        yield
        # Shutdown
        service.unload()
    
    app = FastAPI(
        title="Edge SLM Agent API",
        description="Local tool-calling inference with grammar-constrained decoding",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    @app.post("/v1/tool_call", response_model=ToolCallResponse)
    async def tool_call(request: ToolCallRequest):
        """Execute a tool call based on user query."""
        return await service.call_tool(request)
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return service.get_health()
    
    @app.get("/stats", response_model=StatsResponse)
    async def stats():
        """Get service statistics."""
        return service.get_stats()
    
    return app


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    backend: str = "transformers",
    cloud_api_key: Optional[str] = None,
) -> None:
    """
    Run the agent service.
    
    Args:
        model_path: Path to the model
        host: Host to bind to
        port: Port to listen on
        backend: Inference backend
        cloud_api_key: Optional OpenAI API key
    """
    import uvicorn
    
    app = create_app(model_path, backend, cloud_api_key)
    uvicorn.run(app, host=host, port=port)
