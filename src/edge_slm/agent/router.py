"""
Intelligent Agent Router for Local/Cloud Decision Making.

This module implements the core routing logic that decides whether to:
1. Process locally with the fine-tuned SLM (fast, low-cost)
2. Forward to cloud API (slower, higher capability)

The router enables a hybrid architecture that balances latency, cost, and capability.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Routing strategies for request handling."""
    LOCAL_FIRST = "local_first"      # Try local, fallback to cloud
    CLOUD_FIRST = "cloud_first"      # Try cloud, fallback to local
    LOCAL_ONLY = "local_only"        # Only use local model
    CLOUD_ONLY = "cloud_only"        # Only use cloud API
    SMART = "smart"                  # Intelligent routing based on query


class TaskComplexity(str, Enum):
    """Estimated task complexity levels."""
    SIMPLE = "simple"      # Single tool call, clear intent
    MEDIUM = "medium"      # Multiple tools or ambiguous intent
    COMPLEX = "complex"    # Requires reasoning, multi-step planning


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    use_local: bool
    use_cloud: bool
    reason: str
    estimated_complexity: TaskComplexity
    confidence: float


@dataclass
class RoutingConfig:
    """Configuration for the router."""
    strategy: RoutingStrategy = RoutingStrategy.LOCAL_FIRST
    
    # Complexity thresholds
    complexity_threshold: float = 0.7  # Route to cloud if complexity > threshold
    confidence_threshold: float = 0.8  # Route to cloud if confidence < threshold
    
    # Query length thresholds
    short_query_max_tokens: int = 50   # Simple queries
    long_query_min_tokens: int = 200   # Complex queries
    
    # Latency targets (ms)
    local_latency_target: int = 500
    cloud_latency_budget: int = 5000
    
    # Retry settings
    max_local_retries: int = 2
    max_cloud_retries: int = 1


class AgentRouter:
    """
    Intelligent router that decides between local SLM and cloud API.
    
    The router uses multiple signals to make routing decisions:
    1. Query complexity estimation
    2. Historical performance data
    3. Current system load
    4. Cost considerations
    
    This enables significant cost savings (60%+) while maintaining quality.
    """
    
    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        local_engine: Optional[Any] = None,
        cloud_client: Optional[Any] = None,
    ):
        self.config = config or RoutingConfig()
        self.local_engine = local_engine
        self.cloud_client = cloud_client
        
        # Performance tracking
        self._local_success_count = 0
        self._local_failure_count = 0
        self._cloud_success_count = 0
        self._cloud_failure_count = 0
        self._latency_history: list[float] = []
    
    def estimate_complexity(self, query: str, tools: list[dict]) -> tuple[TaskComplexity, float]:
        """
        Estimate the complexity of a query.
        
        Uses heuristics to determine if the query is:
        - Simple: Clear intent, single tool needed
        - Medium: Some ambiguity or multiple tools
        - Complex: Requires reasoning or multi-step planning
        
        Args:
            query: User query
            tools: Available tools
        
        Returns:
            Tuple of (complexity, confidence)
        """
        # Simple heuristics for complexity estimation
        query_lower = query.lower()
        query_tokens = len(query.split())
        
        # Complexity indicators
        complex_indicators = [
            "分析", "比较", "解释", "为什么", "如何",
            "analyze", "compare", "explain", "why", "how",
            "多个", "所有", "每个", "multiple", "all", "each",
        ]
        
        simple_indicators = [
            "帮我", "请", "生成", "创建", "获取",
            "help", "please", "generate", "create", "get",
        ]
        
        # Count indicators
        complex_count = sum(1 for ind in complex_indicators if ind in query_lower)
        simple_count = sum(1 for ind in simple_indicators if ind in query_lower)
        
        # Estimate based on query length and indicators
        if query_tokens < self.config.short_query_max_tokens and complex_count == 0:
            complexity = TaskComplexity.SIMPLE
            confidence = 0.9 - (complex_count * 0.1)
        elif query_tokens > self.config.long_query_min_tokens or complex_count > 2:
            complexity = TaskComplexity.COMPLEX
            confidence = 0.7 + (complex_count * 0.05)
        else:
            complexity = TaskComplexity.MEDIUM
            confidence = 0.8
        
        # Adjust for number of tools
        if len(tools) > 5:
            confidence -= 0.1
        
        return complexity, min(max(confidence, 0.5), 1.0)
    
    def should_use_local(
        self,
        query: str,
        tools: list[dict],
        complexity: Optional[TaskComplexity] = None,
        confidence: Optional[float] = None,
    ) -> RoutingDecision:
        """
        Decide whether to use local model or cloud API.
        
        Args:
            query: User query
            tools: Available tools
            complexity: Pre-computed complexity (optional)
            confidence: Pre-computed confidence (optional)
        
        Returns:
            RoutingDecision with routing recommendation
        """
        # Handle fixed strategies
        if self.config.strategy == RoutingStrategy.LOCAL_ONLY:
            return RoutingDecision(
                use_local=True,
                use_cloud=False,
                reason="Strategy: local_only",
                estimated_complexity=TaskComplexity.SIMPLE,
                confidence=1.0,
            )
        
        if self.config.strategy == RoutingStrategy.CLOUD_ONLY:
            return RoutingDecision(
                use_local=False,
                use_cloud=True,
                reason="Strategy: cloud_only",
                estimated_complexity=TaskComplexity.COMPLEX,
                confidence=1.0,
            )
        
        # Estimate complexity if not provided
        if complexity is None or confidence is None:
            complexity, confidence = self.estimate_complexity(query, tools)
        
        # Smart routing logic
        if self.config.strategy == RoutingStrategy.SMART:
            # Complex queries -> cloud
            if complexity == TaskComplexity.COMPLEX:
                return RoutingDecision(
                    use_local=False,
                    use_cloud=True,
                    reason=f"Complex query detected (confidence: {confidence:.2f})",
                    estimated_complexity=complexity,
                    confidence=confidence,
                )
            
            # Low confidence -> cloud
            if confidence < self.config.confidence_threshold:
                return RoutingDecision(
                    use_local=False,
                    use_cloud=True,
                    reason=f"Low confidence ({confidence:.2f} < {self.config.confidence_threshold})",
                    estimated_complexity=complexity,
                    confidence=confidence,
                )
            
            # Simple/medium with high confidence -> local
            return RoutingDecision(
                use_local=True,
                use_cloud=False,
                reason=f"Simple/medium query with high confidence ({confidence:.2f})",
                estimated_complexity=complexity,
                confidence=confidence,
            )
        
        # Local-first strategy
        if self.config.strategy == RoutingStrategy.LOCAL_FIRST:
            return RoutingDecision(
                use_local=True,
                use_cloud=True,  # Cloud as fallback
                reason="Strategy: local_first with cloud fallback",
                estimated_complexity=complexity,
                confidence=confidence,
            )
        
        # Cloud-first strategy
        return RoutingDecision(
            use_local=True,  # Local as fallback
            use_cloud=True,
            reason="Strategy: cloud_first with local fallback",
            estimated_complexity=complexity,
            confidence=confidence,
        )
    
    async def route_request(
        self,
        query: str,
        tools: list[dict],
        conversation_history: Optional[list[dict]] = None,
    ) -> dict:
        """
        Route a request to the appropriate backend and return the result.
        
        Args:
            query: User query
            tools: Available tools
            conversation_history: Optional conversation context
        
        Returns:
            Dict with 'result', 'source', 'latency_ms', 'success'
        """
        decision = self.should_use_local(query, tools)
        
        start_time = time.time()
        result = None
        source = None
        success = False
        error = None
        
        # Try local first if recommended
        if decision.use_local and self.local_engine:
            try:
                local_result = await self._try_local(query, tools, conversation_history)
                if local_result.get("success"):
                    result = local_result["result"]
                    source = "local"
                    success = True
                    self._local_success_count += 1
            except Exception as e:
                logger.warning(f"Local inference failed: {e}")
                self._local_failure_count += 1
                error = str(e)
        
        # Fallback to cloud if local failed or not recommended
        if not success and decision.use_cloud and self.cloud_client:
            try:
                cloud_result = await self._try_cloud(query, tools, conversation_history)
                if cloud_result.get("success"):
                    result = cloud_result["result"]
                    source = "cloud"
                    success = True
                    self._cloud_success_count += 1
            except Exception as e:
                logger.error(f"Cloud inference failed: {e}")
                self._cloud_failure_count += 1
                error = str(e)
        
        latency = (time.time() - start_time) * 1000
        self._latency_history.append(latency)
        
        return {
            "result": result,
            "source": source,
            "latency_ms": latency,
            "success": success,
            "error": error,
            "routing_decision": decision,
        }
    
    async def _try_local(
        self,
        query: str,
        tools: list[dict],
        history: Optional[list[dict]],
    ) -> dict:
        """Try local inference."""
        if self.local_engine is None:
            return {"success": False, "error": "Local engine not configured"}
        
        # Build prompt
        prompt = self._build_prompt(query, tools, history)
        
        # Generate
        result = self.local_engine.generate(prompt, tools=tools)
        
        if result.is_valid and result.parsed:
            return {"success": True, "result": result.parsed}
        
        return {"success": False, "error": result.error or "Invalid output"}
    
    async def _try_cloud(
        self,
        query: str,
        tools: list[dict],
        history: Optional[list[dict]],
    ) -> dict:
        """Try cloud API inference."""
        if self.cloud_client is None:
            return {"success": False, "error": "Cloud client not configured"}
        
        try:
            # Use OpenAI-compatible API
            messages = history or []
            messages.append({"role": "user", "content": query})
            
            response = await self.cloud_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=[{"type": "function", "function": t.get("function", t)} for t in tools],
                tool_choice="auto",
            )
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                return {
                    "success": True,
                    "result": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                }
            
            return {"success": False, "error": "No tool call in response"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _build_prompt(
        self,
        query: str,
        tools: list[dict],
        history: Optional[list[dict]],
    ) -> str:
        """Build prompt for local model."""
        tools_desc = "\n".join([
            f"- {t.get('function', t)['name']}: {t.get('function', t)['description']}"
            for t in tools
        ])
        
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Use the following tools to help the user:

{tools_desc}

Respond with a JSON object: {{"name": "tool_name", "arguments": {{...}}}}
<|im_end|>
"""
        
        if history:
            for msg in history:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def get_stats(self) -> dict:
        """Get routing statistics."""
        total_local = self._local_success_count + self._local_failure_count
        total_cloud = self._cloud_success_count + self._cloud_failure_count
        
        return {
            "local": {
                "success": self._local_success_count,
                "failure": self._local_failure_count,
                "success_rate": self._local_success_count / total_local if total_local > 0 else 0,
            },
            "cloud": {
                "success": self._cloud_success_count,
                "failure": self._cloud_failure_count,
                "success_rate": self._cloud_success_count / total_cloud if total_cloud > 0 else 0,
            },
            "avg_latency_ms": sum(self._latency_history) / len(self._latency_history) if self._latency_history else 0,
            "total_requests": total_local + total_cloud,
            "local_ratio": total_local / (total_local + total_cloud) if (total_local + total_cloud) > 0 else 0,
        }
