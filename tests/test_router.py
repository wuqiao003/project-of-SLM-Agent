"""
Tests for Agent Router functionality.
"""

import pytest

from edge_slm.agent.router import (
    AgentRouter,
    RoutingConfig,
    RoutingStrategy,
    RoutingDecision,
    TaskComplexity,
)


class TestAgentRouter:
    """Tests for AgentRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create a router instance."""
        config = RoutingConfig(strategy=RoutingStrategy.SMART)
        return AgentRouter(config)
    
    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions."""
        return [
            {"function": {"name": "parse_video", "description": "Parse video"}},
            {"function": {"name": "generate_subtitles", "description": "Generate subtitles"}},
        ]
    
    def test_estimate_complexity_simple(self, router, sample_tools):
        """Test complexity estimation for simple queries."""
        query = "分析视频 https://example.com/v.mp4"
        complexity, confidence = router.estimate_complexity(query, sample_tools)
        
        assert complexity == TaskComplexity.SIMPLE
        assert confidence > 0.8
    
    def test_estimate_complexity_complex(self, router, sample_tools):
        """Test complexity estimation for complex queries."""
        query = "分析这个视频的内容，比较不同部分的主题，解释为什么会有这样的变化"
        complexity, confidence = router.estimate_complexity(query, sample_tools)
        
        assert complexity in [TaskComplexity.MEDIUM, TaskComplexity.COMPLEX]
    
    def test_routing_local_only(self, sample_tools):
        """Test local-only routing strategy."""
        config = RoutingConfig(strategy=RoutingStrategy.LOCAL_ONLY)
        router = AgentRouter(config)
        
        decision = router.should_use_local("any query", sample_tools)
        
        assert decision.use_local is True
        assert decision.use_cloud is False
    
    def test_routing_cloud_only(self, sample_tools):
        """Test cloud-only routing strategy."""
        config = RoutingConfig(strategy=RoutingStrategy.CLOUD_ONLY)
        router = AgentRouter(config)
        
        decision = router.should_use_local("any query", sample_tools)
        
        assert decision.use_local is False
        assert decision.use_cloud is True
    
    def test_routing_local_first(self, sample_tools):
        """Test local-first routing strategy."""
        config = RoutingConfig(strategy=RoutingStrategy.LOCAL_FIRST)
        router = AgentRouter(config)
        
        decision = router.should_use_local("simple query", sample_tools)
        
        assert decision.use_local is True
        assert decision.use_cloud is True  # Cloud as fallback
    
    def test_routing_smart_simple_query(self, sample_tools):
        """Test smart routing with simple query."""
        config = RoutingConfig(strategy=RoutingStrategy.SMART)
        router = AgentRouter(config)
        
        decision = router.should_use_local("帮我分析视频", sample_tools)
        
        # Simple queries should route to local
        assert decision.use_local is True
    
    def test_get_stats_initial(self, router):
        """Test initial statistics."""
        stats = router.get_stats()
        
        assert stats["total_requests"] == 0
        assert stats["local"]["success"] == 0
        assert stats["cloud"]["success"] == 0


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""
    
    def test_routing_decision_creation(self):
        """Test creating a routing decision."""
        decision = RoutingDecision(
            use_local=True,
            use_cloud=False,
            reason="Test reason",
            estimated_complexity=TaskComplexity.SIMPLE,
            confidence=0.95,
        )
        
        assert decision.use_local is True
        assert decision.use_cloud is False
        assert decision.confidence == 0.95


class TestRoutingConfig:
    """Tests for RoutingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RoutingConfig()
        
        assert config.strategy == RoutingStrategy.LOCAL_FIRST
        assert config.complexity_threshold == 0.7
        assert config.confidence_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RoutingConfig(
            strategy=RoutingStrategy.SMART,
            complexity_threshold=0.5,
            confidence_threshold=0.9,
        )
        
        assert config.strategy == RoutingStrategy.SMART
        assert config.complexity_threshold == 0.5
        assert config.confidence_threshold == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
