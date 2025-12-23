"""
Agent module for intelligent routing and orchestration.
"""

from edge_slm.agent.router import AgentRouter, RoutingDecision, RoutingStrategy
from edge_slm.agent.service import AgentService

__all__ = ["AgentRouter", "RoutingDecision", "RoutingStrategy", "AgentService"]
