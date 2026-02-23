"""Base Agent class — adapted from heterogeneous-agent-swarm.

All agents in the swarm implement this interface to enable uniform
orchestration, NAS capacity scaling, and MetaKernel management.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AgentProposal:
    """A proposal from an agent for an action.

    Attributes:
        agent_name: Name of the proposing agent
        action: The proposed action tensor
        confidence: Agent's confidence in this proposal [0, 1]
        metadata: Additional metadata (e.g., reasoning trace)
    """
    agent_name: str
    action: torch.Tensor
    confidence: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Abstract base class for all swarm agents.

    Each agent must implement:
        - propose(observation) -> AgentProposal
        - get_capacity_metrics() -> dict

    Agents may optionally implement:
        - increase_capacity() / decrease_capacity() for NAS
        - reset_state() for episode boundaries
    """

    def __init__(self, name: str, observation_dim: int, action_dim: int):
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.is_suppressed = False
        self.consecutive_failures = 0

    @abstractmethod
    def propose(self, observation: torch.Tensor) -> AgentProposal:
        """Generate an action proposal given an observation."""
        ...

    @abstractmethod
    def get_capacity_metrics(self) -> Dict[str, Any]:
        """Return capacity/resource metrics for NAS decisions."""
        ...

    def increase_capacity(self) -> None:
        """Increase agent's model capacity (for NAS). Override in subclasses."""
        pass

    def decrease_capacity(self) -> None:
        """Decrease agent's model capacity (for NAS). Override in subclasses."""
        pass

    def reset_state(self) -> None:
        """Reset internal state (e.g., hidden states) at episode boundary."""
        pass

    def suppress(self) -> None:
        """Suppress this agent (MetaKernel decision)."""
        self.is_suppressed = True

    def recover(self) -> None:
        """Recover this agent from suppression."""
        self.is_suppressed = False
        self.consecutive_failures = 0

    def record_outcome(self, success: bool) -> None:
        """Record whether the agent's proposal was successful."""
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
