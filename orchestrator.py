"""Swarm Orchestrator with Test-Time Adaptation (TTA) for flat module layout.

This version assumes all modules live in the repository root and uses
absolute imports instead of package-relative ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from base_agent import BaseAgent, AgentProposal
from config import OrchestratorConfig, TTAConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    """Tracks the orchestrator's internal state."""

    step_count: int = 0
    global_failures: int = 0
    recent_losses: list[float] | None = None
    is_panic: bool = False

    def __post_init__(self) -> None:
        if self.recent_losses is None:
            self.recent_losses = []


class TestTimeAdapter:
    """Test-Time Adaptation for the orchestrator's agent selection.

    Monitors recent proposal quality and performs gradient-free
    adaptation of selection weights when distributional shift is detected.
    """

    def __init__(self, agent_names: List[str], config: TTAConfig) -> None:
        self.config = config
        self.agent_names = agent_names
        # Weight for each agent (softmax-normalized before use)
        self.logits: Dict[str, float] = {name: 1.0 for name in agent_names}
        self.loss_history: List[float] = []
        self.mean_loss = 0.0
        self.std_loss = 1.0

    def should_adapt(self) -> bool:
        """Detect distributional shift via loss spike."""
        if len(self.loss_history) < 10:
            return False
        recent = self.loss_history[-5:]
        recent_mean = sum(recent) / len(recent)
        return recent_mean > self.mean_loss + self.config.shock_threshold * self.std_loss

    def adapt_weights(self, agent_outcomes: Dict[str, float]) -> None:
        """Adapt agent selection weights based on recent outcomes.

        Args:
            agent_outcomes: Dict mapping agent_name -> recent performance score
        """
        for name, score in agent_outcomes.items():
            if name in self.logits:
                # Increase weight for good performers, decrease for bad
                self.logits[name] += self.config.learning_rate * (score - 0.5)
                self.logits[name] = max(0.01, self.logits[name])  # Floor

    def record_loss(self, loss: float) -> None:
        """Record a loss value and update running statistics."""
        self.loss_history.append(loss)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        if len(self.loss_history) >= 2:
            t = torch.tensor(self.loss_history, dtype=torch.float32)
            self.mean_loss = t.mean().item()
            self.std_loss = max(t.std().item(), 0.01)

    def get_weights(self) -> Dict[str, float]:
        """Get softmax-normalized agent weights."""
        vals = torch.tensor([self.logits[n] for n in self.agent_names])
        weights = torch.softmax(vals, dim=0)
        return {n: w.item() for n, w in zip(self.agent_names, weights)}


class Orchestrator:
    """Swarm Orchestrator with Test-Time Adaptation.

    Selects the best action from agent proposals using configurable
    strategies: weighted_perf, consensus, or panic mode.
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        config: Optional[OrchestratorConfig] = None,
        tta_config: Optional[TTAConfig] = None,
    ) -> None:
        if config is None:
            config = OrchestratorConfig()
        if tta_config is None:
            tta_config = TTAConfig()

        self.agents = agents
        self.config = config
        self.state = OrchestratorState()

        # TTA adapter
        self.tta = TestTimeAdapter(list(agents.keys()), tta_config) if config.tta_enabled else None

    def select_action(
        self,
        observation: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, AgentProposal]]:
        """Collect proposals and select best action. Handles dynamic swarm."""
        # Dynamic Swarm Synchronization
        if self.tta and set(self.agents.keys()) != set(self.tta.agent_names):
            for name in self.agents:
                if name not in self.tta.agent_names:
                    self.tta.agent_names.append(name)
                    self.tta.logits[name] = 1.0

        proposals: Dict[str, AgentProposal] = {}
        D_curr = observation.shape[0]

        # If agents have different observation dims, use zero-padding / truncation
        for name, agent in self.agents.items():
            if not agent.is_suppressed:
                obs_to_agent = observation
                if D_curr > agent.observation_dim:
                    obs_to_agent = observation[: agent.observation_dim]
                elif D_curr < agent.observation_dim:
                    obs_to_agent = torch.cat(
                        [observation, torch.zeros(agent.observation_dim - D_curr)]
                    )

                prop = agent.propose(obs_to_agent)

                # Resync proposal to D_curr
                if prop.action.shape[0] != D_curr:
                    new_action = torch.zeros(D_curr)
                    min_d = min(D_curr, prop.action.shape[0])
                    new_action[:min_d] = prop.action[:min_d]
                    prop.action = new_action
                proposals[name] = prop

        if not proposals:
            logger.warning("No active agents! Returning zero action.")
            first_agent = next(iter(self.agents.values()))
            return torch.zeros(first_agent.action_dim), proposals

        # Check if we should enter panic mode
        if self.state.global_failures >= self.config.panic_threshold:
            self.state.is_panic = True
            logger.warning("PANIC MODE: too many global failures")

        # Select action based on mode
        if self.state.is_panic:
            selected = self._select_panic(proposals)
        elif self.config.selection_mode == "consensus":
            selected = self._select_consensus(proposals)
        else:
            selected = self._select_weighted(proposals)

        # TTA: track loss and adapt if needed
        if self.tta is not None and ground_truth is not None:
            loss = (selected - ground_truth).pow(2).mean().item()
            self.tta.record_loss(loss)
            if self.tta.should_adapt():
                logger.info("TTA: distributional shift detected, adapting weights")
                outcomes = {
                    name: 1.0 - (p.action - ground_truth).pow(2).mean().item()
                    for name, p in proposals.items()
                }
                self.tta.adapt_weights(outcomes)

        self.state.step_count += 1
        return selected, proposals

    def _select_weighted(self, proposals: Dict[str, AgentProposal]) -> torch.Tensor:
        """Weighted performance selection (default)."""
        if self.tta is not None:
            weights = self.tta.get_weights()
        else:
            weights = {name: 1.0 / len(proposals) for name in proposals}

        weighted_sum: torch.Tensor | None = None
        total_weight = 0.0
        for name, proposal in proposals.items():
            w = weights.get(name, 0.0) * proposal.confidence
            if weighted_sum is None:
                weighted_sum = w * proposal.action
            else:
                weighted_sum = weighted_sum + w * proposal.action
            total_weight += w

        if total_weight > 0 and weighted_sum is not None:
            return weighted_sum / total_weight
        return next(iter(proposals.values())).action

    def _select_consensus(self, proposals: Dict[str, AgentProposal]) -> torch.Tensor:
        """Consensus selection: average of all proposals."""
        actions = [p.action for p in proposals.values()]
        return torch.stack(actions).mean(dim=0)

    def _select_panic(self, proposals: Dict[str, AgentProposal]) -> torch.Tensor:
        """Panic mode: select highest-confidence proposal."""
        best = max(proposals.values(), key=lambda p: p.confidence)
        return best.action

    def record_outcome(self, success: bool, agent_name: Optional[str] = None) -> None:
        """Record outcome for tracking and TTA."""
        if not success:
            self.state.global_failures += 1
        if agent_name and agent_name in self.agents:
            self.agents[agent_name].record_outcome(success)

    def get_status(self) -> Dict[str, object]:
        """Get orchestrator status summary."""
        return {
            "step_count": self.state.step_count,
            "global_failures": self.state.global_failures,
            "is_panic": self.state.is_panic,
            "active_agents": [n for n, a in self.agents.items() if not a.is_suppressed],
            "tta_weights": self.tta.get_weights() if self.tta else None,
            "tta_adapting": self.tta.should_adapt() if self.tta else False,
        }
