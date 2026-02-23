"""METR-HRS Baseline Predictors.

Three reference agents used as scoring anchors in the HRS evaluation:

  ZeroPredictor    — always predicts zero  (theoretical lower bound)
  IdentityPredictor — persistence forecast (predict current obs as next)
  LinearPredictor  — linear extrapolation from last two observations
"""
import torch
from typing import Optional

from ..agents.base_agent import BaseAgent, AgentProposal


class ZeroPredictor(BaseAgent):
    """Always predicts the zero vector.

    Serves as the lower-bound anchor for HRS scores:
        HRS = max(0, 1 - mse_predictor / mse_zero)
    Any predictor worse than zero scores ≤ 0.
    """

    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__("zero_predictor", observation_dim, action_dim)

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        return AgentProposal(
            agent_name=self.name,
            action=torch.zeros(self.action_dim),
            confidence=1.0,
        )

    def get_capacity_metrics(self):
        return {"model_type": "ZeroPredictor", "agent_name": self.name, "params": 0}


class IdentityPredictor(BaseAgent):
    """Persistence / identity forecast.

    Predicts that the next observation equals the current one.
    Strong on slowly-varying signals; fails on regime switches.
    """

    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__("identity_predictor", observation_dim, action_dim)

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        obs = observation[:self.action_dim]
        if obs.shape[0] < self.action_dim:
            obs = torch.cat([obs, torch.zeros(self.action_dim - obs.shape[0])])
        return AgentProposal(
            agent_name=self.name,
            action=obs.clone(),
            confidence=1.0,
        )

    def get_capacity_metrics(self):
        return {"model_type": "IdentityPredictor", "agent_name": self.name, "params": 0}


class LinearPredictor(BaseAgent):
    """Linear extrapolation from the last two observations.

    Computes:  a_t = obs_t + (obs_t - obs_{t-1})

    Beats identity on linear-drift signals; still fails on chaotic ones.
    Requires at least two observations — uses identity for the first two steps.
    """

    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__("linear_predictor", observation_dim, action_dim)
        self._last: Optional[torch.Tensor] = None  # previous observation

    def _clip_obs(self, obs: torch.Tensor) -> torch.Tensor:
        o = obs[:self.action_dim]
        if o.shape[0] < self.action_dim:
            o = torch.cat([o, torch.zeros(self.action_dim - o.shape[0])])
        return o

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        obs = self._clip_obs(observation)

        if self._last is None:
            # First step — no delta available yet
            action = obs.clone()
        else:
            # Extrapolate: obs_t + (obs_t - obs_{t-1})
            action = obs + (obs - self._last)

        self._last = obs.clone()

        return AgentProposal(
            agent_name=self.name,
            action=action,
            confidence=1.0,
        )

    def reset_state(self) -> None:
        self._last = None

    def get_capacity_metrics(self):
        return {"model_type": "LinearPredictor", "agent_name": self.name, "params": 0}
