import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentProposal
from ..core.mamba_core import MambaSSM
from ..core.config import MambaConfig

logger = logging.getLogger(__name__)

class SSMInternal(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, mamba_config: MambaConfig):
        super().__init__()
        self.mamba = MambaSSM(
            state_dim=mamba_config.state_dim,
            input_dim=observation_dim,
            output_dim=action_dim,
            d_model=mamba_config.d_model,
            d_conv=mamba_config.d_conv,
            expand=mamba_config.expand,
            device=mamba_config.device,
        )
        self.confidence_head = nn.Linear(action_dim, 1)

class SSMStabilityAgent(BaseAgent):
    """
    Structural Sincerity Phase 3: Compositional SSM.
    Decoupled from nn.Module.
    """
    def __init__(self, observation_dim: int, action_dim: int, mamba_config: Optional[MambaConfig] = None):
        super().__init__("ssm_stability", observation_dim, action_dim)
        if mamba_config is None: mamba_config = MambaConfig()
        self.mamba_config = mamba_config
        self.model = SSMInternal(observation_dim, action_dim, mamba_config)
        self._hidden_state = None
        self._observation_history = []

    @property
    def mamba(self) -> MambaSSM:
        """Expose the MambaSSM instance for NAS capacity scaling."""
        return self.model.mamba

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        if self.is_suppressed:
            return AgentProposal(self.name, torch.zeros(self.action_dim), 0.0,
                                 metadata={"status": "suppressed"})

        # ASYMPTOTIC: Robust Perceptual Buffer Management (D-invariant)
        if observation.shape[0] != self.observation_dim:
            obs_sync = torch.zeros(self.observation_dim).to(observation.device)
            min_d = min(self.observation_dim, observation.shape[0])
            obs_sync[:min_d] = observation[:min_d]
            observation = obs_sync

        with torch.no_grad():
            if observation.dim() == 1: observation = observation.unsqueeze(0)
            action, self._hidden_state = self.model.mamba(observation, self._hidden_state)
            conf = torch.sigmoid(self.model.confidence_head(action)).mean().item()

            self._observation_history.append(observation.detach())
            if len(self._observation_history) > 50: self._observation_history.pop(0)

        # ASYMPTOTIC: Enforce rank-1 action tensor for topological consistency
        return AgentProposal(self.name, action.detach().squeeze(), conf)

    def increase_capacity(self) -> None:
        """NAS: increase d_model by factor 1.5 and rebuild the internal model."""
        new_d_model = int(self.mamba_config.d_model * 1.5)
        self.mamba_config = MambaConfig(
            d_model=new_d_model,
            state_dim=self.mamba_config.state_dim,
            d_conv=self.mamba_config.d_conv,
            expand=self.mamba_config.expand,
            device=self.mamba_config.device,
        )
        self.model = SSMInternal(self.observation_dim, self.action_dim, self.mamba_config)
        self._hidden_state = None

    def decrease_capacity(self) -> None:
        """NAS: decrease d_model by factor 0.67 and rebuild the internal model."""
        new_d_model = max(16, int(self.mamba_config.d_model * 0.67))
        self.mamba_config = MambaConfig(
            d_model=new_d_model,
            state_dim=self.mamba_config.state_dim,
            d_conv=self.mamba_config.d_conv,
            expand=self.mamba_config.expand,
            device=self.mamba_config.device,
        )
        self.model = SSMInternal(self.observation_dim, self.action_dim, self.mamba_config)
        self._hidden_state = None

    def get_stability_score(self) -> float:
        """Compute a stability score in [0, 1] based on observation history variance."""
        if len(self._observation_history) < 2:
            return 1.0
        history = torch.stack(self._observation_history[-10:])
        variance = history.var(dim=0).mean().item()
        score = float(1.0 / (1.0 + variance))
        return max(0.0, min(1.0, score))

    def get_capacity_metrics(self) -> Dict[str, Any]:
        metrics = self.model.mamba.get_complexity_info()
        metrics["agent_name"] = self.name
        return metrics

    def reset_state(self):
        self._hidden_state = None
        self._observation_history.clear()
