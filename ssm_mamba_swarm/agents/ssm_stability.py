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

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        if self.is_suppressed:
            return AgentProposal(self.name, torch.zeros(self.action_dim), 0.0)
        
        with torch.no_grad():
            if observation.dim() == 1: observation = observation.unsqueeze(0)
            action, self._hidden_state = self.model.mamba(observation, self._hidden_state)
            conf = torch.sigmoid(self.model.confidence_head(action)).mean().item()
            
            self._observation_history.append(observation.detach())
            if len(self._observation_history) > 50: self._observation_history.pop(0)

        return AgentProposal(self.name, action.detach(), conf)

    def get_capacity_metrics(self) -> Dict[str, Any]:
        return {"agent_name": self.name, "params": sum(p.numel() for p in self.model.parameters())}

    def reset_state(self):
        self._hidden_state = None
        self._observation_history.clear()
