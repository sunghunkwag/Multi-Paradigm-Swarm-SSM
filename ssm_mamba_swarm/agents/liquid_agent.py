import torch
import torch.nn as nn
from ncps.torch import CfC
from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentProposal

class LiquidInternal(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(observation_dim, hidden_dim)
        self.cfc = CfC(hidden_dim, hidden_dim, proj_size=action_dim, backbone_layers=num_layers)

class LiquidControllerAgent(BaseAgent):
    """
    Structural Sincerity Phase 3: Compositional Liquid Net.
    Decoupled from nn.Module.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__("liquid_controller", observation_dim, action_dim)
        self.model = LiquidInternal(observation_dim, action_dim, hidden_dim, num_layers)
        self.state = None

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        if self.is_suppressed:
            return AgentProposal(self.name, torch.zeros(self.action_dim), 0.0)
        
        x = observation.view(1, 1, -1)
        with torch.no_grad():
            x_proj = self.model.input_proj(x)
            out, self.state = self.model.cfc(x_proj, self.state)
            action = torch.tanh(out.squeeze())
        return AgentProposal(self.name, action, 0.8)

    def reset_state(self):
        self.state = None

    def get_capacity_metrics(self) -> Dict[str, Any]:
        return {"agent_name": self.name, "params": sum(p.numel() for p in self.model.parameters())}
