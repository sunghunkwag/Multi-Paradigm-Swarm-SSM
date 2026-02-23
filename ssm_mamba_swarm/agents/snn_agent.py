import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentProposal

class SNNInternal(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, beta: float, threshold: float):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, 64)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(64, action_dim)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())

class SNNReflexAgent(BaseAgent):
    """
    Structural Sincerity Phase 3: Compositional SNN.
    Decoupled from nn.Module.
    """
    def __init__(self, observation_dim: int, action_dim: int, beta: float = 0.95, threshold: float = 1.0):
        super().__init__("snn_reflex", observation_dim, action_dim)
        self.model = SNNInternal(observation_dim, action_dim, beta, threshold)
        self.mem1 = None
        self.mem2 = None

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        if self.is_suppressed:
            return AgentProposal(self.name, torch.zeros(self.action_dim), 0.0)
        
        # ASYMPTOTIC: Robust Perceptual Buffer Management (D-invariant)
        if observation.shape[0] != self.observation_dim:
            obs_sync = torch.zeros(self.observation_dim).to(observation.device)
            min_d = min(self.observation_dim, observation.shape[0])
            obs_sync[:min_d] = observation[:min_d]
            observation = obs_sync

        x = observation.unsqueeze(0)
        if self.mem1 is None:
            self.mem1 = self.model.lif1.init_leaky()
            self.mem2 = self.model.lif2.init_leaky()

        with torch.no_grad():
            spk1, self.mem1 = self.model.lif1(self.model.fc1(x), self.mem1)
            spk2, self.mem2 = self.model.lif2(self.model.fc2(spk1), self.mem2)
            action = torch.tanh(self.mem2.squeeze())
        return AgentProposal(self.name, action, 0.7)

    def reset_state(self):
        self.mem1 = None
        self.mem2 = None

    def get_capacity_metrics(self) -> Dict[str, Any]:
        return {"agent_name": self.name, "params": sum(p.numel() for p in self.model.parameters())}
