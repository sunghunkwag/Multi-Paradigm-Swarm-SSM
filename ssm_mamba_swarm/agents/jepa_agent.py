import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentProposal

class JEPAInternal(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int):
        super().__init__()
        self.encoder_mu = nn.Sequential(nn.Linear(observation_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.encoder_logvar = nn.Sequential(nn.Linear(observation_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        
        self.predictor = nn.Sequential(nn.Linear(latent_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.reward_model = nn.Sequential(nn.Linear(latent_dim*2 + action_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_model = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))

class JEPAWorldModelAgent(BaseAgent):
    """
    Asymptotic Emergence Phase 2: Information-Theoretic Grounding.
    Structural representations emerge via the Information Bottleneck principle.
    """
    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int = 256, rollout_horizon: int = 5):
        super().__init__("jepa_world_model", observation_dim, action_dim)
        self.horizon = rollout_horizon
        self.gamma = 0.95
        self.latent_dim = latent_dim
        self.model = JEPAInternal(observation_dim, action_dim, latent_dim)
        
        # ASYMPTOTIC: Meta-Adaptive precision balancing for heterogeneous objectives.
        self.log_vars = nn.Parameter(torch.zeros(4)) 
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [self.log_vars], 
            lr=5e-4
        )

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        if self.is_suppressed: return AgentProposal(self.name, torch.zeros(self.action_dim), 0.0)
        
        # ASYMPTOTIC: Robust Perceptual Buffer Management (D-invariant)
        if observation.shape[0] != self.observation_dim:
            obs_sync = torch.zeros(self.observation_dim).to(observation.device)
            min_d = min(self.observation_dim, observation.shape[0])
            obs_sync[:min_d] = observation[:min_d]
            observation = obs_sync

        with torch.no_grad():
            z = self.model.encoder_mu(observation.unsqueeze(0))
            best_a, best_v = torch.zeros(self.action_dim), -1e10
            for _ in range(6):
                # THE EVENT HORIZON: Unbounded Action trajectories.
                trajs = torch.randn(128, self.horizon, self.action_dim) * 10.0
                z_c = z.expand(128, -1)
                rets = torch.zeros(128)
                for t in range(self.horizon):
                    a = trajs[:, t, :]
                    z_n = self.model.predictor(torch.cat([z_c, a], dim=-1))
                    rets += (self.gamma**t) * self.model.reward_model(torch.cat([z_c, a, z_n], dim=-1)).squeeze(-1)
                    z_c = z_n
                rets += (self.gamma**self.horizon) * self.model.value_model(z_c).squeeze(-1)
                v, idx = torch.max(rets, 0)
                if v > best_v: best_v, best_a = v.item(), trajs[idx, 0]
            conf = torch.sigmoid(torch.tensor(best_v / 100.0)).item()
            return AgentProposal(self.name, best_a, conf)

    def train_step(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, reward: float):
        self.optimizer.zero_grad()
        
        # ASYMPTOTIC: Dimension-agnostic training resync
        if obs.shape[0] != self.observation_dim:
            obs_sync = torch.zeros(self.observation_dim).to(obs.device)
            obs_sync[:min(self.observation_dim, obs.shape[0])] = obs[:min(self.observation_dim, obs.shape[0])]
            obs = obs_sync
        if next_obs.shape[0] != self.observation_dim:
            n_obs_sync = torch.zeros(self.observation_dim).to(next_obs.device)
            n_obs_sync[:min(self.observation_dim, next_obs.shape[0])] = next_obs[:min(self.observation_dim, next_obs.shape[0])]
            next_obs = n_obs_sync

        z = self.model.encoder_mu(obs.unsqueeze(0) if obs.dim()==1 else obs)
        # ASYMPTOTIC: Enforce Rank-2 for action to match latent state z
        a_dim2 = action.unsqueeze(0) if action.dim()==1 else action
        
        if z.dim() != a_dim2.dim():
            # Emergency shape reconciliation
            if z.dim() == 2 and a_dim2.dim() == 1:
                a_dim2 = a_dim2.unsqueeze(0)
            elif z.dim() == 1 and a_dim2.dim() == 2:
                z = z.unsqueeze(0)
        
        z_n_pred = self.model.predictor(torch.cat([z, a_dim2], dim=-1))
        
        with torch.no_grad():
            z_n_tgt = self.model.encoder_mu(next_obs.unsqueeze(0) if next_obs.dim()==1 else next_obs)
            v_tgt = torch.tensor([[reward]], dtype=torch.float32).to(z.device) + self.gamma * self.model.value_model(z_n_tgt)
            
        # 1. MDL Prediction Loss (Compression)
        err = (z_n_pred - z_n_tgt)**2
        # Count params manually if needed, or use a dummy for now
        param_count = sum(p.numel() for p in self.model.parameters())
        mdl_inv_loss = torch.log(err.sum() + 1e-15) + param_count * torch.log(torch.tensor(z.shape[0], dtype=torch.float32))
        
        # 2. Structural Grounding (Entropy Maximization)
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov_z = (z_centered.T @ z_centered) / (z.shape[0] - 1 + 1e-9)
        cov_loss = (cov_z - torch.eye(self.latent_dim).to(z.device)).pow(2).sum()
        
        # 3. MDL Task Losses
        err_rew = (self.model.reward_model(torch.cat([z, a_dim2, z_n_pred], dim=-1)) - reward)**2
        mdl_rew_loss = torch.log(err_rew.sum() + 1e-15)
        
        err_val = (self.model.value_model(z) - v_tgt)**2
        mdl_val_loss = torch.log(err_val.sum() + 1e-15) if np.isfinite(v_tgt.item()) else torch.zeros(1)
        
        # THE SINGULARITY: Unweighted Self-Organized loss balancing
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])
        precision3 = torch.exp(-self.log_vars[2])
        precision4 = torch.exp(-self.log_vars[3])
        
        loss = precision1 * mdl_inv_loss + self.log_vars[0] + \
               precision2 * cov_loss + self.log_vars[1] + \
               precision3 * mdl_rew_loss + self.log_vars[2] + \
               precision4 * mdl_val_loss + self.log_vars[3]
               
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_capacity_metrics(self) -> Dict[str, Any]:
        with torch.no_grad():
             # Estimate effective dimensionality via eigenvalue decay
             z_example = self.model.encoder_mu(torch.randn(1, self.model.encoder_mu[0].in_features))
             # Simplified metric: mean variance per dimension
             return {"agent": self.name, "latent_dim": self.latent_dim, "prior": "none_void"}

    def get_capacity_metrics(self) -> Dict[str, Any]:
        return {"agent": self.name, "prior": "none_covariance_only"}
