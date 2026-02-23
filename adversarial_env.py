import torch
import numpy as np
from typing import Dict, Any, Tuple
from ssm_mamba_swarm.envs.seq_prediction_env import SequentialPredictionEnv, EnvConfig

class AdversarialEntropyEnv(SequentialPredictionEnv):
    """
    High-Entropy Adversarial Environment.
    Adds non-linear shocks, regime switching, and chaotic noise.
    Forces agents to use real JEPA search and Symbolic Induction.
    """
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.regime = 0
        self.step_count = 0
        self.current_value = 0.5 # Initial state

    def reset(self) -> torch.Tensor:
        super().reset()
        self.step_count = 0
        self.regime = 0
        self.current_value = 0.5
        obs = np.full(self.config.observation_dim, self.current_value)
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # 1. Regime Switching (Adversarial)
        if self.step_count % 30 == 0:
            self.regime = (self.regime + 1) % 3
            # Random shock
            self.current_value += np.random.randn() * 0.5
            
        # 2. Non-linear Transition
        if self.regime == 2:
            # Chaotic logistic map
            self.current_value = 3.9 * self.current_value * (1 - self.current_value) if 0 < self.current_value < 1 else 0.5
        elif self.regime == 1:
            # Oscillatory
            self.current_value = np.sin(self.step_count * 0.2)
        else:
            # Linear drift
            self.current_value += np.random.normal(0, 0.05)

        # 3. Add High-Frequency Noise
        noise = np.random.normal(0, 0.1, self.config.observation_dim)
        obs = np.full(self.config.observation_dim, self.current_value) + noise
        obs_torch = torch.tensor(obs, dtype=torch.float32)
        
        # 4. Reward (Negative MSE)
        mse = torch.mean((action - obs_torch)**2).item()
        reward = -mse
        
        done = self.step_count >= self.config.sequence_length
        info = {"mse": mse, "regime": self.regime, "shock": self.step_count % 30 == 0}
        
        return obs_torch, reward, done, info
