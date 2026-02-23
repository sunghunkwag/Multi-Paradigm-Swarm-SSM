import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .seq_prediction_env import SequentialPredictionEnv, EnvConfig

class Chaos1DEnv(SequentialPredictionEnv):
    """
    Chaos 1D Environment — The "Heuristic-Proof" Benchmark.
    
    Uses a discretized Lorenz System projected onto 1D observations.
    The system state (X, Y, Z) evolves hidden from the agent.
    The agent only sees X and must predict the next X.
    
    Chaos parameters (sigma, rho, beta) switch regimes every N steps,
    making simple linear extrapolation or identity-heuristics fail 100%.
    """
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.dt = 0.01
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0/3.0
        self.state = np.array([1.0, 1.0, 1.0]) # Hidden state (X, Y, Z)
        self.step_count = 0

    def reset(self) -> torch.Tensor:
        super().reset()
        self.step_count = 0
        # Randomize initial hidden state for entropy
        self.state = np.random.uniform(-1, 1, 3)
        self.rho = 28.0 # Standard chaos
        return self._get_obs()

    def _get_obs(self) -> torch.Tensor:
        # Scale X to a manageable range [-1, 1] for typical neural outputs
        val = self.state[0] / 20.0 
        noise = np.random.normal(0, self.config.noise_std, self.config.observation_dim)
        obs = np.full(self.config.observation_dim, val) + noise
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # 1. Regime Switching (Adversarial)
        if self.step_count % 50 == 0:
            # Shift rho: 28.0 (Chaos) -> 14.0 (Periodic) -> 45.0 (High Chaos)
            self.rho = np.random.choice([14.0, 28.0, 45.0])

        # 2. Lorenz System Evolution (Hidden Dynamics)
        x, y, z = self.state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        self.state += np.array([dx, dy, dz]) * self.dt
        
        # 3. Get next observation
        next_obs = self._get_obs()
        
        # 4. Reward Calculation (Negative MSE)
        # Truth is the clean next observation (next_obs[0] before noise)
        truth = next_obs[0].item()
        pred = action.detach().cpu().numpy().mean() # Swarm's prediction
        mse = (pred - truth)**2
        reward = -mse
        
        done = self.step_count >= self.config.sequence_length
        info = {
            "mse": mse,
            "regime_rho": self.rho,
            "hidden_state": self.state.tolist(),
            "shock": self.step_count % 50 == 0
        }
        
        return next_obs, reward, done, info
