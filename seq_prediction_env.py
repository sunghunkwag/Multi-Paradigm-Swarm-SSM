"""Sequential Prediction Benchmark Environment.

A synthetic environment for testing the SSM-Mamba Swarm on temporal
prediction tasks, inspired by C-MAPSS from SSM-MetaRL-TestCompute.

The environment generates sequential time-series data with
configurable patterns (sinusoidal, linear degradation, regime switching)
and challenges the swarm to predict future observations.
"""
import torch
import numpy as np
import math
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the sequential prediction environment."""
    observation_dim: int = 32
    sequence_length: int = 100
    pattern: str = "degradation"   # sinusoidal | degradation | switching
    noise_std: float = 0.05
    regime_switch_prob: float = 0.01  # For switching pattern
    seed: int = 42


class SequentialPredictionEnv:
    """Sequential prediction benchmark for swarm evaluation.

    Generates time-series observations and requires the swarm to predict
    the next observation. This tests temporal modeling capability,
    particularly beneficial for the MambaSSM agent.

    Patterns:
        - sinusoidal: Multi-frequency sinusoidal signals
        - degradation: Linear degradation with noise (like C-MAPSS RUL)
        - switching: Regime-switching dynamics with abrupt changes
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        if config is None:
            config = EnvConfig()
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.step_count = 0
        self.current_regime = 0
        self.is_done = False

        # Generate the full sequence upfront
        self._sequence = self._generate_sequence()

    def _generate_sequence(self) -> torch.Tensor:
        """Generate a full sequence based on the configured pattern."""
        T = self.config.sequence_length
        D = self.config.observation_dim
        t = np.linspace(0, 4 * np.pi, T)

        if self.config.pattern == "sinusoidal":
            # Multi-frequency sinusoidal with per-dimension phase shifts
            phases = self.rng.uniform(0, 2 * np.pi, size=D)
            freqs = self.rng.uniform(0.5, 3.0, size=D)
            data = np.array([
                np.sin(freqs[d] * t + phases[d]) for d in range(D)
            ]).T  # (T, D)

        elif self.config.pattern == "degradation":
            # Linear degradation with noise — mimics run-to-failure
            slopes = self.rng.uniform(-0.02, 0.02, size=D)
            offsets = self.rng.uniform(-1, 1, size=D)
            data = np.array([
                offsets[d] + slopes[d] * np.arange(T)
                for d in range(D)
            ]).T

        elif self.config.pattern == "switching":
            # Regime-switching: alternate between two dynamics
            data = np.zeros((T, D))
            regime = 0
            for i in range(T):
                if self.rng.random() < self.config.regime_switch_prob:
                    regime = 1 - regime
                if regime == 0:
                    data[i] = np.sin(0.5 * t[i] + self.rng.uniform(0, 0.1, D))
                else:
                    data[i] = np.cos(1.5 * t[i] + self.rng.uniform(0, 0.1, D)) * 0.5
        else:
            raise ValueError(f"Unknown pattern: {self.config.pattern}")

        # Add noise
        noise = self.rng.normal(0, self.config.noise_std, size=(T, D))
        data = data + noise

        return torch.tensor(data, dtype=torch.float32)

    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""
        self.step_count = 0
        self.is_done = False
        self._sequence = self._generate_sequence()
        return self._sequence[0]

    def step(self, prediction: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Advance one timestep.

        Args:
            prediction: The swarm's prediction for the CURRENT observation

        Returns:
            Tuple of (next_observation, reward, done, info)
        """
        if self.is_done:
            raise RuntimeError("Environment is done. Call reset().")

        current_obs = self._sequence[self.step_count]
        # Reward = negative MSE between prediction and current observation
        mse = (prediction - current_obs).pow(2).mean().item()
        reward = -mse

        self.step_count += 1
        if self.step_count >= self.config.sequence_length:
            self.is_done = True
            next_obs = current_obs  # Return last observation
        else:
            next_obs = self._sequence[self.step_count]

        info = {
            "step": self.step_count,
            "mse": mse,
            "pattern": self.config.pattern,
        }

        return next_obs, reward, self.is_done, info

    def get_ground_truth(self, step: int) -> torch.Tensor:
        """Get ground truth observation at a specific step."""
        return self._sequence[min(step, len(self._sequence) - 1)]

    def get_full_sequence(self) -> torch.Tensor:
        """Get the full sequence tensor (T, D)."""
        return self._sequence.clone()
