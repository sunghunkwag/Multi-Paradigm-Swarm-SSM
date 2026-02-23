import torch
import numpy as np
from typing import Dict, Any, Tuple
from .seq_prediction_env import SequentialPredictionEnv, EnvConfig

class HighDimChaosEnv(SequentialPredictionEnv):
    """
    Omega Singularity Phase 1: Universal Symbolic Physics.
    The world laws are no longer hardcoded. They evolve from math primitives.
    """
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.D = config.observation_dim
        self.dt_min, self.dt_max = 0.0001, 0.005
        self.sigma_base = 0.02
        self.states = np.random.uniform(-0.1, 0.1, (self.D, 3))
        
        # OMEGA: Evolving Physics PDEs
        # The world itself is a swarm of symbolic expressions.
        from ssm_mamba_swarm.agents.symbolic_agent import VoidNode
        self.laws = []
        for i in range(self.D):
            # Each dimension has an evolving 3D ODE branch
            branch = [self._gen_physics_node() for _ in range(3)]
            self.laws.append(branch)
            
    def _gen_physics_node(self):
        from ssm_mamba_swarm.agents.symbolic_agent import VoidNode
        # Simple hard-coded seeds for initial chaos, but depth is unbounded
        ops = ['+', '*', 'exp', 'sin', 'cos', 'log', 'sqrt', 'p', 's', 'c']
        return VoidNode(np.random.choice(ops), p_idx=np.random.randint(0, 3))

    def _drift(self, s: np.ndarray) -> np.ndarray:
        f = np.zeros_like(s)
        for i in range(self.D):
            x, y, z = s[i]
            # World laws are evaluated just like agent proposals
            for axis in range(3):
                # Using a dummy x_s for the env's internal laws
                f[i, axis] = self.laws[i][axis].eval(x, y, [10.0, 28.0, 8/3])
        return f

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # Void Veracity: Adaptive Stochastic Time-Stepping
        t_accum = 0.0
        t_target = 0.05 # Fixed observation step, but internal steps are adaptive
        
        while t_accum < t_target:
            f = self._drift(self.states)
            # Estimate Local Volatilty for dt selection
            # Higher f/sigma magnitude -> smaller dt
            vol = np.max(np.abs(f)) + self.sigma_base * np.max(np.sqrt(np.abs(self.states)))
            dt = np.clip(0.01 / (vol + 1e-6), self.dt_min, min(self.dt_max, t_target - t_accum))
            
            g = self.sigma_base * np.sqrt(np.abs(self.states))
            gg_prime = 0.5 * (self.sigma_base**2) * np.sign(self.states)
            dW = np.random.normal(0, 1, self.states.shape)
            
            # Milstein Update with adaptive dt
            self.states += f * dt + g * np.sqrt(dt) * dW + \
                           0.5 * gg_prime * ( (dW**2)*dt - dt )
            t_accum += dt
            
        if np.any(np.isnan(self.states)) or np.any(np.isinf(self.states)):
             # THE EVENT HORIZON: No reward floor. No interpreted failure.
             return torch.full((self.D,), np.nan), np.nan, True, {"mse": 1e9, "failed": True, "reason": "Event Horizon Collapse"}
        
        # OMEGA: Fractal Dimensionality (The expansion/contraction of reality)
        drift_vals = self._drift(self.states)
        volatility = np.mean(np.abs(drift_vals))
        if self.step_count % 10 == 0:
            self._check_fractal_scaling(volatility)
            
        next_obs = self._get_obs()
        mse = torch.mean((action - next_obs)**2).item()
        done = self.step_count >= self.config.sequence_length
        return next_obs, -mse, done, {"mse": mse, "omega": True, "D": self.D}

    def _check_fractal_scaling(self, volatility: float):
        """OMEGA: Reality expands on high entropy, contracts on low entropy."""
        if volatility > 50.0 and self.D < 128:
            self.D += 1
            new_state = np.random.uniform(-0.1, 0.1, (1, 3))
            self.states = np.vstack([self.states, new_state])
            self.laws.append([self._gen_physics_node() for _ in range(3)])
        elif volatility < 0.1 and self.D > 8:
            self.D -= 1
            self.states = self.states[:-1]
            self.laws.pop()

    def _get_obs(self) -> torch.Tensor:
        # THE OMEGA: Pure, raw observation.
        obs_vals = self.states[:, 0]
        return torch.tensor(obs_vals, dtype=torch.float32)

    def reset(self) -> torch.Tensor:
        super().reset()
        self.step_count = 0
        self.states = np.random.uniform(-0.05, 0.05, (self.D, 3))
        return self._get_obs()
