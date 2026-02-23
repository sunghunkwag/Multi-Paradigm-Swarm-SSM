"""Unified Configuration Module for SSM-Mamba Swarm.

Merges configurations from:
  - heterogeneous-agent-swarm: Agent configs, Orchestrator params, MetaKernel settings
  - SSM-MetaRL-TestCompute: MambaSSM hyperparams, MAML settings, TTA config
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MambaConfig:
    """Configuration for the MambaSSM core module."""
    state_dim: int = 16          # SSM state expansion factor (N)
    d_model: int = 64            # Internal model dimension
    d_conv: int = 4              # 1D convolution kernel width
    expand: int = 2              # Block expansion factor
    device: str = "cpu"


@dataclass
class MAMLConfig:
    """Configuration for MAML meta-learning."""
    inner_lr: float = 0.01       # Task-level adaptation learning rate
    outer_lr: float = 0.001      # Meta-learning rate
    first_order: bool = False    # First-order MAML (faster but less accurate)
    inner_steps: int = 1         # Number of inner gradient steps


@dataclass
class TTAConfig:
    """Configuration for Test-Time Adaptation."""
    learning_rate: float = 0.01
    num_steps: int = 5
    grad_clip_norm: Optional[float] = 1.0
    shock_threshold: float = 2.0  # Std-devs above mean to detect distributional shift


@dataclass
class AgentConfig:
    """Per-agent configuration."""
    name: str = ""
    enabled: bool = True
    capacity: float = 1.0        # Resource allocation weight [0, 1]
    suppression_threshold: int = 5  # Consecutive failures before suppression


@dataclass
class OrchestratorConfig:
    """Configuration for the Swarm Orchestrator."""
    selection_mode: str = "weighted_perf"   # weighted_perf | consensus | panic
    panic_threshold: int = 10               # Global failure count to trigger panic
    tta_enabled: bool = True                # Enable test-time adaptation in orchestrator


@dataclass
class LiquidConfig:
    """Configuration for ncps Liquid Controller."""
    hidden_size: int = 32
    model_type: str = "cfc"      # cfc | ltc
    backbone_units: int = 32
    backbone_layers: int = 1
    backbone_dropout: float = 0.0

@dataclass
class SNNConfig:
    """Configuration for snntorch Reflex Agent."""
    beta: float = 0.9           # Decay rate
    threshold: float = 1.0       # Spike threshold
    hidden_size: int = 128

@dataclass
class MetaKernelConfig:
    """Configuration for MetaKernelV2 (self-modification engine)."""
    quorum_fraction: float = 0.5     # Votes needed to approve a change
    nas_enabled: bool = True         # Enable Neural Architecture Search
    maml_enabled: bool = True        # Enable MAML-based meta-optimization
    constraint_hard_limit: int = 8   # Max agents allowed simultaneously


@dataclass
class SwarmConfig:
    """Top-level configuration for the entire SSM-Mamba Swarm."""
    # Sub-configs
    mamba: MambaConfig = field(default_factory=MambaConfig)
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    tta: TTAConfig = field(default_factory=TTAConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    meta_kernel: MetaKernelConfig = field(default_factory=MetaKernelConfig)
    liquid: LiquidConfig = field(default_factory=LiquidConfig)
    snn: SNNConfig = field(default_factory=SNNConfig)

    # Agent roster
    agents: Dict[str, AgentConfig] = field(default_factory=lambda: {
        "symbolic_search": AgentConfig(name="symbolic_search"),
        "jepa_world_model": AgentConfig(name="jepa_world_model"),
        "liquid_controller": AgentConfig(name="liquid_controller"),
        "ssm_stability": AgentConfig(name="ssm_stability"),
        "snn_reflex": AgentConfig(name="snn_reflex"),
    })

    # Global settings
    observation_dim: int = 32        # Dimension of environment observations
    action_dim: int = 32             # Dimension of action space
    seed: int = 42
    device: str = "cpu"

    def get_agent_names(self) -> List[str]:
        return [name for name, cfg in self.agents.items() if cfg.enabled]
