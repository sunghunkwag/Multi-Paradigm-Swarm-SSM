"""Integration test for the full SSM-Mamba Swarm hybrid architecture.

Tests the complete pipeline:
  1. Swarm assembly (all 6 agents)
  2. Orchestrator action selection with TTA
  3. MetaKernel self-modification (suppression, NAS, recovery)
  4. Sequential prediction benchmark
  5. MAML meta-learning on MambaSSM
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ssm_mamba_swarm.core.config import SwarmConfig
from ssm_mamba_swarm.core.mamba_core import MambaSSM
from ssm_mamba_swarm.core.meta_maml import MetaMAML
from ssm_mamba_swarm.core.orchestrator import Orchestrator
from ssm_mamba_swarm.core.meta_kernel import MetaKernelV2, ChangeProposal
from ssm_mamba_swarm.agents.ssm_stability import SSMStabilityAgent
from ssm_mamba_swarm.agents import (
    SymbolicSearchAgent, JEPAWorldModelAgent, LiquidControllerAgent,
    SNNReflexAgent, SSMStabilityAgent,
)
from ssm_mamba_swarm.envs.seq_prediction_env import SequentialPredictionEnv, EnvConfig


def _build_swarm(config=None):
    """Helper: construct a full 6-agent swarm."""
    if config is None:
        config = SwarmConfig()
    agents = {
        "symbolic_search": SymbolicSearchAgent(config.observation_dim, config.action_dim),
        "jepa_world_model": JEPAWorldModelAgent(config.observation_dim, config.action_dim),
        "liquid_controller": LiquidControllerAgent(config.observation_dim, config.action_dim),
        "ssm_stability": SSMStabilityAgent(config.observation_dim, config.action_dim, config.mamba),
        "snn_reflex": SNNReflexAgent(config.observation_dim, config.action_dim),
    }
    return agents, config


class TestSwarmAssembly:
    """Test that the full swarm can be assembled."""

    def test_all_agents_created(self):
        agents, config = _build_swarm()
        assert len(agents) == 5
        assert "ssm_stability" in agents
        assert isinstance(agents["ssm_stability"], SSMStabilityAgent)

    def test_all_agents_can_propose(self):
        agents, config = _build_swarm()
        obs = torch.randn(config.observation_dim)
        for name, agent in agents.items():
            proposal = agent.propose(obs)
            assert proposal.agent_name == name
            assert proposal.action is not None


class TestOrchestratorIntegration:
    """Test the orchestrator with TTA across all agents."""

    def test_select_action(self):
        agents, config = _build_swarm()
        orch = Orchestrator(agents, config.orchestrator, config.tta)
        obs = torch.randn(config.observation_dim)
        action, proposals = orch.select_action(obs)
        assert action.shape[-1] == config.action_dim or action.numel() == config.action_dim

    def test_tta_weight_adaptation(self):
        agents, config = _build_swarm()
        orch = Orchestrator(agents, config.orchestrator, config.tta)

        # Simulate many steps with ground truth to trigger TTA
        for i in range(20):
            obs = torch.randn(config.observation_dim)
            gt = torch.randn(config.action_dim)
            orch.select_action(obs, ground_truth=gt)

        status = orch.get_status()
        assert status["step_count"] == 20
        assert status["tta_weights"] is not None

    def test_panic_mode(self):
        agents, config = _build_swarm()
        config.orchestrator.panic_threshold = 3
        orch = Orchestrator(agents, config.orchestrator, config.tta)
        for _ in range(5):
            orch.record_outcome(False)
        # Panic is evaluated during select_action, so call it after failures
        obs = torch.randn(config.observation_dim)
        orch.select_action(obs)
        assert orch.state.is_panic


class TestMetaKernelIntegration:
    """Test MetaKernel self-modification with the full swarm."""

    def test_suppression_flow(self):
        agents, config = _build_swarm()
        mk = MetaKernelV2(agents, config.meta_kernel)

        # Simulate failures for ssm_stability
        for _ in range(6):
            agents["ssm_stability"].record_outcome(False)

        proposals = mk.check_agent_health(suppression_threshold=5)
        assert len(proposals) >= 1
        assert proposals[0].target_agent == "ssm_stability"

        mk.propose_change(proposals[0])
        approved = mk.vote_on_proposals()
        logs = mk.execute_proposals(approved)
        assert "SUPPRESSED" in logs[0]
        assert agents["ssm_stability"].is_suppressed

    def test_nas_capacity_scaling(self):
        agents, config = _build_swarm()
        mk = MetaKernelV2(agents, config.meta_kernel)

        original_d_model = agents["ssm_stability"].mamba.d_model
        mk.auto_execute_nas("ssm_stability", increase=True)
        assert agents["ssm_stability"].mamba.d_model > original_d_model

    def test_emergency_rotation(self):
        agents, config = _build_swarm()
        mk = MetaKernelV2(agents, config.meta_kernel)

        agents["liquid_controller"].suppress()
        agents["snn_reflex"].suppress()
        result = mk.emergency_rotation()
        assert "EMERGENCY ROTATION" in result
        # At least one agent should be recovered
        recovered_count = sum(1 for a in agents.values() if not a.is_suppressed)
        assert recovered_count >= 4


class TestBenchmarkIntegration:
    """Test the swarm on the sequential prediction benchmark."""

    def test_swarm_on_seq_prediction(self):
        agents, config = _build_swarm()
        orch = Orchestrator(agents, config.orchestrator, config.tta)

        env = SequentialPredictionEnv(EnvConfig(
            observation_dim=config.observation_dim,
            sequence_length=50,
            pattern="sinusoidal",
        ))

        obs = env.reset()
        total_reward = 0.0
        for _ in range(49):
            action, proposals = orch.select_action(obs, ground_truth=obs)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs
            if done:
                break

        # Total reward should be negative (MSE-based) but finite
        assert total_reward < 0
        assert not torch.isnan(torch.tensor(total_reward))


class TestMAMLIntegration:
    """Test MAML meta-learning on MambaSSM."""

    def test_maml_adapt_task(self):
        model = MambaSSM(state_dim=16, input_dim=16, output_dim=16, d_model=32)
        maml = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)

        support_x = torch.randn(4, 16)
        support_y = torch.randn(4, 16)
        fast_weights = maml.adapt_task(support_x, support_y, num_steps=1)
        assert len(fast_weights) > 0

    def test_maml_meta_update(self):
        model = MambaSSM(state_dim=16, input_dim=16, output_dim=16, d_model=32)
        maml = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)

        tasks = [
            (torch.randn(4, 16), torch.randn(4, 16),
             torch.randn(4, 16), torch.randn(4, 16))
            for _ in range(3)
        ]
        loss = maml.meta_update(tasks)
        assert isinstance(loss, float)
        assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
