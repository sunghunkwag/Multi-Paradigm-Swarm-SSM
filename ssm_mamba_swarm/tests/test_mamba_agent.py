"""Unit tests for MambaSSM and the upgraded SSMStabilityAgent."""
import torch
import pytest
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ssm_mamba_swarm.core.mamba_core import MambaSSM, MambaBlockFallback
from ssm_mamba_swarm.core.config import MambaConfig
from ssm_mamba_swarm.agents.ssm_stability import SSMStabilityAgent


class TestMambaSSM:
    """Tests for the MambaSSM core module."""

    def test_single_step_forward(self):
        """MambaSSM should handle single-step (B, D) input."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)
        x = torch.randn(4, 32)
        hidden = model.init_hidden(4)
        output, next_h = model(x, hidden)
        assert output.shape == (4, 32), f"Expected (4, 32), got {output.shape}"

    def test_sequence_forward(self):
        """MambaSSM should handle sequence (B, T, D) input."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)
        x = torch.randn(4, 20, 32)
        output, _ = model(x, None)
        assert output.shape == (4, 20, 32), f"Expected (4, 20, 32), got {output.shape}"

    def test_backward(self):
        """MambaSSM should support gradient computation."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)
        x = torch.randn(4, 10, 32)
        output, _ = model(x, None)
        loss = output.mean()
        loss.backward()
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed!"

    def test_complexity_info(self):
        """MambaSSM should report complexity metrics."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)
        info = model.get_complexity_info()
        assert info["model_type"] == "MambaSSM"
        assert info["total_params"] > 0
        assert info["d_model"] == 64

    def test_save_load(self, tmp_path):
        """MambaSSM should save and load checkpoints."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)
        path = str(tmp_path / "test_mamba.pt")
        model.save(path)

        loaded = MambaSSM.load(path)
        assert loaded.d_model == 64
        assert loaded.input_dim == 32


class TestMambaBlockFallback:
    """Tests for the pure-PyTorch Mamba fallback."""

    def test_forward(self):
        block = MambaBlockFallback(d_model=64, d_state=16)
        x = torch.randn(2, 10, 64)
        output, hidden = block(x, None)
        assert output.shape == (2, 10, 64)
        assert hidden.shape[0] == 2  # batch size

    def test_state_propagation(self):
        """Hidden state should change after processing a sequence."""
        block = MambaBlockFallback(d_model=64, d_state=16)
        x = torch.randn(2, 10, 64)
        _, h1 = block(x, None)
        _, h2 = block(x, h1)
        # h2 should differ from h1 (state was propagated)
        assert not torch.allclose(h1, h2), "State should change across calls"


class TestSSMStabilityAgent:
    """Tests for the upgraded SSMStabilityAgent."""

    def test_propose(self):
        """Agent should produce valid proposals."""
        agent = SSMStabilityAgent(observation_dim=32, action_dim=32)
        obs = torch.randn(32)
        proposal = agent.propose(obs)
        assert proposal.agent_name == "ssm_stability"
        assert proposal.action.shape == (1, 32) or proposal.action.shape == (32,)
        assert 0.0 <= proposal.confidence <= 1.0

    def test_suppression(self):
        """Suppressed agent should return zero-confidence proposals."""
        agent = SSMStabilityAgent(observation_dim=32, action_dim=32)
        agent.suppress()
        obs = torch.randn(32)
        proposal = agent.propose(obs)
        assert proposal.confidence == 0.0
        assert proposal.metadata["status"] == "suppressed"

    def test_recovery(self):
        """Agent should produce normal proposals after recovery."""
        agent = SSMStabilityAgent(observation_dim=32, action_dim=32)
        agent.suppress()
        agent.recover()
        obs = torch.randn(32)
        proposal = agent.propose(obs)
        assert proposal.confidence > 0.0

    def test_capacity_scaling(self):
        """NAS capacity changes should modify actual d_model."""
        agent = SSMStabilityAgent(
            observation_dim=32, action_dim=32,
            mamba_config=MambaConfig(d_model=64),
        )
        assert agent.mamba.d_model == 64

        agent.increase_capacity()
        assert agent.mamba.d_model == 96  # 64 * 1.5

        agent.decrease_capacity()
        assert agent.mamba.d_model == 64  # 96 * 0.67 ≈ 64

    def test_hidden_state_persistence(self):
        """Hidden state should persist across propose() calls."""
        agent = SSMStabilityAgent(observation_dim=32, action_dim=32)
        obs1 = torch.randn(32)
        obs2 = torch.randn(32)
        agent.propose(obs1)
        h_after_1 = agent._hidden_state
        agent.propose(obs2)
        h_after_2 = agent._hidden_state
        # States should differ (temporal processing)
        if h_after_1 is not None and h_after_2 is not None:
            assert not torch.allclose(h_after_1, h_after_2)

    def test_stability_score(self):
        """Stability score should be computable."""
        agent = SSMStabilityAgent(observation_dim=32, action_dim=32)
        for _ in range(5):
            agent.propose(torch.randn(32))
        score = agent.get_stability_score()
        assert 0.0 <= score <= 1.0

    def test_capacity_metrics(self):
        """Capacity metrics should include MambaSSM info."""
        agent = SSMStabilityAgent(observation_dim=32, action_dim=32)
        metrics = agent.get_capacity_metrics()
        assert "model_type" in metrics
        assert metrics["model_type"] == "MambaSSM"
        assert metrics["agent_name"] == "ssm_stability"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
