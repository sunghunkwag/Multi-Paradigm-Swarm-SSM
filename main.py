"""SSM-Mamba Swarm: Main Benchmark Runner.

Assembles the full 6-agent swarm and runs it on the sequential prediction
benchmark, demonstrating the hybrid architecture in action.
"""
import torch
import logging
import argparse

from config import SwarmConfig
from orchestrator import Orchestrator
from meta_kernel import MetaKernelV2
from symbolic_agent import SymbolicSearchAgent
from jepa_agent import JEPAWorldModelAgent
from liquid_agent import LiquidControllerAgent
from snn_agent import SNNReflexAgent
from ssm_stability import SSMStabilityAgent
from seq_prediction_env import EnvConfig, SequentialPredictionEnv
from adversarial_env import AdversarialEntropyEnv
from chaos_1d_env import Chaos1DEnv
from high_dim_chaos_env import HighDimChaosEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("chaos_prediction_envs")


def build_swarm(config: SwarmConfig):
    """Construct the full 6-agent heterogeneous swarm."""
    agents = {
        "symbolic_search": SymbolicSearchAgent(config.observation_dim, config.action_dim),
        "jepa_world_model": JEPAWorldModelAgent(config.observation_dim, config.action_dim),
        "liquid_controller": LiquidControllerAgent(config.observation_dim, config.action_dim),
        "ssm_stability": SSMStabilityAgent(config.observation_dim, config.action_dim, config.mamba),
        "snn_reflex": SNNReflexAgent(config.observation_dim, config.action_dim),
    }
    return agents


def run_benchmark(config: SwarmConfig, env_config: EnvConfig | None = None):
    """Run the swarm on the sequential prediction benchmark."""
    if env_config is None:
        env_config = EnvConfig(
            observation_dim=config.observation_dim,
            sequence_length=100,
            pattern="degradation",
        )

    # Build swarm
    agents = build_swarm(config)
    orchestrator = Orchestrator(agents, config.orchestrator, config.tta)
    meta_kernel = MetaKernelV2(agents, config.meta_kernel)

    # Build environment (Fully Coupled Chaos)
    env = HighDimChaosEnv(env_config)

    logger.info("=" * 60)
    logger.info("SSM-Mamba Swarm — Chaos Prediction Benchmark")
    logger.info("=" * 60)
    logger.info(f"Agents: {list(agents.keys())}")
    logger.info("=" * 60)

    # Run benchmark
    obs = env.reset()
    total_reward = 0.0
    total_mse = 0.0
    step = 0

    while True:
        # Get swarm's prediction
        action, proposals = orchestrator.select_action(obs, ground_truth=obs)

        # Environment step
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        total_mse += info["mse"]
        step += 1

        # Training step for agents
        for name, agent in agents.items():
            if hasattr(agent, "train_step") and not agent.is_suppressed:
                try:
                    import inspect

                    sig = inspect.signature(agent.train_step)
                    if "reward" in sig.parameters:
                        agent.train_step(obs, action, next_obs, reward)
                    else:
                        agent.train_step(obs, action, next_obs)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Training failed for %s: %s", name, e)

        # Periodic health check and meta-kernel updates
        if step % 20 == 0:
            health_proposals = meta_kernel.check_agent_health()
            if health_proposals:
                for p in health_proposals:
                    meta_kernel.propose_change(p)
                approved = meta_kernel.vote_on_proposals()
                logs = meta_kernel.execute_proposals(approved)
                for log in logs:
                    logger.info("MetaKernel: %s", log)

            status = orchestrator.get_status()
            logger.info(
                "Step %d: MSE=%.6f, Reward=%.6f, Active=%d, TTA=%s",
                step,
                info["mse"],
                reward,
                len(status["active_agents"]),
                "adapting" if status.get("tta_adapting") else "stable",
            )

        obs = next_obs
        if done:
            break

    # Final report
    avg_mse = total_mse / step
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info("Total steps:       %d", step)
    logger.info("Average MSE:       %.6f", avg_mse)
    logger.info("Total reward:      %.4f", total_reward)
    logger.info("Final status:      %s", orchestrator.get_status())
    logger.info("MetaKernel status: %s", meta_kernel.get_swarm_status())
    logger.info("=" * 60)

    return {"avg_mse": avg_mse, "total_reward": total_reward, "steps": step}


def main() -> None:
    parser = argparse.ArgumentParser(description="SSM-Mamba Swarm Benchmark Runner")
    parser.add_argument("--pattern", choices=["sinusoidal", "degradation", "switching"],
                        default="degradation", help="Benchmark pattern")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    parser.add_argument("--obs-dim", type=int, default=32, help="Observation dimension")
    parser.add_argument("--d-model", type=int, default=64, help="MambaSSM d_model")
    args = parser.parse_args()

    config = SwarmConfig(
        observation_dim=args.obs_dim,
        action_dim=args.obs_dim,
    )
    config.mamba.d_model = args.d_model

    env_config = EnvConfig(
        observation_dim=args.obs_dim,
        sequence_length=args.seq_len,
        pattern=args.pattern,
    )

    run_benchmark(config, env_config)


if __name__ == "__main__":
    main()
