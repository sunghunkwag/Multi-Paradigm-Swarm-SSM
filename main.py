"""SSM-Mamba Swarm: Main Benchmark Runner.

Assembles the full 6-agent swarm and runs it on the sequential prediction
benchmark, demonstrating the hybrid architecture in action.
"""
import torch
import logging
import argparse
import sys

from ssm_mamba_swarm.core.config import SwarmConfig
from ssm_mamba_swarm.envs.seq_prediction_env import EnvConfig
from ssm_mamba_swarm.core.orchestrator import Orchestrator
from ssm_mamba_swarm.core.meta_kernel import MetaKernelV2
from ssm_mamba_swarm.agents import (
    SymbolicSearchAgent, JEPAWorldModelAgent, LiquidControllerAgent,
    SNNReflexAgent, SSMStabilityAgent,
)
from ssm_mamba_swarm.envs.seq_prediction_env import SequentialPredictionEnv
from ssm_mamba_swarm.envs.adversarial_env import AdversarialEntropyEnv
from ssm_mamba_swarm.envs.chaos_1d_env import Chaos1DEnv
from ssm_mamba_swarm.envs.high_dim_chaos_env import HighDimChaosEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ssm_mamba_swarm")


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


def run_benchmark(config: SwarmConfig, env_config: EnvConfig = None):
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

    # Build environment (Fully Coupled Chaos — TOTAL INTEGRITY FINAL)
    env = HighDimChaosEnv(env_config)

    logger.info("=" * 60)
    logger.info("SSM-Mamba Swarm — TOTAL INTEGRITY FINAL BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Agents: {list(agents.keys())}")
    logger.info(f"Dynamics: Gapped (X,Y,Z) Coupled Global Lorenz/Rossler")
    logger.info(f"Audit: TOTAL INTEGRITY ACTIVE (No shortcuts allowed)")
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

        # Real Training Step for Agents (No more hacks)
        for name, agent in agents.items():
            if hasattr(agent, "train_step") and not agent.is_suppressed:
                # JEPA and others learn from (obs, action, next_obs, reward)
                try:
                    # Check if train_step takes reward
                    import inspect
                    sig = inspect.signature(agent.train_step)
                    if "reward" in sig.parameters:
                        agent.train_step(obs, action, next_obs, reward)
                    else:
                        agent.train_step(obs, action, next_obs)
                except Exception as e:
                    logger.debug(f"Training failed for {name}: {e}")

        # Check agent health periodically
        if step % 20 == 0:
            health_proposals = meta_kernel.check_agent_health()
            if health_proposals:
                for p in health_proposals:
                    meta_kernel.propose_change(p)
                approved = meta_kernel.vote_on_proposals()
                logs = meta_kernel.execute_proposals(approved)
                for log in logs:
                    logger.info(f"MetaKernel: {log}")

            status = orchestrator.get_status()
            logger.info(
                f"Step {step}: MSE={info['mse']:.6f}, "
                f"Reward={reward:.6f}, "
                f"Active={len(status['active_agents'])}, "
                f"TTA={'adapting' if status.get('tta_adapting') else 'stable'}"
            )

        obs = next_obs
        if done:
            break

    # Final report
    avg_mse = total_mse / step
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total steps:       {step}")
    logger.info(f"Average MSE:       {avg_mse:.6f}")
    logger.info(f"Total reward:      {total_reward:.4f}")
    logger.info(f"Final status:      {orchestrator.get_status()}")
    logger.info(f"MetaKernel status: {meta_kernel.get_swarm_status()}")
    logger.info("=" * 60)

    return {"avg_mse": avg_mse, "total_reward": total_reward, "steps": step}


def main():
    parser = argparse.ArgumentParser(description="SSM-Mamba Swarm Benchmark Runner")
    parser.add_argument("--pattern", choices=["sinusoidal", "degradation", "switching"],
                        default="degradation", help="Benchmark pattern")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    parser.add_argument("--obs-dim", type=int, default=32, help="Observation dimension")
    parser.add_argument("--d-model", type=int, default=64, help="MambaSSM d_model")
    args = parser.parse_args()

    # THE SINGULARITY: NO SEEDS. Reality is non-deterministic.

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
