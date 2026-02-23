"""METR-HRS Evaluation Harness.

Runs a standardized benchmark suite across multiple fixed seeds and produces
a METR-HRS compatible JSON report.

HRS Score (per task):
    hrs = max(0, 1 - mse_predictor / mse_zero_predictor)

    1.0 → perfect prediction
    0.0 → on par with always-predicting-zero
   <0.0 → worse than zero (clamped to 0)

Overall HRS score = mean of per-task scores.

Usage:
    from ssm_mamba_swarm.eval import EvalHarness, ZeroPredictor, IdentityPredictor

    harness = EvalHarness(observation_dim=32, seeds=[100, 101, 102])
    report  = harness.evaluate({"zero_predictor": ZeroPredictor(32, 32),
                                "identity":       IdentityPredictor(32, 32),
                                "my_agent":       MyAgent(32, 32)})
    print(report.to_json())
"""
import json
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from ..agents.base_agent import BaseAgent, AgentProposal
from ..envs.seq_prediction_env import SequentialPredictionEnv, EnvConfig
from ..envs.chaos_1d_env import Chaos1DEnv
from ..envs.adversarial_env import AdversarialEntropyEnv


# ---------------------------------------------------------------------------
# Fixed evaluation seeds — never used during training
# ---------------------------------------------------------------------------
EVAL_SEEDS: List[int] = [100, 101, 102, 103, 104]

# ---------------------------------------------------------------------------
# METR-HRS task registry
# ---------------------------------------------------------------------------
METR_TASKS: List[Dict] = [
    {
        "name": "sinusoidal_prediction",
        "env_cls": SequentialPredictionEnv,
        "pattern": "sinusoidal",
        "seq_len": 200,
        "difficulty": "easy",
    },
    {
        "name": "degradation_prediction",
        "env_cls": SequentialPredictionEnv,
        "pattern": "degradation",
        "seq_len": 200,
        "difficulty": "easy",
    },
    {
        "name": "switching_prediction",
        "env_cls": SequentialPredictionEnv,
        "pattern": "switching",
        "seq_len": 200,
        "difficulty": "medium",
    },
    {
        "name": "chaos_1d_lorenz",
        "env_cls": Chaos1DEnv,
        "pattern": "sinusoidal",   # unused by Chaos1DEnv.step()
        "seq_len": 200,
        "difficulty": "hard",
    },
    {
        "name": "adversarial_entropy",
        "env_cls": AdversarialEntropyEnv,
        "pattern": "sinusoidal",   # unused by AdversarialEntropyEnv.step()
        "seq_len": 200,
        "difficulty": "hard",
    },
]


# ---------------------------------------------------------------------------
# Orchestrator wrapper
# ---------------------------------------------------------------------------
class OrchestratorPredictor(BaseAgent):
    """Adapts the swarm Orchestrator to the BaseAgent interface.

    This lets the Orchestrator participate in the eval harness alongside
    simple baseline agents without changing the Orchestrator itself.
    """

    def __init__(self, orchestrator, name: str = "ssm_swarm"):
        first_agent = next(iter(orchestrator.agents.values()))
        super().__init__(name, first_agent.observation_dim, first_agent.action_dim)
        self.orch = orchestrator

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        action, _ = self.orch.select_action(observation)
        return AgentProposal(
            agent_name=self.name,
            action=action,
            confidence=1.0,
        )

    def reset_state(self) -> None:
        for agent in self.orch.agents.values():
            agent.reset_state()

    def get_capacity_metrics(self) -> Dict:
        return {
            "model_type": "OrchestratorSwarm",
            "agent_name": self.name,
            "active_agents": [n for n, a in self.orch.agents.items()
                              if not a.is_suppressed],
        }


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class TaskResult:
    """Single (task, predictor, seed) run."""
    task_name: str
    predictor_name: str
    seed: int
    mse_per_step: List[float]
    total_reward: float

    @property
    def mean_mse(self) -> float:
        return float(np.mean(self.mse_per_step)) if self.mse_per_step else float("inf")

    @property
    def median_mse(self) -> float:
        return float(np.median(self.mse_per_step)) if self.mse_per_step else float("inf")

    @property
    def valid(self) -> bool:
        return len(self.mse_per_step) > 0


@dataclass
class TaskSummary:
    """Aggregated result across seeds for one (task, predictor) pair."""
    task_name: str
    predictor_name: str
    difficulty: str
    mean_mse: float
    std_mse: float
    median_mse: float
    mean_reward: float
    n_valid_runs: int
    n_total_runs: int


@dataclass
class METRReport:
    """METR-HRS compatible evaluation report.

    hrs_scores layout:
        {predictor_name: {task_name: score, ..., "overall": float}}
    """
    system_name: str
    eval_date: str
    observation_dim: int
    n_seeds: int
    task_summaries: List[TaskSummary]
    hrs_scores: Dict[str, Dict[str, float]]

    def overall_hrs(self, predictor_name: str) -> float:
        return self.hrs_scores.get(predictor_name, {}).get("overall", 0.0)

    def to_json(self) -> str:
        d = {
            "system_name": self.system_name,
            "eval_date": self.eval_date,
            "observation_dim": self.observation_dim,
            "n_seeds": self.n_seeds,
            "task_summaries": [vars(s) for s in self.task_summaries],
            "hrs_scores": self.hrs_scores,
        }
        return json.dumps(d, indent=2)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------
class EvalHarness:
    """METR-HRS evaluation harness.

    Args:
        observation_dim: Dimension of observations (and actions).
        seeds:           Evaluation seeds (must not overlap with training seeds).
        tasks:           List of task dicts; defaults to METR_TASKS.
    """

    def __init__(
        self,
        observation_dim: int = 32,
        seeds: Optional[List[int]] = None,
        tasks: Optional[List[Dict]] = None,
    ):
        self.observation_dim = observation_dim
        self.seeds = seeds if seeds is not None else EVAL_SEEDS
        self.tasks = tasks if tasks is not None else METR_TASKS

    # ------------------------------------------------------------------
    def _make_env(self, task: Dict, seed: int):
        cfg = EnvConfig(
            observation_dim=self.observation_dim,
            sequence_length=task["seq_len"],
            pattern=task.get("pattern", "sinusoidal"),
            seed=seed,
        )
        return task["env_cls"](cfg)

    def _run_episode(
        self,
        env,
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        seed: int,
    ) -> Tuple[List[float], float]:
        """Run one episode; returns (mse_per_step, total_reward)."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        obs = env.reset()
        mse_list: List[float] = []
        total_reward = 0.0
        done = False

        while not done:
            action = predict_fn(obs)
            # Align action dim with env observation dim if needed
            env_dim = obs.shape[0]
            if action.shape[0] != env_dim:
                aligned = torch.zeros(env_dim)
                n = min(env_dim, action.shape[0])
                aligned[:n] = action[:n]
                action = aligned

            next_obs, reward, done, info = env.step(action)

            mse = info.get("mse", float("nan"))
            if math.isnan(mse) or math.isinf(mse):
                break   # NaN collapse (e.g. HighDimChaosEnv event horizon)

            mse_list.append(float(mse))
            total_reward += float(reward)
            obs = next_obs

        return mse_list, total_reward

    # ------------------------------------------------------------------
    def evaluate(
        self,
        predictors: Dict[str, BaseAgent],
        system_name: str = "SSMSwarm",
    ) -> METRReport:
        """Run the full METR-HRS suite and return a structured report.

        Args:
            predictors:  {name: BaseAgent}.  Include "zero_predictor" for
                         meaningful HRS scores.
            system_name: Label written into the report header.
        """
        # Difficulty lookup
        difficulty_map = {t["name"]: t.get("difficulty", "medium") for t in self.tasks}

        # Collect raw results
        raw: Dict[Tuple[str, str], List[TaskResult]] = {}
        for task in self.tasks:
            task_name = task["name"]
            for pred_name, predictor in predictors.items():
                key = (task_name, pred_name)
                raw[key] = []
                for seed in self.seeds:
                    env = self._make_env(task, seed)
                    predictor.reset_state()

                    def predict_fn(obs, p=predictor):
                        proposal = p.propose(obs)
                        return proposal.action

                    mse_list, total_reward = self._run_episode(env, predict_fn, seed)
                    raw[key].append(TaskResult(
                        task_name=task_name,
                        predictor_name=pred_name,
                        seed=seed,
                        mse_per_step=mse_list,
                        total_reward=total_reward,
                    ))

        # Aggregate into TaskSummary
        summaries: List[TaskSummary] = []
        for (task_name, pred_name), results in raw.items():
            valid = [r for r in results if r.valid]
            mse_means = [r.mean_mse for r in valid]
            rewards = [r.total_reward for r in valid]
            summaries.append(TaskSummary(
                task_name=task_name,
                predictor_name=pred_name,
                difficulty=difficulty_map.get(task_name, "medium"),
                mean_mse=float(np.mean(mse_means)) if mse_means else float("inf"),
                std_mse=float(np.std(mse_means)) if len(mse_means) > 1 else 0.0,
                median_mse=float(np.median(mse_means)) if mse_means else float("inf"),
                mean_reward=float(np.mean(rewards)) if rewards else float("-inf"),
                n_valid_runs=len(valid),
                n_total_runs=len(results),
            ))

        # Build summary lookup: (task_name, pred_name) -> TaskSummary
        summap: Dict[Tuple[str, str], TaskSummary] = {
            (s.task_name, s.predictor_name): s for s in summaries
        }

        # Compute HRS scores relative to zero_predictor
        hrs_scores: Dict[str, Dict[str, float]] = {}
        for pred_name in predictors:
            task_scores: Dict[str, float] = {}
            for task in self.tasks:
                tn = task["name"]
                zero = summap.get((tn, "zero_predictor"))
                pred = summap.get((tn, pred_name))
                if zero and pred and zero.mean_mse > 0 and not math.isinf(pred.mean_mse):
                    score = max(0.0, 1.0 - pred.mean_mse / zero.mean_mse)
                    task_scores[tn] = round(score, 4)
                else:
                    task_scores[tn] = 0.0
            task_scores["overall"] = round(float(np.mean(list(task_scores.values()))), 4)
            hrs_scores[pred_name] = task_scores

        return METRReport(
            system_name=system_name,
            eval_date=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            observation_dim=self.observation_dim,
            n_seeds=len(self.seeds),
            task_summaries=summaries,
            hrs_scores=hrs_scores,
        )
