"""METR-HRS evaluation package for the Multi-Paradigm Swarm SSM."""
from .baselines import ZeroPredictor, IdentityPredictor, LinearPredictor
from .eval_harness import (
    EvalHarness,
    METRReport,
    TaskSummary,
    TaskResult,
    OrchestratorPredictor,
    METR_TASKS,
    EVAL_SEEDS,
)

__all__ = [
    "ZeroPredictor",
    "IdentityPredictor",
    "LinearPredictor",
    "EvalHarness",
    "METRReport",
    "TaskSummary",
    "TaskResult",
    "OrchestratorPredictor",
    "METR_TASKS",
    "EVAL_SEEDS",
]
