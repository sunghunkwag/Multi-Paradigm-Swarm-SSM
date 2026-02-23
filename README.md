# Multi-Paradigm Swarm SSM: Heterogeneous Multi-Agent Architecture

A heterogeneous swarm of five agents (Symbolic, JEPA, Liquid, SSM, SNN) coordinated via a Mamba-inspired SSM backbone (pure-PyTorch selective scan; optional mamba-ssm CUDA backend) with MAML meta-learning and Test-Time Adaptation (TTA). The system supports neural architecture search (NAS) through runtime capacity scaling and agent suppression/recovery managed by MetaKernelV2.

**Test suite**: 54 tests (25 unit/integration + 29 METR-HRS benchmark) covering MambaSSM forward/backward passes, SSMStabilityAgent NAS operations, orchestrator TTA, MetaKernel self-modification, end-to-end sequential prediction benchmarks, and the full evaluation harness. All 54 pass on CPU with the pure-PyTorch fallback.

```bash
pip install -r ssm_mamba_swarm/requirements.txt && pip install snntorch ncps
python -m pytest ssm_mamba_swarm/tests/ -v
python -m ssm_mamba_swarm.main --seq-len 100 --pattern switching
```

## METR-HRS Evaluation

The `ssm_mamba_swarm/eval/` package provides a METR-HRS compatible benchmarking harness with three reference baselines and a structured JSON report.

**HRS score** (per task): `max(0, 1 − MSE_predictor / MSE_zero)` — 1.0 = perfect, 0.0 = on par with always-predicting-zero.

| Task | Difficulty | Description |
|------|-----------|-------------|
| `sinusoidal_prediction` | easy | Multi-frequency sinusoidal signals |
| `degradation_prediction` | easy | Linear drift with noise (RUL-style) |
| `switching_prediction` | medium | Abrupt regime switching |
| `chaos_1d_lorenz` | hard | Lorenz system with parameter shifts |
| `adversarial_entropy` | hard | Logistic map + oscillatory + linear drift |

```python
from ssm_mamba_swarm.eval import EvalHarness, ZeroPredictor, OrchestratorPredictor

harness = EvalHarness(observation_dim=32, seeds=[100, 101, 102])
report  = harness.evaluate({
    "zero_predictor": ZeroPredictor(32, 32),
    "ssm_swarm":      OrchestratorPredictor(my_orchestrator),
})
print(report.to_json())   # METR-compatible JSON output
```
