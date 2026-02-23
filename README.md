# Multi-Paradigm Swarm SSM: Heterogeneous Multi-Agent Architecture

A heterogeneous swarm of five agents (Symbolic, JEPA, Liquid, SSM, SNN) coordinated via a Mamba-inspired SSM backbone (pure-PyTorch selective scan; optional mamba-ssm CUDA backend) with MAML meta-learning and Test-Time Adaptation (TTA). The system supports neural architecture search (NAS) through runtime capacity scaling and agent suppression/recovery managed by MetaKernelV2.

**Test suite**: 54 tests (25 unit/integration + 29 METR-HRS benchmark) covering MambaSSM forward/backward passes, SSMStabilityAgent NAS operations, orchestrator TTA, MetaKernel self-modification, end-to-end sequential prediction benchmarks, and the full evaluation harness. All 54 pass on CPU with the pure-PyTorch fallback.

```bash
pip install -r ssm_mamba_swarm/requirements.txt && pip install snntorch ncps
python -m pytest ssm_mamba_swarm/tests/ -v
python -m ssm_mamba_swarm.main --seq-len 100 --pattern switching
```

## METR-HRS Evaluation

`ssm_mamba_swarm/eval/` — multi-seed harness over 5 fixed tasks (easy→hard) with `ZeroPredictor` / `IdentityPredictor` / `LinearPredictor` baselines. Outputs a JSON report with per-task HRS scores: `max(0, 1 − MSE / MSE_zero)`.

```python
from ssm_mamba_swarm.eval import EvalHarness, ZeroPredictor, OrchestratorPredictor

harness = EvalHarness(observation_dim=32, seeds=[100, 101, 102])
report  = harness.evaluate({"zero_predictor": ZeroPredictor(32, 32),
                             "ssm_swarm": OrchestratorPredictor(my_orchestrator)})
print(report.to_json())
```
