# Multi-Paradigm Swarm SSM: Heterogeneous Multi-Agent Architecture

A heterogeneous swarm of five agents (Symbolic, JEPA, Liquid, SSM, SNN) coordinated via a Mamba-inspired SSM backbone (pure-PyTorch selective scan; optional mamba-ssm CUDA backend) with MAML meta-learning and Test-Time Adaptation (TTA). The system supports neural architecture search (NAS) through runtime capacity scaling and agent suppression/recovery managed by MetaKernelV2.

**Test suite**: 25 unit and integration tests covering MambaSSM forward/backward passes, SSMStabilityAgent NAS operations, orchestrator TTA, MetaKernel self-modification, and end-to-end sequential prediction benchmarks. All 25 tests pass on CPU with the pure-PyTorch fallback.

```bash
pip install -r ssm_mamba_swarm/requirements.txt && pip install snntorch ncps
python -m pytest ssm_mamba_swarm/tests/ -v
python -m ssm_mamba_swarm.main --seq-len 100 --pattern switching
```
