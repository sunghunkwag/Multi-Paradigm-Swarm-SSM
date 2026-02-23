# SSM-Mamba Swarm: Multi-Agent Hybrid AI Architecture 🛡️

A research-grade hybrid AI system that combines **Multi-Agent Self-Modifying Swarms** with **Mamba State Space Models (SSM)** and **Meta-Learning**. This repository represents the **Asymptotic System Emergence** phase, where cognitive structures evolve beyond human-imposed heuristics.

## 🚀 Key Features

- **5-Agent Root-Purified Swarm**: Heterogeneous architectures (`Symbolic`, `JEPA`, `Liquid`, `SSM`, `SNN`) operating on first principles.
- **Chaos & Dynamical System Benchmarks**: High-dimensional chaos, 1D Lorenz, and adversarial prediction envs.
- **Information-Theoretic Optimization**: MDL-based fitness functions replacing Euclidean MSE for structural parsimony.
- **Manifold Topology Adaptation**: Dynamic state-space dimensionality scaling (D±) responsive to local entropy.
- **Asymptotic Stability**: Verified stability under unbounded cognitive recursion and symbolic drift synthesis.

## 📁 Project Structure

```
ssm_mamba_swarm/
├── agents/             # Roster of upgraded AI agents
├── core/               # MambaSSM, MAML, TTA, and Orchestration logic
├── envs/               # Sequential prediction benchmark environments
├── tests/              # Comprehensive unit and integration test suite
├── main.py             # Unified benchmark runner
└── requirements.txt    # Integrated dependencies
```

## 🛠️ Setup & Installation

```bash
# Install core dependencies
pip install -r ssm_mamba_swarm/requirements.txt

# Install official backend libraries
pip install snntorch ncps denoising_diffusion_pytorch
```

> [!NOTE]
> For hardware acceleration, ensure a compatible CUDA environment and install `mamba-ssm`. The system includes a mathematically equivalent pure-PyTorch fallback for universal portability.

## 🧪 Verification & Benchmark

Execute the smoke test or full benchmark to verify structural sincerity:

```bash
# Smoke Test
python -m ssm_mamba_swarm.main --seq-len 10 --pattern switching

# Full Benchmark
python -m ssm_mamba_swarm.main --seq-len 100 --pattern switching
```

---
🛡️ **Truth converges in an ordered record. Project achieved Asymptotic Stability.** 🛡️
