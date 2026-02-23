# SSM-Mamba Swarm: Multi-Agent Hybrid AI Architecture

A research-grade hybrid AI system that combines **Multi-Agent Self-Modifying Swarms** with **Mamba State Space Models (SSM)** and **Meta-Learning**.

## 🚀 Key Features

- **5-Agent Root-Purified Swarm**: Heterogeneous architectures operating on first principles without heuristics.
- **Chaos 1D Lorenz Benchmark**: Heuristic-proof evaluation environment based on chaotic dynamical systems.
- **Principled Implementations**:
  - **Recursive JEPA**: 5-step recursive latent rollout world model with trajectory optimization.
  - **1D-Native Symbolic**: Authentic program synthesis over 1D primitives (Trend, Cycle, Shock).
  - **Mamba Selective Scan**: O(T·d) linear-time stability backbone.
  - **Liquid (CfC) & SNN (LIF)**: Official multi-layer neural dynamic backends.
- **Meta-Kernel V2**: Self-modifying architecture with Neural Architecture Search (NAS).
- **Online Adaptation**: Test-Time Adaptation (TTA) and MAML-based meta-learning for distributional shift.

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
pip install -r requirements.txt

# Install official backend libraries
pip install snntorch ncps denoising_diffusion_pytorch
```

## 🧪 Verification

Run the full suite of integration tests to verify system stability and functional parity:

```bash
python -m pytest ssm_mamba_swarm/tests/test_hybrid_integration.py -v
```

## 📊 Benchmark

Execute the sequential prediction benchmark:

```bash
python main.py --pattern switching --seq-len 100
```

---
Developed as a unified hybrid of `heterogeneous-agent-swarm` and `SSM-MetaRL-TestCompute`.
