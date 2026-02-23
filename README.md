# SSM-Mamba Swarm: Multi-Agent Hybrid AI Architecture

A research-grade hybrid AI system that integrates **Multi-Agent Self-Modifying Swarms** with **Mamba State Space Models (SSM)** and **Meta-Learning** for sequential prediction under chaotic dynamics.

## Key Features

- **5-Agent Root-Purified Swarm**: Heterogeneous agent architectures operating on first-principles reasoning without reliance on hand-crafted heuristics.
- **Chaos 1D Lorenz Benchmark**: A heuristic-proof evaluation environment derived from chaotic dynamical systems, designed to stress-test predictive models under sensitive dependence on initial conditions.
- **Principled Implementations**:
  - **Recursive JEPA**: A 5-step recursive latent rollout world model with trajectory optimization.
  - **1D-Native Symbolic**: Program synthesis over 1D temporal primitives (Trend, Cycle, Shock).
  - **Mamba Selective Scan**: O(T·d) linear-time stability backbone based on selective state space modeling.
  - **Liquid (CfC) and SNN (LIF)**: Multi-layer neural dynamic backends utilizing Closed-form Continuous-time and Leaky Integrate-and-Fire neuron models, respectively.
- **Meta-Kernel V2**: A self-modifying architecture incorporating Neural Architecture Search (NAS) for adaptive structural optimization.
- **Online Adaptation**: Test-Time Adaptation (TTA) and MAML-based meta-learning to handle distributional shift at inference time.

## Project Structure

```
ssm_mamba_swarm/
├── agents/             # Heterogeneous AI agent implementations
├── core/               # MambaSSM, MAML, TTA, and orchestration modules
├── envs/               # Sequential prediction benchmark environments
├── tests/              # Unit and integration test suite
├── main.py             # Unified benchmark entry point
└── requirements.txt    # Dependency specification
```

## Setup and Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install backend libraries
pip install snntorch ncps denoising_diffusion_pytorch
```

## Verification

Run the integration test suite to validate system stability and functional correctness:

```bash
python -m pytest ssm_mamba_swarm/tests/test_hybrid_integration.py -v
```

## Benchmark Execution

Run the sequential prediction benchmark with the following command:

```bash
python main.py --pattern switching --seq-len 100
```

---
This project is a unified integration of the `heterogeneous-agent-swarm` and `SSM-MetaRL-TestCompute` codebases.
