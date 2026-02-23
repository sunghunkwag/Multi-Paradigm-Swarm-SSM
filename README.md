# chaos-prediction-envs

Benchmark envs for chaotic temporal prediction.

## Environments

| Env | Description |
|-----|-------------|
| `SequentialPredictionEnv` | Base class: sinusoidal / degradation / switching patterns |
| `Chaos1DEnv` | 1D Lorenz with rho regime switching (14→28→45) |
| `HighDimChaosEnv` | Coupled Lorenz/Rössler grid, Milstein SDE |
| `AdversarialEntropyEnv` | High-entropy adversarial dynamics |

## Install

```bash
pip install -r requirements.txt
```

## Status

Experimental. Tests may fail by design — environments are built to break heuristics.

## Requirements

- Python 3.9+
- torch, numpy, pytest
