# Experiment 04: Darcy Objective Ablation

This experiment compares conditional flow matching with maximum-likelihood
training for the same continuous neural ODE flow on 2D Darcy.

```text
condition: [kappa, source]
target:    solution u
backbone:  same small UNet without attention
flow:      NeuralODEFlow
objectives:
  - flow matching
  - MLE with Hutchinson trace, 1 sample
  - MLE with Hutchinson trace, 4 samples
```

The default Tier 1 config uses 32x32 Darcy with 3000/200/200 train/val/test
splits. Batch size is 1 for both flow matching and MLE so the objective
comparison uses the same optimization granularity. All variants use the same
small UNet (`base_channels=8`, `max_channels=64`) and fixed-step RK4
integration. The UNet attention block is disabled because MLE needs
higher-order derivatives through the vector field for the trace estimator, and
attention kernels are not always compatible with that double-backward path.

Run all variants:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m experiments.exp04_objective_ablation_darcy.run --config experiments/configs/exp04_objective_ablation/darcy.yaml
```

Run one variant:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m experiments.exp04_objective_ablation_darcy.run --config experiments/configs/exp04_objective_ablation/darcy.yaml --variants mle_hutchinson_1
```

Smoke test:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m experiments.exp04_objective_ablation_darcy.run --config experiments/configs/exp04_objective_ablation/darcy.yaml --quick
```

After training, generate the tradeoff and inference figures:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m experiments.exp04_objective_ablation_darcy.plot --config experiments/configs/exp04_objective_ablation/darcy.yaml
```

Estimate MLE step time on the target machine before launching the full run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m experiments.exp04_objective_ablation_darcy.benchmark_mle --config experiments/configs/exp04_objective_ablation/darcy.yaml --grid 32 --batch-size 1 --variant mle_hutchinson_1 --device cuda
```
