# FlowPDE Experiments

This folder contains runnable experiments built on top of the local `flowpde`
library.

## Experiment 01: Conditioning Ablation

The first experiment studies whether conditioning helps a flow-matching UNet
solve 2D Darcy forward problems:

```text
condition: [kappa, source]
target:    solution
objective: flow matching
backbone:  UNet without attention
```

The first two variants are:

- `null`: ignore the condition with `NullConditioner`
- `concat`: concatenate `[x_t, condition]` with `ConcatConditioner`

Run both variants:

```bash
uv run python -m experiments.exp01_conditioning_ablation.run --config experiments/configs/exp01_conditioning_ablation/darcy.yaml
```

Run one variant only:

```bash
uv run python -m experiments.exp01_conditioning_ablation.run --config experiments/configs/exp01_conditioning_ablation/darcy.yaml --variants concat
```

For a quick smoke test:

```bash
uv run python -m experiments.exp01_conditioning_ablation.run --config experiments/configs/exp01_conditioning_ablation/darcy.yaml --quick
```

## Experiment 02: Burgers Backbone Ablation

The second experiment compares neural velocity backbones for conditional flow
matching on the 1D Burgers forward problem:

```text
condition: initial state u(x, 0)
target:    final state u(x, T)
objective: flow matching
backbones: ConvNet, ResNet, UNet, UNet without attention
```

Run all variants:

```bash
uv run python -m experiments.exp02_backbone_ablation_burgers.run --config experiments/configs/exp02_backbone_ablation/burgers.yaml
```

Generate the comparison figure after training:

```bash
uv run python -m experiments.exp02_backbone_ablation_burgers.plot --config experiments/configs/exp02_backbone_ablation/burgers.yaml
```

## Experiment 03: Darcy Backbone Ablation

The third experiment compares the same neural velocity backbones on the 2D
Darcy forward problem:

```text
condition: [kappa, source]
target:    solution u
objective: flow matching
backbones: ConvNet, ResNet, UNet, UNet without attention
```

Run all variants:

```bash
uv run python -m experiments.exp03_backbone_ablation_darcy.run --config experiments/configs/exp03_backbone_ablation/darcy.yaml
```

Generate the comparison and inference-example figures after training:

```bash
uv run python -m experiments.exp03_backbone_ablation_darcy.plot --config experiments/configs/exp03_backbone_ablation/darcy.yaml
```

## Experiment 04: Darcy Objective Ablation

The fourth experiment compares conditional flow matching against
maximum-likelihood training for the same conditional neural ODE flow:

```text
condition: [kappa, source]
target:    solution u
backbone:  same small UNet without attention
objectives: flow matching, MLE with Hutchinson trace
```

Run all Tier 1 variants:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m experiments.exp04_objective_ablation_darcy.run --config experiments/configs/exp04_objective_ablation/darcy.yaml
```

Generate the tradeoff and inference-example figures:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m experiments.exp04_objective_ablation_darcy.plot --config experiments/configs/exp04_objective_ablation/darcy.yaml
```
