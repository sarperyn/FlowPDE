# Experiment 03: Darcy Backbone Ablation

This experiment compares neural velocity backbones for conditional flow
matching on the 2D Darcy forward problem.

```text
condition: [kappa, source]
target:    solution u
objective: flow matching
backbones: ConvNet, ResNet, UNet, UNet without attention
```

The default split uses 3000 training samples, 200 validation samples, and
200 test samples. All variants share the same generated data, normalization,
optimizer, flow objective, ODE sampler, and concat conditioning. The ablation
therefore isolates the neural backbone.

Run all variants:

```bash
uv run python -m experiments.exp03_backbone_ablation_darcy.run --config experiments/configs/exp03_backbone_ablation/darcy.yaml
```

Run one variant only:

```bash
uv run python -m experiments.exp03_backbone_ablation_darcy.run --config experiments/configs/exp03_backbone_ablation/darcy.yaml --variants unet
```

For a quick smoke test:

```bash
uv run python -m experiments.exp03_backbone_ablation_darcy.run --config experiments/configs/exp03_backbone_ablation/darcy.yaml --quick
```

After training, build the comparison figure, inference-example figure, and
diagnostic CSVs:

```bash
uv run python -m experiments.exp03_backbone_ablation_darcy.plot --config experiments/configs/exp03_backbone_ablation/darcy.yaml
```
