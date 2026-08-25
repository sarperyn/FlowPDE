# Experiment 02: Burgers Backbone Ablation

This experiment compares neural velocity backbones for conditional flow
matching on the 1D Burgers forward problem.

```text
condition: initial state u(x, 0)
target:    final state u(x, T)
objective: flow matching
backbones: ConvNet, ResNet, UNet, UNet without attention
```

The default split uses 3000 training samples, 200 validation samples, and
200 test samples. All variants share the same generated data, normalization,
optimizer, flow objective, ODE sampler, and concat conditioning. The ablation
therefore isolates the neural backbone.

Run all variants:

```bash
uv run python -m experiments.exp02_backbone_ablation_burgers.run --config experiments/configs/exp02_backbone_ablation/burgers.yaml
```

Run one variant only:

```bash
uv run python -m experiments.exp02_backbone_ablation_burgers.run --config experiments/configs/exp02_backbone_ablation/burgers.yaml --variants unet
```

For a quick smoke test:

```bash
uv run python -m experiments.exp02_backbone_ablation_burgers.run --config experiments/configs/exp02_backbone_ablation/burgers.yaml --quick
```

After training, build the paper-style figure and Burgers diagnostics:

```bash
uv run python -m experiments.exp02_backbone_ablation_burgers.plot --config experiments/configs/exp02_backbone_ablation/burgers.yaml
```
