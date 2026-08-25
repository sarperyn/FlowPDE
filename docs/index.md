# FlowPDE

**Flow-based generative models for forward and inverse PDE problems.**

---

FlowPDE is a PyTorch library that uses **flow matching** and **neural ODE flows** to
learn neural operators for PDEs. Instead of solving a PDE numerically at inference
time, it trains a flow-based generative model that produces solutions conditioned on
PDE parameters.

## Key Idea

Learn a velocity field $v_\theta$ that transports Gaussian noise to the solution
distribution, conditioned on the PDE input $f$, then generate by integrating:

\[
\frac{dx_t}{dt} = v_\theta(x_t, f, t), \quad x_0 \sim \mathcal{N}(0, I) \;\longrightarrow\; x_1 \approx u
\]

The library targets two problems:

1. **Forward problems** — show the learned operator solves the PDE, measured against a
   deterministic baseline and against the number of solver steps.
2. **Inverse problems and uncertainty quantification** — recover PDE inputs from noisy
   or partial observations, where the posterior is genuinely non-degenerate and a
   generative model earns its keep.

## Features

- :material-lightning-bolt: **Two objectives on one flow** — velocity regression (flow matching) or exact log-likelihood, both wrapping the same `NeuralODEFlow`
- :material-cog: **Composable, not subclassed** — paths, time samplers, couplings, and sources are pluggable pieces, so variants are configuration
- :material-cube-outline: **Multiple backbones** — MLP, UNet, ConvNet, and ResNet velocity fields
- :material-database: **Exponax integration** — spectral data generation for Poisson, Burgers, and Darcy flow via [Exponax](https://fkoehler.site/exponax/)
- :material-arrow-left-right: **Forward and inverse problems** — one `problem=` flag, with observation noise and masking for inverse setups
- :material-refresh: **Reflow** — iterative path straightening for fewer solver steps at inference

## Quick Example

```python
import torch
from torch.utils.data import DataLoader

from flowpde import NeuralODEFlow, FlowMatchingObjective, Trainer, UNet
from flowpde.datasets import PoissonGenerator, FieldNormalizer

# 1. Data
generator = PoissonGenerator(num_spatial_dims=2, num_points=64, domain_extent=10.0)
train_ds = generator.generate(num_samples=1000, seed=42, problem="forward")

normalizer = FieldNormalizer.from_dataset(train_ds)
train_ds.set_normalizer(normalizer)
loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 2. Model, flow, objective
model = UNet(spatial_dim=2, spatial_size=64, base_channels=64)
flow = NeuralODEFlow(model, target_key="target", condition_key="input")
objective = FlowMatchingObjective(flow, path="linear", time_sampler="uniform")

# 3. Train
optimizer = torch.optim.Adam(objective.parameters(), lr=1e-4)
trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.999)
trainer.train(loader, epochs=100, print_stats_interval=10,
              save_dir="results/poisson/", save_interval=25)

# 4. Sample
batch = next(iter(loader))
samples = flow.sample(condition=batch["input"], n_steps=50)   # (B, D), flattened
```

See the [Quickstart](getting_started/quickstart.md) for the full version, including
validation and evaluation in physical units.

## Installation

```bash
git clone https://github.com/sarperyn/FlowPDE.git
cd FlowPDE
uv venv && uv sync
```

See the [Installation Guide](getting_started/installation.md) for details.

## Citation

```bibtex
@software{flowpde,
  title  = {FlowPDE: Flow-based Generative Models for PDEs},
  author = {Yurtseven, Sarper},
  url    = {https://github.com/sarperyn/FlowPDE},
  year   = {2026},
}
```
