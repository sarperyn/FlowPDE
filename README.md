# FlowPDE

**Flow-based generative models for forward and inverse PDE problems.**

FlowPDE is a PyTorch library that uses flow matching and neural ODE flows to learn
neural operators for PDEs. Instead of solving a PDE numerically at inference time, it
trains a flow-based model that generates solutions conditioned on PDE parameters.

📖 **[Documentation](https://sarperyn.github.io/FlowPDE/)**

## Setup (uv)

```bash
uv python install 3.11
uv python pin 3.11
uv venv
uv sync
source .venv/bin/activate
```

## Quick Example

```python
import torch
from torch.utils.data import DataLoader

from flowpde import NeuralODEFlow, FlowMatchingObjective, Trainer, UNet
from flowpde.datasets import PoissonGenerator, FieldNormalizer

# Data
generator = PoissonGenerator(num_spatial_dims=2, num_points=64, domain_extent=10.0)
train_ds = generator.generate(num_samples=1000, seed=42, problem="forward")

normalizer = FieldNormalizer.from_dataset(train_ds)
train_ds.set_normalizer(normalizer)
loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# The flow is the dynamics; the objective is how you train it.
model = UNet(spatial_dim=2, spatial_size=64, base_channels=64)
flow = NeuralODEFlow(model, target_key="target", condition_key="input")
objective = FlowMatchingObjective(flow, path="linear", time_sampler="uniform")

# Train
optimizer = torch.optim.Adam(objective.parameters(), lr=1e-4)
trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.999)
trainer.train(loader, epochs=100, print_stats_interval=10,
              save_dir="results/poisson/", save_interval=25)
```

See the [Quickstart](https://sarperyn.github.io/FlowPDE/getting_started/quickstart/)
for the full version with validation and evaluation in physical units.

## Common tasks

```bash
# Tests
uv run -m pytest                    # full suite
uv run -m pytest -m "not slow"      # skip the Exponax integration tests

# Docs
pip install -e ".[docs]"
mkdocs serve                        # preview at http://127.0.0.1:8000
mkdocs build --strict               # static site into site/
```

## License

See [LICENSE](LICENSE).
