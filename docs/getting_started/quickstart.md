# Quickstart

This guide trains a flow matching model on the 2D Poisson equation, end to end.

FlowPDE is a library, not a script runner: a training run is composed in Python, and
no configuration layer is required.

## 1. Generate Data

FlowPDE uses [Exponax](https://fkoehler.site/exponax/) spectral solvers to generate
PDE datasets directly:

```python
from flowpde.datasets import PoissonGenerator

generator = PoissonGenerator(
    num_spatial_dims=2,
    num_points=64,
    domain_extent=10.0,
)
train_ds = generator.generate(num_samples=1000, seed=42, problem="forward")
val_ds   = generator.generate(num_samples=200,  seed=43, problem="forward")
```

Every dataset emits `{'input': condition, 'target': solution}`. For **inverse
problems** (solution → source), pass `problem="inverse"`, which also enables
`obs_noise_std` and `obs_mask_fraction`.

!!! note "Masking adds a channel"
    When masking is on, the binary observation mask is appended as an extra condition
    channel. Build the model with `condition_channels` matching
    `sample['input'].shape[0]`.

Burgers and Darcy work the same way:

```python
from flowpde.datasets import BurgersGenerator, DarcyGenerator

burgers = BurgersGenerator(
    num_spatial_dims=1, num_points=160,
    dt=0.001, num_steps=50,
    diffusivity_min=0.0003, diffusivity_max=0.0003,   # a range, not a scalar
)
burgers_ds = burgers.generate(num_samples=500, seed=0)   # 'initial' -> 'final'

darcy = DarcyGenerator(num_spatial_dims=2, num_points=64)
darcy_ds = darcy.generate(
    num_samples=500, seed=0,
    problem="inverse",
    inverse_mode="coefficient",    # (u, f) -> kappa; a generate() argument
)
```

`inverse_mode` is an argument to `generate()`, not to the constructor. It takes
`'both'` (u → κ, f), `'coefficient'` ((u, f) → κ), or `'source'` ((u, κ) → f).

## 2. Normalize

Flow matching transports $\mathcal{N}(0, I)$ to the data, so raw PDE fields with
non-unit scale make the velocity regression badly conditioned. **Always normalize.**

```python
from flowpde.datasets import FieldNormalizer

normalizer = FieldNormalizer.from_dataset(train_ds)
train_ds.set_normalizer(normalizer)
val_ds.set_normalizer(normalizer)   # SAME instance — never refit on val/test
```

## 3. Define Model, Flow, and Objective

The flow is the *dynamics*; the objective is *how you train it*. They are separate
objects, and both objectives wrap the same flow.

```python
from flowpde import NeuralODEFlow, FlowMatchingObjective, UNet

model = UNet(
    spatial_dim=2,
    spatial_size=64,
    base_channels=64,
    solution_channels=1,
    condition_channels=1,
)

flow = NeuralODEFlow(
    model,
    target_key="target",       # the Exponax datasets use 'target'/'input',
    condition_key="input",     # not the 'u'/'f' defaults
)

objective = FlowMatchingObjective(
    flow,
    path="linear",           # x_t = (1-t)x_0 + t*x_1
    time_sampler="uniform",  # t ~ U(0,1)
    coupling="independent",
    source="gaussian",
)
```

Presets are available for common variants:

```python
from flowpde import create_flow_matching

objective = create_flow_matching(flow, variant="ot_cfm")
# 'standard' | 'rectified' | 'ot_cfm' | 'ot_cfm_coupled'
```

## 4. Train

```python
import torch
from torch.utils.data import DataLoader
from flowpde import Trainer
from flowpde.trainers import FlowEvaluator

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

evaluator = FlowEvaluator(
    objective,
    val_loader,
    n_steps=50,
    solver="euler",
    normalizer=normalizer,              # score in physical units
    target_fields=train_ds.target_fields,
)

optimizer = torch.optim.Adam(objective.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

trainer = Trainer(
    objective, optimizer, scheduler,
    device="cuda",
    ema_decay=0.999,                    # averaged weights are what you ship
    validator=evaluator,
    monitor="rel_l2",
    val_interval=5,
    checkpoint_extra={"normalizer_state": normalizer.state_dict()},
)

trainer.train(
    train_loader,
    epochs=100,
    print_stats_interval=10,
    save_dir="results/poisson_forward/",
    save_interval=25,
)
```

!!! warning "Training loss is not a model-selection signal"
    The flow-matching loss has a large irreducible floor — many $(x_0, x_1)$ pairs give
    the same $x_t$. Select on `FlowEvaluator` output, not `train_loss`.

## 5. Sample Solutions

```python
batch = next(iter(val_loader))
samples = flow.sample(
    condition=batch["input"],
    n_steps=50,
    solver="dopri5",
)                                       # flattened, shape (B, D)

samples = samples.view_as(batch["target"])

# Back to physical units before scoring
pred = normalizer.denormalize_channels(val_ds.target_fields, samples)
```

`sample()` returns a flattened `(B, D)` tensor — reshape it to the spatial layout
before denormalizing. If the target and condition dimensions differ, pass
`target_shape=` to `sample()`.

!!! tip
    `FlowEvaluator` already does the integrate → reshape → denormalize → score
    sequence for you; do this by hand only when you want the samples themselves.

## 6. Straighten with Reflow (optional)

Reflow generates `(z, ODE(z))` pairs with the current model and retrains on them, so
fewer solver steps suffice at inference.

```python
from flowpde.trainers import reflow

reflow(
    objective, train_loader,
    optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-4),
    num_iterations=2,
    epochs_per_iteration=50,
)
```

`reflow()` restores the original source distribution on exit, so sampling behaviour is
unchanged afterwards.

## Recording a Run

Configuration is specified in Python and *recorded* as YAML/JSON, not the other way
around. Every flow, objective, and component exposes `get_config()`:

```python
import json
with open("results/poisson_forward/config.json", "w") as fh:
    json.dump(objective.get_config(), fh, indent=2)
```

Save the resolved dataset and objective configuration with each run so reported
results can be reproduced. The project report documents the benchmark configurations
used for its results.

## Next Steps

- [Architecture Overview](../concepts/architecture.md) — how the pieces fit together
- [Flow Matching](../concepts/flow_matching.md) — the theory
- [API Reference](../api/index.md) — module documentation
