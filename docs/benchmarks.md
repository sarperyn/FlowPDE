# Benchmarks and Pretrained Models

FlowPDE ships a benchmark suite: four forward and inverse PDE tasks, each with
fixed data, splits, backbone, training budget and evaluation protocol. The
same scripts train the models, score them, and export them as pretrained files
you can load in one line.

## Results

Relative $L^2$ and $H^1$ errors are measured in physical units on the held-out
test split, scoring the EMA weights selected on validation error. One sample is
drawn per test input with the listed ODE solver.

<!-- benchmark-table:start -->
_No benchmark runs recorded yet._
<!-- benchmark-table:end -->

This table is generated from recorded runs by `python -m benchmarks.report`;
it is never edited by hand.

## Tasks

| Task | PDE / direction | Conditioning input | Target |
|---|---|---|---|
| `burgers_forward` | 1D viscous Burgers, $\nu = 0.01$ | initial state $u(x, 0)$ | final state $u(x, T)$ |
| `poisson_forward` | 2D Poisson, eight-mode sources | source $f$ | solution $u$ |
| `darcy_forward` | 2D Darcy, log-normal $\kappa$ | $(\kappa, f)$ | pressure $u$ |
| `darcy_inverse` | 2D Darcy | 15 % observed, noisy $u$; known $f$; mask | coefficient $\kappa$ |

All tasks use 3000 / 200 / 200 train / validation / test samples with fixed
seeds, a ConvNet backbone with concatenation conditioning, linear-path flow
matching, AdamW with a cosine schedule, gradient clipping and EMA (decay
0.999). The settings follow the course-project experiments in the
[project report](https://github.com/sarperyn/FlowPDE/blob/main/report.pdf);
the full definitions are in
[`benchmarks/tasks.py`](https://github.com/sarperyn/FlowPDE/blob/main/benchmarks/tasks.py).

## Using a pretrained model

```python
from torch.utils.data import DataLoader
from flowpde.pretrained import list_pretrained, load_pretrained

print(list_pretrained())                          # published names

model = load_pretrained("darcy_forward_convnet")  # downloads once, checks sha256
print(model.metrics["rel_l2"])                    # test error it was published with

# Fresh data from the training distribution, normalized for this model
# (needs `pip install "flowpde[data]"`).
test = model.make_dataset(num_samples=16, seed=2024)
batch = next(iter(DataLoader(test, batch_size=8)))

u = model.sample(batch["input"], n_steps=50)      # (8, 1, 64, 64), physical units
```

`load_pretrained` also accepts a local file or a URL, so models you train
yourself load the same way:

```python
model = load_pretrained("benchmarks/runs/quick/burgers_forward/burgers_forward.pt", device="mps")
```

Downloads are cached in `~/.cache/flowpde/pretrained` (override with the
`FLOWPDE_CACHE` environment variable). Files load with
`torch.load(weights_only=True)`, so a pretrained file cannot run code.

## Reproducing the table

```bash
python -m benchmarks.run --all --scale full --device cuda
python -m benchmarks.report benchmarks/runs/full --write docs/benchmarks.md
```

`--scale quick` runs a reduced version that fits a laptop (CPU or Apple MPS),
and `--device auto` picks CUDA, then MPS, then CPU. See
[`benchmarks/README.md`](https://github.com/sarperyn/FlowPDE/blob/main/benchmarks/README.md)
for the scales, the SLURM template, and how results are published.

## Saving your own models

`save_pretrained` writes the same self-describing format from any trained
objective:

```python
from flowpde.pretrained import data_spec, save_pretrained

save_pretrained(
    "my_model.pt", objective, recipe,
    target_shape=train_ds[0]["target"].shape,
    normalizer=normalizer,
    data=data_spec(generator, problem="forward", dataset=train_ds),
    metrics={"rel_l2": 0.031},
)
```

The `recipe` is a plain dict naming the backbone, conditioner and objective
settings; `build_objective(recipe)` rebuilds the model from it. See the
[API reference](api/pretrained.md).
