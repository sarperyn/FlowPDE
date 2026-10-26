# FlowPDE benchmarks

Scripts that train the benchmark tasks, score them on a held-out test split,
and export pretrained models for `flowpde.pretrained.load_pretrained`. The
results table lives in [`docs/benchmarks.md`](../docs/benchmarks.md).

| File | What it does |
|---|---|
| [`tasks.py`](tasks.py) | Task definitions: data, splits, backbone, optimiser, evaluation, at three scales |
| [`run.py`](run.py) | Train + evaluate + export one or more tasks |
| [`report.py`](report.py) | Build the Markdown table from recorded `metrics.json` files |
| [`publish.py`](publish.py) | Register exported models and print the GitHub release upload command |
| [`slurm.sh`](slurm.sh) | SLURM array-job template for the full-scale runs |

Run everything from the repository root with the `data` extra installed
(`uv sync`, or `pip install -e ".[data]"`).

## Tasks and scales

```bash
python -m benchmarks.run --list
```

| Task | PDE / direction | Report |
|---|---|---|
| `burgers_forward` | 1D Burgers, $u(x,0) \to u(x,T)$ | E2 |
| `poisson_forward` | 2D Poisson, eight-mode source $f \to u$ | E10 |
| `darcy_forward` | 2D Darcy, $(\kappa, f) \to u$ | E3 |
| `darcy_inverse` | 2D Darcy, 15 % observed noisy $u$ and $f \to \kappa$ | E6 |

`--scale` picks the budget; the task definition is the same at every scale.

| Scale | Data | Training | Use |
|---|---|---|---|
| `full` | report settings: 64 / 64², 3000 / 200 / 200 | 150–500 epochs | GPU; the published numbers |
| `quick` | 2D at 32², 1000 / 64 / 200 | 60–100 epochs | laptop CPU or Apple MPS; indicative only |
| `smoke` | 16 / 16², 16 / 8 / 8 | 1 epoch | CI; checks the pipeline, numbers are meaningless |

## Running

```bash
python -m benchmarks.run burgers_forward --scale quick          # one task
python -m benchmarks.run --all --scale full --device cuda       # everything, on a GPU
```

`--device auto` (the default) uses CUDA if present, then Apple MPS, then CPU.
Pass `cpu`, `mps`, `cuda` or `cuda:1` to choose explicitly, and `--amp` for
mixed precision on CUDA. On a cluster, adapt [`slurm.sh`](slurm.sh).

Each task writes to `benchmarks/runs/<scale>/<task>/`:

- `<task>.pt` — the pretrained model (selected EMA weights, recipe, normalizer,
  data settings and test metrics in one file)
- `metrics.json` — test metrics, sampling speed, timings, environment and the
  exact settings used
- `train/` — Trainer checkpoints and loss curves

Data generation is part of the run and is seeded, so a rerun reproduces the
same splits. The model and shuffling seed is `--seed` (default 42).

## Publishing results

1. **Table.** Regenerate the docs table from the runs you want to show:

   ```bash
   python -m benchmarks.report benchmarks/runs/full --write docs/benchmarks.md
   ```

2. **Weights.** Register the models and upload them as GitHub release assets:

   ```bash
   python -m benchmarks.publish benchmarks/runs/full --tag weights-v1
   ```

   This adds each model to `flowpde/pretrained/registry.json` (URL + sha256)
   and prints the `gh release` commands for the upload. Upload first, then
   commit the registry, so no released version points at a missing file.

3. **Release** a new flowpde version so `pip install flowpde` users see the
   new registry entries; `load_pretrained("<task>_convnet")` then downloads and
   verifies the file.

## Adding a task

Add an entry to `TASKS` in [`tasks.py`](tasks.py): the generator and its
settings, `problem` (and `inverse_mode` for Darcy), and any changes to the
shared model, training or evaluation defaults. Give it a `quick` override
that fits a laptop. The smoke scale is derived automatically, and
`tests/test_benchmarks.py` runs every task at that scale.
