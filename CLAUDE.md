# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlowPDE is a PyTorch library for learning neural operators for PDEs using **Flow Matching** and **neural ODE flows**. Instead of solving PDEs numerically at inference time, it trains flow-based generative models that generate PDE solutions conditioned on PDE parameters.

The scientific goal is twofold:

1. **Forward problems** — show the learned operator solves the PDE (accuracy vs. a deterministic baseline, and vs. number of solver steps).
2. **Inverse problems + uncertainty quantification** — recover PDE inputs from noisy/partial observations, where the posterior is genuinely non-degenerate and a generative model earns its keep.

## Commands

```bash
# Environment (uv)
uv python install 3.11 && uv python pin 3.11
uv venv && uv sync
source .venv/bin/activate

# Tests
uv run -m pytest                    # full suite (~8s)
uv run -m pytest -m "not slow"      # skip the Exponax integration tests
uv run -m pytest tests/test_trainer.py -q

# Docs
mkdocs serve       # local preview
mkdocs build       # static site
```

## Architecture

### Module layout

```
flowpde/
  core/        Abstract bases: BaseFlow, BaseSolver, BaseConditioner + config system
  flows/       NeuralODEFlow (the dynamics) + components/ (paths, time samplers, couplings)
  objectives/  FlowMatchingObjective, MaximumLikelihoodObjective (how the flow is trained)
  models/      Velocity-field backbones: MLP, UNet, ConvNet, ResNet
  solvers/     ODEFlowSolver (torchdiffeq wrapper) for inference
  trainers/    Trainer, EMA, FlowEvaluator, reflow
  datasets/    FieldNormalizer + exponax/ (PDE data generation)
  utils/       Metrics, visualization, config/arg helpers
configs/       benchmark_configs.py — named, citable dataset configs
tests/         111 tests
notebooks/     Analysis and visualization
```

### The central separation: flow vs. objective

This is the most important structural fact about the codebase.

- **`NeuralODEFlow`** (`flows/neural_ode.py`) is the continuous-time dynamics: it owns the model, sampling by ODE integration, and exact log-likelihood via the Jacobian trace.
- **Objectives** (`objectives/`) are *how you train that flow*. `FlowMatchingObjective` regresses velocities along interpolation paths; `MaximumLikelihoodObjective` maximizes exact log-likelihood.

Both objectives wrap the same flow. Do not add training logic to the flow, or dynamics to an objective.

```python
flow = NeuralODEFlow(model, target_key="target", condition_key="input")
objective = FlowMatchingObjective(flow, path="linear", time_sampler="uniform")
trainer = Trainer(objective, optimizer, ...)
```

### Flow matching components

`FlowMatchingObjective` composes three pluggable pieces, so variants are *configuration*, not subclasses:

- **Path** (`flows/components/paths.py`): `LinearPath`, `OTConditionalPath`
- **TimeSampler** (`flows/components/time_samplers.py`): `UniformSampler`, `LogitNormalSampler`, `BetaSampler`, `TruncatedSampler`
- **Coupling** (`flows/components/couplings.py`): `IndependentCoupling`, `MiniBatchOTCoupling`
- **Source** (`flows/components/sources.py`): `GaussianSource`, `BatchSource` — where trajectories start

`create_flow_matching(flow, variant=...)` provides presets: `standard`, `rectified`, `ot_cfm`, `ot_cfm_coupled`.

**Invariant:** a path's `velocity()` must be the exact time derivative of its `interpolate()`. If they disagree, training silently regresses a target that doesn't match the interpolation shown to the model. `tests/test_components.py` checks this by finite differences — keep that test passing when adding a path.

### Batch keys and tensor shapes

- Datasets emit `{'input': condition, 'target': solution}`. The flow/objective map these via `target_key` / `condition_key`, which default to `'u'` / `'f'` — **pass `target_key="target", condition_key="input"` when using the Exponax datasets.**
- `BaseFlow._extract_target_condition` **flattens everything to `(B, D)`** before the model sees it. Convolutional backbones (`UNet`, `ConvNet`, `ResNet`) reshape internally using their `spatial_size` / `solution_channels` / `condition_channels` and return flattened velocity when `return_spatial=False`.
- Condition and target need not share a dimension. When they differ, pass `target_shape=` to `sample()`.

### Normalization (always use it)

Flow matching transports `N(0, I)` to the data. Raw PDE fields with non-unit scale make the velocity regression badly conditioned.

```python
normalizer = FieldNormalizer.from_dataset(train_ds)   # reuses stats already in metadata
train_ds.set_normalizer(normalizer)
val_ds.set_normalizer(normalizer)                     # SAME instance — never refit on val/test
```

- Statistics are keyed by **raw field name** (`source`, `solution`, `kappa`, `initial`, `final`), not by role. This is why one normalizer stays correct across `problem='forward'` and `'inverse'`.
- `obs_mask` has no registered statistics, so it passes through unchanged and stays binary.
- Datasets expose `input_fields` / `target_fields` — the field names per channel, needed to denormalize multi-field targets (Darcy `inverse_mode='both'`).
- **Report metrics in physical units.** Denormalize predictions before scoring; `FlowEvaluator` does this when given `normalizer=` and `target_fields=`.
- Store `normalizer.state_dict()` in the checkpoint via `Trainer(checkpoint_extra=...)` so inference can reproduce preprocessing.

### Training

`Trainer` (`trainers/trainer.py`) takes any object exposing `compute_loss(batch)` and `.model`.

```python
trainer = Trainer(
    objective, optimizer, scheduler,
    ema_decay=0.999,              # averaged weights are what you evaluate and ship
    validator=evaluator,          # FlowEvaluator
    monitor="rel_l2",
    val_interval=5,
    checkpoint_extra={"normalizer_state": normalizer.state_dict()},
)
```

- **EMA** (`trainers/ema.py`): updated once per optimizer step, with a warmup ramp. Validation and checkpointing run under averaged weights; checkpoints store EMA weights as `model_state`.
- **`FlowEvaluator`** (`trainers/evaluation.py`): integrates the ODE, denormalizes, and scores against ground truth. Evaluation noise uses a **fixed seed** so epoch-to-epoch differences reflect the model, not the draw. `ensemble_size > 1` scores the ensemble mean and reports spread — the UQ hook.
- **Model selection** uses the validation metric when a validator is present, falling back to train loss otherwise.

### Dataset generation

Generators subclass `ExponaxDatasetGenerator` (`datasets/exponax/generator.py`), which provides shared observation augmentation, JAX→torch conversion, statistics, and dataset wrapping.

- **`PoissonGenerator`**: source → solution (1D/2D/3D)
- **`BurgersGenerator`**: IC → final state (1D/2D)
- **`DarcyGenerator`**: (κ, f) → u, with `inverse_mode` in `{'both', 'coefficient', 'source'}`

`problem='inverse'` swaps input/target and enables `obs_noise_std` / `obs_mask_fraction`. When masking is on, the binary mask is **appended as an extra condition channel**, so build the model with `condition_channels` matching `sample['input'].shape[0]`.

`configs/benchmark_configs.py` holds named, citable dataset configs — prefer these for reported results.

## Gotchas that have already bitten this project

- **Training loss is not a model-selection signal.** The flow-matching loss has a large irreducible floor (many `(x_0, x_1)` pairs give the same `x_t`). Select on `FlowEvaluator` output, not `train_loss`.
- **Never refit normalization on validation/test.** Share the train normalizer instance.
- **The forward problem has a Dirac posterior.** For Poisson/Burgers forward maps, `p(u|f)` is deterministic — sampler spread there is model error, not physical uncertainty. Put the UQ story on inverse problems.
- **Straightness means chord deviation**, not the spread of velocity norms. `estimate_straightness` implements the Liu et al. definition with `mode='trajectory'` (the model's own ODE paths) and `mode='interpolant'`. A field that turns at constant speed must not score as straight.
- **The Poisson dataset is currently the "easy" variant** — 3 sine terms at wavenumbers 1–3 (`source_num_terms`, `source_max_mode`). That is a low-dimensional subspace and will produce optimistic errors. Restore a harder source distribution before reporting operator-learning results.

## Known open items

- **No worked end-to-end example.** Nothing outside tests and notebooks demonstrates a full run; see the library-design section for the intended shape.
- **`flow._target_dim` is set as a side effect of `compute_loss`.** A checkpoint loaded fresh for inference never calls it, so `sample()` silently falls back to the condition dimension. Set the target shape at construction instead.
- **Docstring escape warnings** in `datasets/exponax/{burgers,darcy}.py` (`SyntaxWarning: invalid escape sequence`) — the LaTeX docstrings need `r"""`.

## Library design: API first, config never required

**FlowPDE is a library.** The deliverable is `pip install flowpde` followed by composing objects in Python, the way one uses PyTorch itself. This constrains the architecture:

- **Core must never import a config layer.** `flowpde.FlowMatchingObjective(...)` has to work with zero config machinery in the picture. This is the non-negotiable rule; everything else below is preference.
- **Accept "string or instance" everywhere.** `get_path`, `get_time_sampler`, `get_coupling`, and `get_source` all take either a registry name or a constructed object. This is what makes the API config-friendly without making config a dependency — a config layer is just a caller that happens to pass strings. Follow this pattern for any new component.
- **Specify in Python, record in YAML/JSON.** Configuration objects are Python dataclasses (see `configs/benchmark_configs.py`), which give type checking, IDE completion, no string typos, and can carry things YAML cannot — a custom `PathInterpolant` instance, a callable. Serialization runs the *other* way: `get_config()` exists on `NeuralODEFlow`, `FlowMatchingObjective`, `MaximumLikelihoodObjective`, and every component, returning a plain dict. Dump that into the results directory so each run records what produced it.

A YAML-driven runner was considered and rejected: it would invert the dependency, make strings the primary interface, and serve the repo's own experiments rather than the library's users. If YAML *input* is ever wanted as a convenience, it belongs in an optional module that core never imports.

### The deleted training scripts

Four scripts (`train_poisson_2d_flowmatching.py`, `train_burgers_1d_flowmatching.py`, `train_burgers_1d_rectified_flow.py`, `train_darcy_2d_flowmatching.py`) were removed. They were ~940 lines that were largely copies of each other — hand-rolled argparse, print blocks, and object wiring — and because nothing imported them, a single refactor broke all four silently.

**Do not recreate per-PDE training scripts.** A user of this library should be able to write a training run in roughly twenty lines, and that is the interface to optimize. What the repo needs instead:

- One worked end-to-end example, in `docs/` or a notebook, that a user can copy.
- Good defaults, so the twenty lines are mostly meaningful choices.
- For the ablation matrix (path × time sampler × coupling × source × backbone × PDE × solver steps), a sweep helper over Python config objects — not a per-experiment script.

Keep `configs/benchmark_configs.py`. Dataclass-based named dataset configs carrying citations are exactly the right shape, and they slot into any sweep helper directly.

## Rectified Flow and reflow

These are two different things and the distinction matters:

- **The objective.** `create_flow_matching(flow, variant='rectified')` sets a linear path with logit-normal time sampling. Note that 1-rectified flow is mathematically identical to standard flow matching with a linear path and independent coupling — the objective alone straightens nothing. (The logit-normal sampler is from Esser et al. 2024/SD3; Liu et al. use uniform.)
- **Reflow** (`trainers/reflow.py`) is the procedure that does the straightening: generate `(z, ODE(z))` pairs with the current model, retrain on those pairs, repeat.

**The correctness requirement:** reflow training must use the *same* `z` that produced each generated target. The pairing is deterministic and induced by the model; resampling noise independently decouples the pairs and silently reduces reflow to training on the model's own samples. The previous implementation had exactly this bug — it stored the noise as `_z` and never read it.

This is why the source distribution is a first-class component:

```python
from flowpde.flows import BatchSource
from flowpde.trainers import generate_reflow_pairs, reflow

pairs = generate_reflow_pairs(objective, loader, n_steps=100)   # (z, ODE(z))
objective.source = BatchSource()                               # consume that exact z
# ...or run the whole loop:
reflow(objective, loader, optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-4),
       num_iterations=2, epochs_per_iteration=50)
```

`BatchSource` is `strict=True` by default so a missing key raises rather than silently resampling. `compute_loss` also skips the coupling when the source already defines the pairing, since minibatch-OT would reorder the pairs and destroy them. `reflow()` restores the original source on exit, so sampling behaviour is unchanged after it returns.

`tests/test_reflow.py::test_reflow_pairs_are_preserved_during_training` pins the old bug; `test_reflow_straightens_trajectories` checks the procedure actually reduces measured curvature.

## Testing conventions

- Prefer **analytically known** expectations over regression baselines. Existing examples: path velocity vs. finite differences, solver vs. `dx/dt = -x`, straightness of `v = 2tc` equals exactly 1/3, EMA recursion arithmetic.
- Backbones zero-initialize their output projection, so a freshly constructed model outputs exactly zero. Perturb `output_proj.weight` before testing input sensitivity.
- Mark tests that run the Exponax solvers with `@pytest.mark.slow`.
- When fixing a bug, add the failure mode as a test (see `test_detects_curvature_at_constant_speed`, which pins the old straightness metric's blind spot).

## Key dependencies

- `torch` — core deep learning
- `torchdiffeq` — ODE integration for inference
- `exponax` + `jax` — spectral PDE data generation
- `pyyaml` — config management
- `pytest` — tests (dev extra)
- `mkdocs-material` + `mkdocstrings` — documentation
