# Changelog

All notable changes to FlowPDE are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, minor
releases may contain breaking API changes; they will be listed here.

## [Unreleased]

## [0.2.0] - 2026-10-06

### Added

- Library logging: progress and dataset summaries go through the `flowpde`
  logger. `flowpde.set_verbosity("WARNING")` silences them, and
  `flowpde.disable_default_handler()` routes them to your own logging setup.
  Default output is unchanged.
- `Trainer.load_checkpoint(..., weights_only=...)`, for trusted checkpoints that
  store custom objects in `checkpoint_extra`.
- The Python examples in the README and getting-started docs run in CI.
- `CITATION.cff`, Zenodo metadata, `CONTRIBUTING.md`, and issue and pull-request
  templates.

### Changed

- `Trainer.load_checkpoint` loads with `torch.load(weights_only=True)` by default,
  so a checkpoint from an untrusted source cannot run code. Checkpoints written
  by `Trainer` load unchanged; one that pickles custom objects in
  `checkpoint_extra` now needs `weights_only=False`.
- `zip` over UNet stages and normalizer fields is now `strict=True`, so a length
  mismatch raises instead of silently truncating.
- Wider lint rules (pycodestyle, import sorting, bugbear) across the codebase.

## [0.1.0] - 2026-10-06

First public release.

### Added

- `NeuralODEFlow`: conditional continuous-time flow with pluggable paths
  (linear, OT-conditional), time samplers (uniform, logit-normal, beta),
  couplings (independent, mini-batch OT) and source distributions.
- Training objectives: `FlowMatchingObjective` and `MaximumLikelihoodObjective`
  over the same flow.
- Backbones: `MLP`, `ConvNet`, `ResNet` and `UNet`, with FiLM and concatenation
  conditioners.
- `ODEFlowSolver` with fixed-step and adaptive solvers via torchdiffeq.
- `Trainer` with EMA, validation-based model selection and checkpointing;
  `FlowEvaluator` for evaluation in physical units; reflow utilities.
- Poisson, Burgers and Darcy dataset generators built on Exponax, for forward
  and inverse problems with noisy and partial observations.
- `FieldNormalizer`, error metrics and uncertainty-quantification metrics
  (CRPS, energy score, rank histogram, spread–skill, coverage).

### Packaging

- JAX and Exponax are an optional extra: `pip install "flowpde[data]"`. The
  core library and `FieldNormalizer` import without them.

[Unreleased]: https://github.com/sarperyn/FlowPDE/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sarperyn/FlowPDE/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sarperyn/FlowPDE/releases/tag/v0.1.0
