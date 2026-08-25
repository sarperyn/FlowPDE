# Exponax Integration

Dataset generation using [Exponax](https://fkoehler.site/exponax/) spectral PDE solvers.

Exponax provides pseudo-spectral solvers for a range of PDEs on periodic domains.
FlowPDE wraps these to generate training data for flow-based models.

All generators subclass `ExponaxDatasetGenerator`, which provides shared observation
augmentation, JAX→torch conversion, statistics, and dataset wrapping. Every dataset
emits `{'input': condition, 'target': solution}`.

## Base Generator

::: flowpde.datasets.exponax.generator.ExponaxDatasetGenerator

## Poisson Generator

Source → solution pairs for the Poisson equation $\nabla^2 u = f$ (1D/2D/3D).

::: flowpde.datasets.exponax.poisson.PoissonGenerator

::: flowpde.datasets.exponax.poisson.PoissonConfig

## Burgers Generator

Initial-condition → final-state pairs (optionally full trajectories) for the Burgers
equation $\partial_t u + u \cdot \nabla u = \nu \nabla^2 u$ (1D/2D).

::: flowpde.datasets.exponax.burgers.BurgersGenerator

::: flowpde.datasets.exponax.burgers.BurgersConfig

## Darcy Generator

$(\kappa, f) \rightarrow u$ for variable-coefficient Poisson
$-\nabla \cdot (\kappa \nabla u) = f$. `generate()` accepts `inverse_mode` in
`{'both', 'coefficient', 'source'}`.

::: flowpde.datasets.exponax.darcy.DarcyGenerator

::: flowpde.datasets.exponax.darcy.DarcyConfig

## Datasets

::: flowpde.datasets.exponax.base.PDEDataset

::: flowpde.datasets.exponax.base.GenerationConfig

## Utilities

::: flowpde.datasets.exponax.utilities
