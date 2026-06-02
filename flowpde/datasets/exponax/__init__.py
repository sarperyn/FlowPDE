"""
Exponax Integration Module for FlowPDE

Main Components:
    - ``PoissonGenerator``: source → solution pairs for the Poisson equation
    - ``BurgersGenerator``: IC → final-state pairs for the Burgers equation
    - ``DarcyGenerator``: (κ, f) → solution pairs for variable-coefficient
      Poisson / Darcy-flow (−∇·(κ∇u) = f)
    - ``PDEDataset``: PyTorch Dataset wrapping generated data
    - ``DarcyDataset``: PyTorch Dataset for Darcy-flow data
    - ``jax_to_torch`` / ``torch_to_jax``: array conversion utilities

Quick Start::

    from flowpde.datasets.exponax import PoissonGenerator, DarcyGenerator

    # Constant-coefficient Poisson
    gen = PoissonGenerator(num_points=64, domain_extent=10.0)
    dataset = gen.generate(num_samples=1000, seed=42)

    # Variable-coefficient Poisson / Darcy flow
    gen = DarcyGenerator(num_points=64, kappa_alpha=2.0, kappa_tau=3.0)
    dataset = gen.generate(num_samples=1000, seed=42)
    sample = dataset[0]
    # sample['input']  → (2, 64, 64)  cat([κ, f])
    # sample['target'] → (1, 64, 64)  solution u
"""

from .base import PDEDataset, GenerationConfig
from .generator import (
    ExponaxDatasetGenerator,
    FourierFieldConfig,
    log_uniform,
    sample_fourier_fields,
)
from .poisson import PoissonGenerator, PoissonConfig
from .burgers import BurgersGenerator, BurgersConfig
from .darcy import DarcyGenerator, DarcyConfig, DarcyDataset
from .utilities import (
    jax_to_torch,
    torch_to_jax,
    compute_normalization_stats,
    apply_spectral_cutoff,
    gaussian_bump_ic_batch,
)

__all__ = [
    'PDEDataset',
    'GenerationConfig',
    'ExponaxDatasetGenerator',
    'FourierFieldConfig',
    'sample_fourier_fields',
    'log_uniform',
    'PoissonGenerator',
    'PoissonConfig',
    'BurgersGenerator',
    'BurgersConfig',
    'DarcyGenerator',
    'DarcyConfig',
    'DarcyDataset',
    'jax_to_torch',
    'torch_to_jax',
    'compute_normalization_stats',
    'apply_spectral_cutoff',
    'gaussian_bump_ic_batch',
]
