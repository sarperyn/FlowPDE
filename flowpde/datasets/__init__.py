"""
Dataset Modules for FlowPDE
============================

Provides PyTorch Dataset classes for PDE problems using Exponax
for direct PDE data generation. All datasets return `sample['input']` and
`sample['target']`; choosing `problem='inverse'` swaps the learning
direction instead of requiring a wrapper.

Main Components:
----------------
- PoissonGenerator: Generate Poisson equation datasets
- BurgersGenerator: Generate Burgers equation datasets
- DarcyGenerator: Generate Darcy-flow / variable-coefficient Poisson datasets
- PDEDataset: PyTorch Dataset for generated data
- DarcyDataset: PyTorch Dataset for Darcy-flow data

Supported PDEs:
---------------
- Poisson equation (1D/2D/3D): Elliptic source-to-solution
- Burgers equation (1D/2D): Time-dependent viscous flow
- Darcy flow (1D/2D): Variable-coefficient Poisson -∇·(κ∇u)=f

Quick Start:
------------
    from flowpde.datasets import PoissonGenerator
    from torch.utils.data import DataLoader

    # 1. Generate Poisson 2D dataset (source → solution)
    gen = PoissonGenerator(num_points=64, domain_extent=10.0)
    dataset = gen.generate(num_samples=1000, seed=42)

    # 2. Use with DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        condition = batch['input']   # Conditioning input
        target = batch['target']     # Target to learn

Burgers Example:
----------------
    from flowpde.datasets import BurgersGenerator

    gen = BurgersGenerator(num_spatial_dims=1, num_points=160, num_steps=50)
    dataset = gen.generate(num_samples=500, seed=0)

Dependencies:
-------------
The generators need JAX and Exponax: pip install "flowpde[data]"
`FieldNormalizer` is pure PyTorch and works without them.
"""

from importlib import import_module

from .normalization import FieldNormalizer

# Everything else lives in the Exponax integration, which imports JAX.
# Load it on first attribute access so `import flowpde.datasets` and
# `FieldNormalizer` work without the optional `data` extra installed.
_EXPONAX_NAMES = frozenset({
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
    'sample_sine_fields',
})


def __getattr__(name):
    if name in _EXPONAX_NAMES:
        value = getattr(import_module('.exponax', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _EXPONAX_NAMES)


__all__ = [
    # Generators
    'PoissonGenerator',
    'PoissonConfig',
    'BurgersGenerator',
    'BurgersConfig',
    'DarcyGenerator',
    'DarcyConfig',

    # Datasets
    'PDEDataset',
    'DarcyDataset',
    'GenerationConfig',
    'ExponaxDatasetGenerator',
    'FourierFieldConfig',

    # Normalization
    'FieldNormalizer',

    # Utilities
    'sample_fourier_fields',
    'log_uniform',
    'jax_to_torch',
    'sample_sine_fields',
]
