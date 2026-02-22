"""
Exponax Integration Module for FlowPDE

Main Components:
    - ``PoissonGenerator``: source → solution pairs for the Poisson equation
    - ``BurgersGenerator``: IC → final-state pairs for the Burgers equation
    - ``PDEDataset``: PyTorch Dataset wrapping generated data
    - ``jax_to_torch`` / ``torch_to_jax``: array conversion utilities

Quick Start::

    from flowpde.datasets.exponax import PoissonGenerator

    gen = PoissonGenerator(num_points=64, domain_extent=10.0)
    dataset = gen.generate(num_samples=1000, seed=42)

    sample = dataset[0]
    source, solution = sample['input'], sample['target']
"""

from .base import PDEDataset, GenerationConfig
from .poisson import PoissonGenerator, PoissonConfig
from .burgers import BurgersGenerator, BurgersConfig
from .converters import jax_to_torch, torch_to_jax, compute_normalization_stats

__all__ = [
    'PDEDataset',
    'GenerationConfig',
    'PoissonGenerator',
    'PoissonConfig',
    'BurgersGenerator',
    'BurgersConfig',
    'jax_to_torch',
    'torch_to_jax',
    'compute_normalization_stats',
]
