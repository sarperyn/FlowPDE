"""
APEBench Integration Module for FlowPDE
========================================

This module provides a unified interface for generating PDE datasets using
APEBench's JAX-based spectral solvers, converted to PyTorch format for use
with FlowPDE's normalizing flow training pipeline.

Supported PDEs:
---------------
- Burgers equation (1D, 2D)
- Poisson equation (1D, 2D)
- (Extensible to Navier-Stokes, Kuramoto-Sivashinsky, etc.)

Quick Start:
------------
    from flowpde.datasets.apebench import APEBenchProvider
    
    # Create a Burgers 1D dataset
    dataset = APEBenchProvider.create(
        pde='burgers_1d',
        problem='forward',
        resolution=160,
        viscosity=0.0003,
        n_train_samples=500,
        cache=True,  # Opt-in caching (recommended)
    )
    
    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

Architecture:
-------------
- APEBenchProvider: High-level factory for dataset creation
- BaseAPEBenchScenario: Abstract base class for PDE scenarios
- BurgersScenario: 1D/2D Burgers equation implementation
- PoissonScenario: 1D/2D Poisson equation implementation

Dependencies:
-------------
Requires optional APEBench dependencies:
    pip install apebench exponax jax

"""

from .provider import APEBenchProvider
from .base import BaseAPEBenchScenario
from .scenarios import BurgersScenario, PoissonScenario
from .converters import jax_to_torch, torch_to_jax
from .cache import CacheManager

__all__ = [
    # Main API
    'APEBenchProvider',
    
    # Base classes (for extension)
    'BaseAPEBenchScenario',
    
    # Concrete scenarios
    'BurgersScenario',
    'PoissonScenario',
    
    # Utilities
    'jax_to_torch',
    'torch_to_jax',
    'CacheManager',
]
