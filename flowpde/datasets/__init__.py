"""
Dataset Modules for FlowPDE
============================

Provides PyTorch Dataset classes for PDE problems using APEBench
for procedural data generation.

Main Components:
----------------
- APEBenchProvider: Factory for generating PDE datasets
- FlowDatasetWrapper: Converts datasets to flow training format

Supported PDEs:
---------------
- Burgers equation (1D): Time-dependent viscous flow
- Poisson equation (2D): Elliptic source-to-solution

Quick Start:
------------
    from flowpde.datasets import APEBenchProvider, FlowDatasetWrapper
    from torch.utils.data import DataLoader
    
    # 1. Generate Burgers 1D dataset
    dataset = APEBenchProvider.create(
        pde='burgers_1d',
        problem='forward',  # IC → final state
        num_points=160,
        viscosity=0.0003,
        num_train_samples=500,
        cache=True,  # Recommended: cache to disk
    )
    
    # 2. Wrap for flow training (converts to {'f', 'u'} format)
    flow_dataset = FlowDatasetWrapper(dataset)
    
    # 3. Use with DataLoader
    loader = DataLoader(flow_dataset, batch_size=32, shuffle=True)
    
    for batch in loader:
        condition = batch['f']  # Conditioning input
        target = batch['u']     # Target to learn

Poisson Example:
----------------
    # Generate Poisson 2D dataset (source → solution)
    dataset = APEBenchProvider.create(
        pde='poisson_2d',
        problem='forward',
        num_points=64,
        num_train_samples=1000,
        cache=True,
    )

Dependencies:
-------------
Requires APEBench: pip install apebench exponax jax
"""

# Wrappers (always available)
from .wrappers import FlowDatasetWrapper, InverseFlowDatasetWrapper

# APEBench integration
try:
    from .apebench import (
        APEBenchProvider,
        BurgersScenario,
        PoissonScenario,
        CacheManager,
    )
    _APEBENCH_AVAILABLE = True
except ImportError as e:
    _APEBENCH_AVAILABLE = False
    _APEBENCH_ERROR = str(e)
    APEBenchProvider = None
    BurgersScenario = None
    PoissonScenario = None
    CacheManager = None


def check_apebench_available() -> bool:
    """Check if APEBench integration is available."""
    if not _APEBENCH_AVAILABLE:
        print(f"APEBench not available: {_APEBENCH_ERROR}")
        print("Install with: pip install apebench exponax jax")
    return _APEBENCH_AVAILABLE


__all__ = [
    # APEBench (primary)
    'APEBenchProvider',
    'BurgersScenario',
    'PoissonScenario',
    'CacheManager',
    'check_apebench_available',
    
    # Wrappers
    'FlowDatasetWrapper',
    'InverseFlowDatasetWrapper',
]
