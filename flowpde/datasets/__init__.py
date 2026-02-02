"""
Dataset modules for FlowPDE
=============================

PyTorch Dataset classes for forward and inverse PDE problems.

Available Datasets:
-------------------

APEBench Integration (Recommended):
    - APEBenchProvider: Factory for generating datasets from APEBench
    - BurgersScenario: 1D/2D Burgers equation via spectral solvers
    - PoissonScenario: 1D/2D/3D Poisson equation via spectral solvers

Legacy Datasets:
    - PoissonForwardDataset: (source, coefficient) → solution
    - PoissonInverseDataset: observation → (source, coefficient)
    - BurgersForwardDataset: initial_condition → solution_trajectory
    - BurgersInverseDataset: final_observation → initial_condition

Wrappers:
    - FlowDatasetWrapper: Convert {'input', 'target'} → {'f', 'u'} for flows

APEBench Quick Start:
---------------------
    from flowpde.datasets.apebench import APEBenchProvider
    
    # Create Burgers 1D dataset with caching
    dataset = APEBenchProvider.create(
        pde='burgers_1d',
        problem='forward',
        num_points=160,
        viscosity=0.0003,
        num_train_samples=500,
        cache=True,  # Recommended
    )
    
    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

Legacy Usage:
-------------
    from flowpde.datasets import PoissonForwardDataset
    dataset = PoissonForwardDataset('data/datasets/poisson/forward/train.pt')
"""

# Legacy datasets (for backward compatibility)
from .poisson import PoissonForwardDataset, PoissonInverseDataset
from .burgers import BurgersForwardDataset, BurgersInverseDataset
from .wrappers import FlowDatasetWrapper, InverseFlowDatasetWrapper

# APEBench integration (new, recommended)
try:
    from .apebench import (
        APEBenchProvider,
        BurgersScenario,
        PoissonScenario,
        CacheManager,
    )
    _APEBENCH_AVAILABLE = True
except ImportError:
    _APEBENCH_AVAILABLE = False
    # Create placeholder for documentation
    APEBenchProvider = None
    BurgersScenario = None
    PoissonScenario = None
    CacheManager = None


def check_apebench_available() -> bool:
    """Check if APEBench integration is available."""
    return _APEBENCH_AVAILABLE


__all__ = [
    # APEBench integration (new)
    'APEBenchProvider',
    'BurgersScenario',
    'PoissonScenario',
    'CacheManager',
    'check_apebench_available',
    
    # Legacy datasets
    'PoissonForwardDataset',
    'PoissonInverseDataset',
    'BurgersForwardDataset',
    'BurgersInverseDataset',
    
    # Wrappers
    'FlowDatasetWrapper',
    'InverseFlowDatasetWrapper',
]
