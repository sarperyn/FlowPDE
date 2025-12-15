"""
Dataset modules for FlowPDE
=============================

PyTorch Dataset classes for forward and inverse PDE problems.

Available Datasets:
-------------------

Poisson Equation (Static/Elliptic):
    - PoissonForwardDataset: (source, coefficient) → solution
    - PoissonInverseDataset: observation → (source, coefficient)

Burgers Equation (Time-dependent/Parabolic):
    - BurgersForwardDataset: initial_condition → solution_trajectory
    - BurgersInverseDataset: final_observation → initial_condition

Usage Example:
--------------
    from flowpde.datasets import PoissonForwardDataset
    from torch.utils.data import DataLoader
    
    # Load dataset
    dataset = PoissonForwardDataset('data/datasets/poisson/forward/train.pt')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Iterate
    for batch in loader:
        inputs = batch['input']    # (B, 2, H, W) - source + coefficient
        targets = batch['target']  # (B, 1, H, W) - solution
        # Train your model...
"""

from .poisson import PoissonForwardDataset, PoissonInverseDataset
from .burgers import BurgersForwardDataset, BurgersInverseDataset
from .wrappers import FlowDatasetWrapper, InverseFlowDatasetWrapper

__all__ = [
    'PoissonForwardDataset',
    'PoissonInverseDataset',
    'BurgersForwardDataset',
    'BurgersInverseDataset',
    'FlowDatasetWrapper',
    'InverseFlowDatasetWrapper',
]
