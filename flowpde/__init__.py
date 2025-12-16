"""
FlowPDE: A Backend Library for Normalizing Flows Solving Inverse Problems

Quick Start:
    >>> import torch
    >>> from flowpde.flows import FlowMatching
    >>> from flowpde.models import UNet
    >>> 
    >>> # Create model and flow
    >>> model = UNet(input_dim=64, base_ch=64)
    >>> flow = FlowMatching(model)
    >>> 
    >>> # Train
    >>> optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    >>> for batch in trainloader:
    >>>     loss = flow.compute_loss(batch)
    >>>     loss.backward()
    >>>     optimizer.step()
    >>> 
    >>> # Sample
    >>> samples = flow.sample(condition=f, n_steps=50)

Architecture:
    - flowpde.flows: Flow algorithms (FlowMatching, CNF, RectifiedFlow)
    - flowpde.models: Neural networks (MLP, UNet)
    - flowpde.solvers: ODE/SDE solvers
    - flowpde.inverse: Posterior sampling, uncertainty quantification
    - flowpde.datasets: Dataset loaders
"""

# Direct imports - Natural API (recommended)
from flowpde.flows import FlowMatching, ContinuousNormalizingFlow, RectifiedFlow
from flowpde.models import MLP, UNet
from flowpde.solvers import ODEFlowSolver
from flowpde.trainers import FlowMatchingTrainer, CNFTrainer, RectifiedFlowTrainer
from flowpde.datasets import (
    PoissonForwardDataset, PoissonInverseDataset,
    BurgersForwardDataset, BurgersInverseDataset,
    FlowDatasetWrapper, InverseFlowDatasetWrapper
)

# Public API
__all__ = [
    
    # Flows (direct import - API)
    'FlowMatching',
    'ContinuousNormalizingFlow',
    'RectifiedFlow',
    
    # Models (direct import - API)
    'MLP',
    'UNet',
    
    # Solvers
    'ODEFlowSolver',
    
    # Trainers
    'FlowMatchingTrainer',
    'CNFTrainer',
    'RectifiedFlowTrainer',
    
    # Datasets
    'PoissonForwardDataset',
    'PoissonInverseDataset',
    'BurgersForwardDataset',
    'BurgersInverseDataset',
    'FlowDatasetWrapper',
    'InverseFlowDatasetWrapper',
]

