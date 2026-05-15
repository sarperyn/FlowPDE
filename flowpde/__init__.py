"""
FlowPDE: A Backend Library for Normalizing Flows Solving Inverse Problems

Quick Start:
    >>> import torch
    >>> from flowpde.flows import NeuralODEFlow
    >>> from flowpde.objectives import FlowMatchingObjective
    >>> from flowpde.models import UNet
    >>> 
    >>> # Create model and flow
    >>> model = UNet(input_dim=64, base_ch=64)
    >>> flow = NeuralODEFlow(model)
    >>> objective = FlowMatchingObjective(flow)
    >>> 
    >>> # Train
    >>> optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    >>> for batch in trainloader:
    >>>     loss = objective.compute_loss(batch)
    >>>     loss.backward()
    >>>     optimizer.step()
    >>> 
    >>> # Sample
    >>> samples = flow.sample(condition=f, n_steps=50)

Architecture:
    - flowpde.flows: Neural ODE flow objects
    - flowpde.objectives: Training objectives
    - flowpde.models: Neural networks (MLP, UNet, ConvNet, ResNet)
    - flowpde.solvers: ODE/SDE solvers
    - flowpde.inverse: Posterior sampling, uncertainty quantification
    - flowpde.datasets: Dataset loaders
"""

# Direct imports - Natural API (recommended)
from flowpde.flows import (
    NeuralODEFlow,
)
from flowpde.objectives import (
    FlowMatchingObjective,
    MaximumLikelihoodObjective,
    create_flow_matching,
)
from flowpde.models import MLP, UNet, ConvNet, ResNet
from flowpde.solvers import ODEFlowSolver
from flowpde.trainers import Trainer

# Public API
__all__ = [
    
    # Flows (direct import - API)
    'NeuralODEFlow',

    # Objectives
    'FlowMatchingObjective',
    'MaximumLikelihoodObjective',
    'create_flow_matching',
    
    # Models (direct import - API)
    'MLP',
    'UNet',
    'ConvNet',
    'ResNet',
    
    # Solvers
    'ODEFlowSolver',
    
    # Trainers
    'Trainer',
    
]
