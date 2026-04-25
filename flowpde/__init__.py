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
    - flowpde.flows: Flow algorithms (FlowMatching, CNF)
    - flowpde.models: Neural networks (MLP, UNet, ConvNet, ResNet)
    - flowpde.solvers: ODE/SDE solvers
    - flowpde.inverse: Posterior sampling, uncertainty quantification
    - flowpde.datasets: Dataset loaders
"""

# Direct imports - Natural API (recommended)
from flowpde.flows import FlowMatching, ContinuousNormalizingFlow, create_flow_matching
from flowpde.models import MLP, UNet, ConvNet, ResNet
from flowpde.solvers import ODEFlowSolver
from flowpde.trainers import FlowTrainer, CNFTrainer

# Public API
__all__ = [
    
    # Flows (direct import - API)
    'FlowMatching',
    'ContinuousNormalizingFlow',
    'create_flow_matching',
    
    # Models (direct import - API)
    'MLP',
    'UNet',
    'ConvNet',
    'ResNet',
    
    # Solvers
    'ODEFlowSolver',
    
    # Trainers
    'FlowTrainer',
    'CNFTrainer',
    
]
