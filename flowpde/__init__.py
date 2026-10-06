"""
FlowPDE: Flow-based generative models for forward and inverse PDE problems.

Quick Start:
    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> from flowpde import NeuralODEFlow, FlowMatchingObjective, Trainer, UNet
    >>> from flowpde.datasets import PoissonGenerator, FieldNormalizer
    >>>
    >>> # Data: source -> solution pairs, normalized
    >>> generator = PoissonGenerator(num_spatial_dims=2, num_points=64)
    >>> train_ds = generator.generate(num_samples=1000, seed=42, problem="forward")
    >>> train_ds.set_normalizer(FieldNormalizer.from_dataset(train_ds))
    >>> loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    >>>
    >>> # The flow is the dynamics; the objective is how they are trained
    >>> model = UNet(spatial_dim=2, spatial_size=64, base_channels=64)
    >>> flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    >>> objective = FlowMatchingObjective(flow, path="linear", time_sampler="uniform")
    >>>
    >>> # Train
    >>> optimizer = torch.optim.Adam(objective.parameters(), lr=1e-4)
    >>> trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.999)
    >>> trainer.train(loader, epochs=100, print_stats_interval=10,
    ...               save_dir="results/poisson/", save_interval=25)
    >>>
    >>> # Sample: returns a flattened (B, D) tensor.  The solver and step
    >>> # count default to the ones the flow was built with; n_steps only
    >>> # applies to fixed-step solvers.
    >>> batch = next(iter(loader))
    >>> samples = flow.sample(condition=batch["input"], solver="euler", n_steps=50)

Architecture:
    - flowpde.core: Base classes for flows, solvers, and conditioners
    - flowpde.flows: Neural ODE flow objects and their pluggable components
    - flowpde.objectives: Training objectives (flow matching, maximum likelihood)
    - flowpde.models: Neural networks (MLP, UNet, ConvNet, ResNet)
    - flowpde.solvers: ODE solvers for sampling
    - flowpde.trainers: Training loop, EMA, evaluation, reflow
    - flowpde.datasets: Exponax PDE data generation and normalization
    - flowpde.utils: Error metrics, UQ metrics, and general utilities

Documentation: https://sarperyn.github.io/FlowPDE/
"""

__version__ = "0.1.0"

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
    '__version__',

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
