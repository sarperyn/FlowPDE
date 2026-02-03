"""
Trainers for Normalizing Flow Algorithms.

FlowPDE provides trainers for different flow-based training schemas:

Main Classes:
- FlowTrainer: Unified trainer for all flow matching variants
  - Works with FlowMatching (standard, rectified, OT-CFM)
  - Includes reflow() for iterative path straightening
  - Supports gradient clipping, mixed precision

- CNFTrainer: For continuous normalizing flows with exact log probabilities

All trainers are dataset-agnostic and work with both forward and inverse problems.
The choice of trainer depends on the flow algorithm, not the PDE or problem type.

Example:
    ```python
    from flowpde.flows import FlowMatching
    from flowpde.trainers import FlowTrainer
    
    # Create flow
    flow = FlowMatching(model, path='linear', time_sampler='uniform')
    
    # Create trainer
    trainer = FlowTrainer(flow, optimizer, scheduler, device='cuda')
    
    # Train
    trainer.train(data_loader, epochs=100, save_dir='results/')
    
    # Optional: reflow for straighter paths
    trainer.reflow(data_loader, num_iterations=2)
    ```
"""

from .trainer import Trainer
from .flow_trainer import FlowTrainer
from .cnf_trainer import CNFTrainer
from ..utils.generic_training import *

__all__ = [
    # Base class
    'Trainer',
    
    # Unified trainer
    'FlowTrainer',
    
    # CNF trainer
    'CNFTrainer',
]
