"""
Trainers for Normalizing Flow Algorithms

FlowPDE provides trainers for different flow-based training schemas:

- FlowMatchingTrainer: For flow matching with linear or OT paths
- CNFTrainer: For continuous normalizing flows with exact log probabilities  
- RectifiedFlowTrainer: For rectified flows with straight transport paths

All trainers are dataset-agnostic and work with both forward and inverse problems.
The choice of trainer depends on the flow algorithm, not the PDE or problem type.
"""

from .trainer import *
from ..utils.generic_training import *
from .flow_matching_trainer import FlowMatchingTrainer
from .cnf_trainer import CNFTrainer
from .rectified_flow_trainer import RectifiedFlowTrainer

__all__ = [
    'Trainer',
    'FlowMatchingTrainer',
    'CNFTrainer',
    'RectifiedFlowTrainer'
]

