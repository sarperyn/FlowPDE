"""
FlowPDE Core Module

This module provides the foundational abstractions and base classes
for the FlowPDE library, including:
- Base classes for flows, solvers, and conditioners
- Configuration management
- Common interfaces and protocols
"""

from .base_flow import BaseFlow
from .base_solver import BaseSolver, ODESolver, SDESolver
from .base_conditioner import BaseConditioner, ConcatConditioner, FiLMConditioner, NullConditioner
from .config import (
    Config,
    ConfigDict,
    BaseConfig,
    ModelConfig,
    SolverConfig,
    FlowConfig,
    TrainingConfig,
    InferenceConfig,
    DataConfig,
    ExperimentConfig,
    load_config,
    create_default_config,
)

__all__ = [
    # Base classes
    'BaseFlow',
    'BaseSolver',
    'ODESolver',
    'SDESolver',
    'BaseConditioner',
    'ConcatConditioner',
    'FiLMConditioner',
    'NullConditioner',
        
    # Configuration
    'Config',
    'ConfigDict',
    'BaseConfig',
    'ModelConfig',
    'SolverConfig',
    'FlowConfig',
    'TrainingConfig',
    'InferenceConfig',
    'DataConfig',
    'ExperimentConfig',
    'load_config',
    'create_default_config',
]
