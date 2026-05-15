"""
Flow objects for FlowPDE.

This module provides flow-based generative models for solving PDEs:

Main Classes:
- NeuralODEFlow: Conditional continuous-time flow dynamics

Components:
- PathInterpolant: Base class for interpolation paths
- TimeSampler: Base class for time distributions
- Coupling: Base class for coupling strategies

"""

# Core flow classes
from .neural_ode import NeuralODEFlow, NeuralODELogProbVectorField

# Components for advanced customization
from .components import (
    # Paths
    PathInterpolant,
    LinearPath,
    OTConditionalPath,
    get_path,
    # Time samplers
    TimeSampler,
    UniformSampler,
    LogitNormalSampler,
    BetaSampler,
    TruncatedSampler,
    get_time_sampler,
    # Couplings
    Coupling,
    IndependentCoupling,
    MiniBatchOTCoupling,
    get_coupling,
)

__all__ = [
    # Main classes
    'NeuralODEFlow',
    'NeuralODELogProbVectorField',
    
    # Path components
    'PathInterpolant',
    'LinearPath',
    'OTConditionalPath',
    'get_path',
    
    # Time sampler components
    'TimeSampler',
    'UniformSampler',
    'LogitNormalSampler',
    'BetaSampler',
    'TruncatedSampler',
    'get_time_sampler',
    
    # Coupling components
    'Coupling',
    'IndependentCoupling',
    'MiniBatchOTCoupling',
    'get_coupling',
]
