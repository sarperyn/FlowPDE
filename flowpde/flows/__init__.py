"""
Flow objects for FlowPDE.

This module provides flow-based generative models for solving PDEs:

Main Classes:
- NeuralODEFlow: Conditional continuous-time flow dynamics

Components:
- PathInterpolant: Base class for interpolation paths
- TimeSampler: Base class for time distributions
- Coupling: Base class for coupling strategies
- SourceDistribution: Base class for where trajectories start

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
    get_time_sampler,
    # Couplings
    Coupling,
    IndependentCoupling,
    MiniBatchOTCoupling,
    get_coupling,
    # Sources
    SourceDistribution,
    GaussianSource,
    BatchSource,
    get_source,
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
    'get_time_sampler',
    
    # Coupling components
    'Coupling',
    'IndependentCoupling',
    'MiniBatchOTCoupling',
    'get_coupling',

    # Source components
    'SourceDistribution',
    'GaussianSource',
    'BatchSource',
    'get_source',
]
