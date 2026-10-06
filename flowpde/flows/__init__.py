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

# Components for advanced customization
from .components import (
    BatchSource,
    BetaSampler,
    Coupling,
    GaussianSource,
    IndependentCoupling,
    LinearPath,
    LogitNormalSampler,
    MiniBatchOTCoupling,
    OTConditionalPath,
    PathInterpolant,
    SourceDistribution,
    TimeSampler,
    UniformSampler,
    get_coupling,
    get_path,
    get_source,
    get_time_sampler,
)

# Core flow classes
from .neural_ode import NeuralODEFlow, NeuralODELogProbVectorField

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
