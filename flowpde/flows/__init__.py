"""
Flow algorithms for FlowPDE.

This module provides flow-based generative models for solving PDEs:

Main Classes:
- FlowMatching: Unified flow matching (standard, rectified, OT-CFM)
- ContinuousNormalizingFlow: Original CNF formulation

Components (for advanced customization):
- PathInterpolant: Base class for interpolation paths
- TimeSampler: Base class for time distributions
- Coupling: Base class for coupling strategies

Factory Functions:
- create_flow_matching: Create FlowMatching with preset configurations
"""

# Core flow classes
from .flow_matching import FlowMatching, create_flow_matching
from .cnf import ContinuousNormalizingFlow

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
    'FlowMatching',
    'ContinuousNormalizingFlow',
    'create_flow_matching',
    
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
