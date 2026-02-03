"""
Modular components for Flow Matching algorithms.

This module provides composable building blocks:
- PathInterpolant: Interpolation path strategies (linear, OT, etc.)
- TimeSampler: Time distribution samplers (uniform, logit-normal, etc.)
- Coupling: Noise-data coupling strategies
"""

from .paths import (
    PathInterpolant,
    LinearPath,
    OTConditionalPath,
    get_path,
)

from .time_samplers import (
    TimeSampler,
    UniformSampler,
    LogitNormalSampler,
    BetaSampler,
    TruncatedSampler,
    get_time_sampler,
)

from .couplings import (
    Coupling,
    IndependentCoupling,
    MiniBatchOTCoupling,
    get_coupling,
)

__all__ = [
    # Paths
    'PathInterpolant',
    'LinearPath',
    'OTConditionalPath',
    'get_path',
    # Time samplers
    'TimeSampler',
    'UniformSampler',
    'LogitNormalSampler',
    'BetaSampler',
    'TruncatedSampler',
    'get_time_sampler',
    # Couplings
    'Coupling',
    'IndependentCoupling',
    'MiniBatchOTCoupling',
    'get_coupling',
]
