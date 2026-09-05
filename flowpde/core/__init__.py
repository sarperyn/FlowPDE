"""
FlowPDE Core Module

This module provides the foundational abstractions and base classes
for the FlowPDE library, including:
- Base classes for flows, solvers, and conditioners
- Common interfaces and protocols
"""

from .base_flow import BaseFlow
from .base_solver import BaseSolver, ODESolver
from .base_conditioner import BaseConditioner, ConcatConditioner, FiLMConditioner, NullConditioner

__all__ = [
    # Base classes
    'BaseFlow',
    'BaseSolver',
    'ODESolver',
    'BaseConditioner',
    'ConcatConditioner',
    'FiLMConditioner',
    'NullConditioner',
]
