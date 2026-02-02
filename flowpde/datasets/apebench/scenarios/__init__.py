"""
Scenarios Package for APEBench Integration
===========================================

Contains concrete implementations of PDE scenario wrappers.
"""

from .burgers import BurgersScenario, BurgersSingleChannelScenario
from .poisson import PoissonScenario, PoissonDataset

__all__ = [
    'BurgersScenario',
    'BurgersSingleChannelScenario',
    'PoissonScenario',
    'PoissonDataset',
]
