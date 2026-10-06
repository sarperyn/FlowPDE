"""
Exponax Integration Module for FlowPDE

Main Components:
    - `PoissonGenerator`: source → solution pairs for the Poisson equation
    - `BurgersGenerator`: IC → final-state pairs for the Burgers equation
    - `DarcyGenerator`: (κ, f) → solution pairs for variable-coefficient
      Poisson / Darcy-flow (−∇·(κ∇u) = f)
    - `PDEDataset`: PyTorch Dataset wrapping generated data
    - `DarcyDataset`: PyTorch Dataset for Darcy-flow data
    - `jax_to_torch`: array conversion utility

Forward/inverse direction is controlled directly by each generator's
`problem` argument. Returned samples always expose `'input'` and
`'target'` keys for training.

Quick Start:

```python
from flowpde.datasets.exponax import PoissonGenerator, DarcyGenerator

# Constant-coefficient Poisson
gen = PoissonGenerator(num_points=64, domain_extent=10.0)
dataset = gen.generate(num_samples=1000, seed=42)

# Variable-coefficient Poisson / Darcy flow
gen = DarcyGenerator(num_points=64, kappa_alpha=2.0, kappa_tau=3.0)
dataset = gen.generate(num_samples=1000, seed=42)
sample = dataset[0]
# sample['input']  → (2, 64, 64)  cat([κ, f])
# sample['target'] → (1, 64, 64)  solution u
```
"""

try:
    import exponax  # noqa: F401
    import jax  # noqa: F401
except ImportError as error:
    raise ImportError(
        "FlowPDE's PDE data generators need JAX and Exponax, which are optional. "
        'Install them with: pip install "flowpde[data]"'
    ) from error

from .base import GenerationConfig, PDEDataset
from .burgers import BurgersConfig, BurgersGenerator
from .darcy import DarcyConfig, DarcyDataset, DarcyGenerator
from .generator import (
    ExponaxDatasetGenerator,
    FourierFieldConfig,
    log_uniform,
    sample_fourier_fields,
)
from .poisson import PoissonConfig, PoissonGenerator
from .utilities import (
    compute_normalization_stats,
    jax_to_torch,
    sample_sine_fields,
)

__all__ = [
    'PDEDataset',
    'GenerationConfig',
    'ExponaxDatasetGenerator',
    'FourierFieldConfig',
    'sample_fourier_fields',
    'log_uniform',
    'PoissonGenerator',
    'PoissonConfig',
    'BurgersGenerator',
    'BurgersConfig',
    'DarcyGenerator',
    'DarcyConfig',
    'DarcyDataset',
    'jax_to_torch',
    'compute_normalization_stats',
    'sample_sine_fields',
]
