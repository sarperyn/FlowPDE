"""
Poisson Equation Data Generation
==================================

Generates source → solution pairs for the Poisson equation
using Exponax's spectral Poisson solver on periodic domains.

The Poisson equation:

    ∇²u = f    (on a periodic domain)

Given a smooth sine/cosine source term *f*, the solver computes the
solution *u* spectrally.

Example::

    generator = PoissonGenerator(num_points=64, domain_extent=10.0)
    dataset = generator.generate(num_samples=1000, seed=42)
"""

from dataclasses import dataclass
from typing import Literal, Optional

import jax
import exponax as ex

from .base import GenerationConfig, PDEDataset
from .generator import ExponaxDatasetGenerator
from .utilities import sample_sine_fields


@dataclass
class PoissonConfig(GenerationConfig):
    """
    Configuration specific to the Poisson equation.

    Attributes:
        order: Order of the Poisson operator (default 2 -> Laplacian).
        source_num_terms: Number of sine/cosine terms per source field.
        source_max_mode: Largest integer wavenumber sampled per dimension.
    """
    order: int = 2
    source_num_terms: int = 3
    source_max_mode: int = 3


class PoissonGenerator(ExponaxDatasetGenerator):
    r"""
    Generate Poisson equation datasets using Exponax.

    Workflow:
        1. Create simple smooth sine/cosine source terms *f*
        2. Solve $\nabla^2 u = f$ via 'exponax.poisson.Poisson'
        3. Convert JAX arrays -> PyTorch tensors
        4. Wrap in a 'PDEDataset'

    Args:
        config: A 'PoissonConfig' instance.  All keyword arguments
                are forwarded to 'PoissonConfig' if *config* is None.
    """

    config_cls = PoissonConfig

    def generate(
        self,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        problem: Literal['forward', 'inverse'] = 'forward',
    ) -> PDEDataset:
        """
        Generate a Poisson dataset.

        Args:
            num_samples: Override 'config.num_samples'.
            seed: Override 'config.seed'.
            problem: 'forward' (data->solution) or 'inverse' (solution->data).

        Returns:
            A ``PDEDataset`` with keys ``'source'`` and ``'solution'``.
        """
        cfg, n, s = self.resolve_run(num_samples, seed)
        self.validate_problem(problem)

        solver = ex.poisson.Poisson(
            num_spatial_dims=cfg.num_spatial_dims,
            domain_extent=cfg.domain_extent,
            num_points=cfg.num_points,
            order=cfg.order,
        )

        sources = sample_sine_fields(
            jax.random.PRNGKey(s),
            n=n,
            num_spatial_dims=cfg.num_spatial_dims,
            num_points=cfg.num_points,
            domain_extent=cfg.domain_extent,
            num_terms=cfg.source_num_terms,
            max_mode=cfg.source_max_mode,
        )
        solutions = jax.vmap(solver)(sources)

        data = self.to_torch_data({
            'source': sources,
            'solution': solutions,
        })
        self.apply_observation_augmentation(
            data,
            observation_key='solution',
            problem=problem,
            n=n,
            seed=s,
        )

        self.print_summary("Poisson", n)
        return self.wrap_dataset(data, problem=problem)
