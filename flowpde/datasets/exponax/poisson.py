"""
Poisson Equation Data Generation
==================================

Generates source → solution pairs for the Poisson equation
using Exponax's spectral Poisson solver on periodic domains.

The Poisson equation:

    ∇²u = f    (on a periodic domain)

Given a random source term *f* (generated via truncated Fourier
series), the solver computes the solution *u* spectrally.

Example::

    generator = PoissonGenerator(num_points=64, domain_extent=10.0)
    dataset = generator.generate(num_samples=1000, seed=42)
"""

from dataclasses import dataclass
from typing import Optional, Literal

import jax
import jax.numpy as jnp
import exponax as ex

from .base import GenerationConfig, PDEDataset
from .converters import jax_to_torch, compute_normalization_stats


@dataclass
class PoissonConfig(GenerationConfig):
    """
    Configuration specific to the Poisson equation.

    Attributes:
        order: Order of the Poisson operator (default 2 -> Laplacian).
        ic_cutoff: Fourier cutoff for random source generation.
        ic_max_one: Whether to normalize ICs so max absolute value is 1.
    """
    order: int = 2
    ic_cutoff: int = 5
    ic_max_one: bool = True


class PoissonGenerator:
    """
    Generate Poisson equation datasets using Exponax.

    Workflow:
        1. Create random source terms *f* using ``exponax.ic.RandomTruncatedFourierSeries``
        2. Solve $\nabla^2 u = f$ via 'exponax.poisson.Poisson'
        3. Convert JAX arrays -> PyTorch tensors
        4. Wrap in a 'PDEDataset'

    Args:
        config: A 'PoissonConfig' instance.  All keyword arguments
                are forwarded to 'PoissonConfig' if *config* is None.
    """

    def __init__(self, config: Optional[PoissonConfig] = None, **kwargs):
        if config is None:
            config = PoissonConfig(**kwargs)
        self.config = config

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
        cfg = self.config
        n = num_samples or cfg.num_samples
        s = seed if seed is not None else cfg.seed

        #Exponax objects
        solver = ex.poisson.Poisson(
            num_spatial_dims=cfg.num_spatial_dims,
            domain_extent=cfg.domain_extent,
            num_points=cfg.num_points,
            order=cfg.order,
        )
        ic_gen = ex.ic.RandomTruncatedFourierSeries(
            num_spatial_dims=cfg.num_spatial_dims,
            cutoff=cfg.ic_cutoff,
            max_one=cfg.ic_max_one,
        )

        #Generate data
        key = jax.random.PRNGKey(s)
        sources = ex.build_ic_set(
            ic_gen,
            num_points=cfg.num_points,
            num_samples=n,
            key=key,
        )  #(N, C, *spatial)

        solutions = jax.vmap(solver)(sources)  #(N, C, *spatial)

        #Convert to PyTorch
        sources_pt = jax_to_torch(sources, device=cfg.torch_device)
        solutions_pt = jax_to_torch(solutions, device=cfg.torch_device)

        data = {
            'source': sources_pt,
            'solution': solutions_pt,
        }

        #Metadata
        stats = {
            'source': compute_normalization_stats(sources_pt),
            'solution': compute_normalization_stats(solutions_pt),
        }

        metadata = {
            'stats': stats,
            'config': cfg.to_dict(),
        }

        print(
            f"Generated Poisson {cfg.num_spatial_dims}D dataset: "
            f"{n} samples, {cfg.num_points}{'x' + str(cfg.num_points) if cfg.num_spatial_dims >= 2 else ''} grid"
        )

        return PDEDataset(data, problem=problem, metadata=metadata)
