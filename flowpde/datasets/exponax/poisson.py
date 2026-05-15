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
from typing import Literal, Optional

import jax
import exponax as ex

from .base import GenerationConfig, PDEDataset
from .generator import ExponaxDatasetGenerator, FourierFieldConfig, sample_fourier_fields


@dataclass
class PoissonConfig(GenerationConfig):
    """
    Configuration specific to the Poisson equation.

    Attributes:
        order: Order of the Poisson operator (default 2 -> Laplacian).
        ic_cutoff_min: Minimum Fourier cutoff sampled per source function.
            Lower values produce smoother, low-frequency source terms.
        ic_cutoff_max: Maximum Fourier cutoff sampled per source function.
            Higher values allow sharp, high-frequency source terms.  IC
            generation always uses this as the upper band limit; each
            sample is then masked down to its individually drawn cutoff
            in spectral space, producing a mixture of smoothness levels.
        ic_max_one: Whether to normalize ICs so max absolute value is 1.
            Disabled by default; use amplitude_min/amplitude_max instead
            to preserve per-sample amplitude variability.
        amplitude_min: Lower bound of the uniform per-sample amplitude
            multiplier applied after IC generation.
        amplitude_max: Upper bound of the uniform per-sample amplitude
            multiplier.
        gaussian_bump_prob: Probability that each source function is drawn
            from a Gaussian-bump generator instead of the truncated Fourier
            series.  0.0 = always Fourier; 1.0 = always bumps; 0.5 (default)
            = equal mixture.  Only applied to single-channel fields.
        gaussian_bump_max_bumps: Maximum number of Gaussian bumps per source
            function when using the bump generator (default: 5).
    """
    order: int = 2
    ic_cutoff_min: int = 0
    ic_cutoff_max: int = 2
    ic_max_one: bool = False
    amplitude_min: float = 0.1
    amplitude_max: float = 5.0
    gaussian_bump_prob: float = 0.5
    gaussian_bump_max_bumps: int = 5


class PoissonGenerator(ExponaxDatasetGenerator):
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

        solver = ex.poisson.Poisson(
            num_spatial_dims=cfg.num_spatial_dims,
            domain_extent=cfg.domain_extent,
            num_points=cfg.num_points,
            order=cfg.order,
        )

        sources = sample_fourier_fields(
            jax.random.PRNGKey(s),
            n=n,
            cfg=cfg,
            field=FourierFieldConfig(
                cutoff_min=cfg.ic_cutoff_min,
                cutoff_max=cfg.ic_cutoff_max,
                max_one=cfg.ic_max_one,
                amplitude_min=1.0,
                amplitude_max=1.0,
                gaussian_bump_prob=0.0,
                gaussian_bump_max_bumps=cfg.gaussian_bump_max_bumps,
                random_cutoff=False,
                normalize=False,
            ),
        )
        solutions = jax.vmap(solver)(sources)

        data = self.to_torch_data({
            'source': sources,
            'solution': solutions,
        })
        self.apply_inverse_observations(
            data,
            observation_key='solution',
            problem=problem,
            n=n,
            seed=s,
        )

        self.print_summary("Poisson", n)
        return self.wrap_dataset(data, problem=problem)
