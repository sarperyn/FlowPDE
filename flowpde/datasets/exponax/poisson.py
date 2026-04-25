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

import torch
import jax
import jax.numpy as jnp
import exponax as ex

from .base import GenerationConfig, PDEDataset
from .converters import jax_to_torch, compute_normalization_stats, apply_spectral_cutoff, gaussian_bump_ic_batch


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
            cutoff=cfg.ic_cutoff_max,
            max_one=cfg.ic_max_one,
        )

        #Generate data
        key = jax.random.PRNGKey(s)
        key, amp_key, cutoff_key, bump_key, mix_key = jax.random.split(key, 5)
        sources = ex.build_ic_set(
            ic_gen,
            num_points=cfg.num_points,
            num_samples=n,
            key=key,
        )  #(N, C, *spatial)

        # Per-sample random spectral cutoff — each source gets an independently
        # drawn cutoff from [ic_cutoff_min, ic_cutoff_max], yielding a mixture
        # of smooth (low cutoff) and sharp (high cutoff) source functions.
        # cutoffs = jax.random.randint(
        #     cutoff_key,
        #     shape=(n,),
        #     minval=cfg.ic_cutoff_min,
        #     maxval=cfg.ic_cutoff_max + 1,
        # )
        # num_dims = cfg.num_spatial_dims
        # sources = jax.vmap(
        #     lambda f, c: apply_spectral_cutoff(f, c, num_dims)
        # )(sources, cutoffs)

        # Mix Fourier and Gaussian bump source functions (single-channel only).
        # if sources.shape[1] == 1:
        #     sources_gaussian = gaussian_bump_ic_batch(
        #         bump_key,
        #         n=n,
        #         num_points=cfg.num_points,
        #         domain_extent=cfg.domain_extent,
        #         num_spatial_dims=cfg.num_spatial_dims,
        #         max_bumps=cfg.gaussian_bump_max_bumps,
        #     )  # (N, 1, *spatial)
        #     mix_mask = jax.random.uniform(mix_key, (n,)) < cfg.gaussian_bump_prob
        #     mask = mix_mask.reshape((n,) + (1,) * (sources.ndim - 1))
        #     sources = jnp.where(mask, sources_gaussian, sources)

        # # Per-sample amplitude scaling — restores amplitude variability
        # # that ic_max_one=True would suppress.
        # amp_shape = (n,) + (1,) * (sources.ndim - 1)
        # amplitudes = jax.random.uniform(
        #     amp_key,
        #     shape=amp_shape,
        #     minval=cfg.amplitude_min,
        #     maxval=cfg.amplitude_max,
        # )
        # sources = sources * amplitudes

        solutions = jax.vmap(solver)(sources)  #(N, C, *spatial)

        #Convert to PyTorch
        sources_pt = jax_to_torch(sources, device=cfg.torch_device)
        solutions_pt = jax_to_torch(solutions, device=cfg.torch_device)

        if problem == 'inverse':
            # Additive observation noise
            if cfg.obs_noise_std > 0.0:
                solutions_pt = solutions_pt + cfg.obs_noise_std * torch.randn_like(solutions_pt)

            # Partial observations — random Bernoulli mask over spatial grid.
            # Mask shape: (N, 1, *spatial) so it broadcasts over all channels.
            # Stored in data so __getitem__ can return it alongside input/target.
            if cfg.obs_mask_fraction < 1.0:
                spatial_shape = (cfg.num_points,) * cfg.num_spatial_dims
                mask_gen = torch.Generator().manual_seed(s)
                obs_mask = (
                    torch.rand((n, 1) + spatial_shape, generator=mask_gen)
                    < cfg.obs_mask_fraction
                ).float().to(device=cfg.torch_device)
                solutions_pt = solutions_pt * obs_mask
            else:
                obs_mask = None
        else:
            obs_mask = None

        data = {
            'source': sources_pt,
            'solution': solutions_pt,
        }
        if obs_mask is not None:
            data['obs_mask'] = obs_mask

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
