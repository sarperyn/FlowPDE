"""
Burgers Equation Data Generation
==================================

Generates initial-condition → final-state pairs (and optionally full
trajectories) for the Burgers equation using Exponax's ETDRK stepper
on periodic domains.

The Burgers equation:

    ∂u/∂t + u · ∇u = ν ∇²u

Random initial conditions are created via truncated Fourier series,
then evolved forward in time using the spectral stepper.

Example::

    generator = BurgersGenerator(
        num_spatial_dims=1,
        num_points=160,
        diffusivity=0.0003,
        dt=0.001,
        num_steps=50,
    )
    dataset = generator.generate(num_samples=500, seed=0)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

import jax
import jax.numpy as jnp
import exponax as ex

from .base import GenerationConfig, PDEDataset
from .converters import jax_to_torch, compute_normalization_stats


@dataclass
class BurgersConfig(GenerationConfig):
    """
    Configuration specific to the Burgers equation.

    Attributes:
        dt: Time-step size for the ETDRK stepper.
        num_steps: Number of time steps to advance.
        diffusivity: Viscosity / diffusion coefficient ν.
        convection_scale: Scaling of the nonlinear convection term.
        single_channel: If True, use a single-channel formulation
            (scalar Burgers) regardless of spatial dimension.
        ic_cutoff: Fourier cutoff for random IC generation.
        ic_max_one: Normalize ICs so max |u| = 1.
        store_trajectory: If True, keep the full rollout trajectory
            in the dataset (key ``'trajectory'``).
    """
    num_spatial_dims: int = 1
    domain_extent: float = 1.0
    dt: float = 0.001
    num_steps: int = 50
    diffusivity: float = 0.0003
    convection_scale: float = 1.0
    single_channel: bool = False
    ic_cutoff: int = 5
    ic_max_one: bool = True
    store_trajectory: bool = False


class BurgersGenerator:
    """
    Generate Burgers equation datasets using Exponax.

    Workflow:
        1. Create random ICs using ``exponax.ic.RandomTruncatedFourierSeries``
        2. Step forward using ``exponax.stepper.Burgers`` (optionally via
           ``exponax.rollout`` for trajectories)
        3. Convert JAX arrays → PyTorch tensors
        4. Wrap in a ``PDEDataset``

    Args:
        config: A ``BurgersConfig`` instance.  Keyword arguments are
                forwarded to ``BurgersConfig`` if *config* is None.
    """

    def __init__(self, config: Optional[BurgersConfig] = None, **kwargs):
        if config is None:
            config = BurgersConfig(**kwargs)
        self.config = config

    def generate(
        self,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        problem: Literal['forward', 'inverse'] = 'forward',
    ) -> PDEDataset:
        """
        Generate a Burgers dataset.

        Args:
            num_samples: Override ``config.num_samples``.
            seed: Override ``config.seed``.
            problem: ``'forward'`` (IC→final) or ``'inverse'``.

        Returns:
            A ``PDEDataset`` with keys ``'initial'`` and ``'final'``
            (and optionally ``'trajectory'``).
        """
        cfg = self.config
        n = num_samples or cfg.num_samples
        s = seed if seed is not None else cfg.seed

        # --- Exponax stepper ---
        stepper = ex.stepper.Burgers(
            num_spatial_dims=cfg.num_spatial_dims,
            domain_extent=cfg.domain_extent,
            num_points=cfg.num_points,
            dt=cfg.dt,
            diffusivity=cfg.diffusivity,
            convection_scale=cfg.convection_scale,
            single_channel=cfg.single_channel,
        )

        # --- IC generation ---
        ic_gen = ex.ic.RandomTruncatedFourierSeries(
            num_spatial_dims=cfg.num_spatial_dims,
            cutoff=cfg.ic_cutoff,
            max_one=cfg.ic_max_one,
        )

        key = jax.random.PRNGKey(s)
        ics = ex.build_ic_set(
            ic_gen,
            num_points=cfg.num_points,
            num_samples=n,
            key=key,
        )  # (N, C, *spatial)

        # --- Time integration ---
        if cfg.store_trajectory:
            # Full rollout: returns (N, T, C, *spatial)
            rollout_fn = ex.rollout(stepper, cfg.num_steps, include_init=True)
            trajectories = jax.vmap(rollout_fn)(ics)  # (N, T+1, C, *spatial)
            finals = trajectories[:, -1]
        else:
            # Just apply the stepper num_steps times using RepeatedStepper
            repeated = ex.RepeatedStepper(stepper, cfg.num_steps)
            finals = jax.vmap(repeated)(ics)  # (N, C, *spatial)
            trajectories = None

        # --- Convert to PyTorch ---
        ics_pt = jax_to_torch(ics, device=cfg.torch_device)
        finals_pt = jax_to_torch(finals, device=cfg.torch_device)

        data = {
            'initial': ics_pt,
            'final': finals_pt,
        }

        if trajectories is not None:
            data['trajectory'] = jax_to_torch(trajectories, device=cfg.torch_device)

        # --- Metadata ---
        stats = {
            'initial': compute_normalization_stats(ics_pt),
            'final': compute_normalization_stats(finals_pt),
        }

        metadata = {
            'stats': stats,
            'config': cfg.to_dict(),
        }

        spatial_str = 'x'.join([str(cfg.num_points)] * cfg.num_spatial_dims)
        print(
            f"Generated Burgers {cfg.num_spatial_dims}D dataset: "
            f"{n} samples, {spatial_str} grid, "
            f"{cfg.num_steps} steps (dt={cfg.dt}), ν={cfg.diffusivity}"
        )

        return PDEDataset(data, problem=problem, metadata=metadata)
