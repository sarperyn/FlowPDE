"""
Burgers Equation Data Generation
==================================

Generates initial-condition → final-state pairs (and optionally full
trajectories) for the Burgers equation using Exponax's ETDRK stepper
on periodic domains.

The Burgers equation:

.. math::

    \frac{\partial u}{\partial t} + u \cdot \nabla u = \nu \nabla^2 u

Random initial conditions are created via truncated Fourier series,
then evolved forward in time using the spectral stepper.

Example::

    generator = BurgersGenerator(
        num_spatial_dims=1,
        num_points=160,
        diffusivity_min=1e-4,
        diffusivity_max=1e-2,
        dt=0.001,
        num_steps=50,
    )
    dataset = generator.generate(num_samples=500, seed=0)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import jax
import jax.numpy as jnp
import exponax as ex

from .base import GenerationConfig, PDEDataset
from .converters import jax_to_torch, compute_normalization_stats, apply_spectral_cutoff, gaussian_bump_ic_batch


@dataclass
class BurgersConfig(GenerationConfig):
    """
    Configuration specific to the Burgers equation.

    Attributes:
        dt: Time-step size for the ETDRK stepper.
        num_steps: Number of time steps to advance.
        diffusivity_min: Lower bound of the log-uniform per-sample viscosity
            ν.  Each sample gets an independently drawn ν from
            LogUniform(diffusivity_min, diffusivity_max), so the dataset
            spans multiple physical regimes (smooth ↔ shock-dominated).
        diffusivity_max: Upper bound of the log-uniform per-sample viscosity.
        convection_scale: Scaling of the nonlinear convection term.
        single_channel: If True, use a single-channel formulation
            (scalar Burgers) regardless of spatial dimension.
        ic_cutoff_min: Minimum Fourier cutoff sampled per IC.
            Lower values produce smoother, low-frequency initial conditions.
        ic_cutoff_max: Maximum Fourier cutoff sampled per IC.
            Higher values allow sharp, high-frequency initial conditions.  IC
            generation always uses this as the upper band limit; each
            sample is then masked down to its individually drawn cutoff
            in spectral space, producing a mixture of smoothness levels.
        ic_max_one: Normalize ICs so max |u| = 1. Disabled by default;
            use amplitude_min/amplitude_max instead to preserve
            per-sample amplitude variability.
        amplitude_min: Lower bound of the uniform per-sample amplitude
            multiplier applied after IC generation.
        amplitude_max: Upper bound of the uniform per-sample amplitude
            multiplier.
        gaussian_bump_prob: Probability that each IC is drawn from a
            Gaussian-bump generator instead of the truncated Fourier series.
            0.0 = always Fourier; 1.0 = always bumps; 0.5 (default) = equal
            mixture.  Only applied to single-channel fields.
        gaussian_bump_max_bumps: Maximum number of Gaussian bumps per IC
            when using the bump generator (default: 5).
        store_trajectory: If True, keep the full rollout trajectory
            in the dataset (key ``'trajectory'``).
    """
    num_spatial_dims: int = 1
    domain_extent: float = 1.0
    dt: float = 0.001
    num_steps: int = 400
    diffusivity_min: float = 1e-4
    diffusivity_max: float = 1e-2
    convection_scale: float = 1.0
    single_channel: bool = False
    ic_cutoff_min: int = 3
    ic_cutoff_max: int = 20
    ic_max_one: bool = False
    amplitude_min: float = 0.1
    amplitude_max: float = 5.0
    gaussian_bump_prob: float = 0.5
    gaussian_bump_max_bumps: int = 5
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

        # --- IC generation ---
        ic_gen = ex.ic.RandomTruncatedFourierSeries(
            num_spatial_dims=cfg.num_spatial_dims,
            cutoff=cfg.ic_cutoff_max,
            max_one=cfg.ic_max_one,
        )

        key = jax.random.PRNGKey(s)
        key, amp_key, cutoff_key, nu_key, bump_key, mix_key = jax.random.split(key, 6)

        # Per-sample log-uniform diffusivity — spans smooth (high ν) to
        # shock-dominated (low ν) regimes in a single dataset.
        log_nus = jax.random.uniform(
            nu_key,
            shape=(n,),
            minval=jnp.log(cfg.diffusivity_min),
            maxval=jnp.log(cfg.diffusivity_max),
        )
        nus = jnp.exp(log_nus)
        ics = ex.build_ic_set(
            ic_gen,
            num_points=cfg.num_points,
            num_samples=n,
            key=key,
        )  # (N, C, *spatial)

        # Per-sample random spectral cutoff — each IC gets an independently
        # drawn cutoff from [ic_cutoff_min, ic_cutoff_max], yielding a mixture
        # of smooth (low cutoff) and sharp (high cutoff) initial conditions.
        cutoffs = jax.random.randint(
            cutoff_key,
            shape=(n,),
            minval=cfg.ic_cutoff_min,
            maxval=cfg.ic_cutoff_max + 1,
        )
        num_dims = cfg.num_spatial_dims
        ics = jax.vmap(
            lambda f, c: apply_spectral_cutoff(f, c, num_dims)
        )(ics, cutoffs)

        # Mix Fourier and Gaussian bump ICs (single-channel only).
        if ics.shape[1] == 1:
            ics_gaussian = gaussian_bump_ic_batch(
                bump_key,
                n=n,
                num_points=cfg.num_points,
                domain_extent=cfg.domain_extent,
                num_spatial_dims=cfg.num_spatial_dims,
                max_bumps=cfg.gaussian_bump_max_bumps,
            )  # (N, 1, *spatial)
            mix_mask = jax.random.uniform(mix_key, (n,)) < cfg.gaussian_bump_prob
            mask = mix_mask.reshape((n,) + (1,) * (ics.ndim - 1))
            ics = jnp.where(mask, ics_gaussian, ics)

        # Per-sample amplitude scaling — restores amplitude variability
        # that ic_max_one=True would suppress.
        amp_shape = (n,) + (1,) * (ics.ndim - 1)
        amplitudes = jax.random.uniform(
            amp_key,
            shape=amp_shape,
            minval=cfg.amplitude_min,
            maxval=cfg.amplitude_max,
        )
        ics = ics * amplitudes

        # --- Time integration (per-sample ν) ---
        # The stepper is constructed inside the vmapped function so that JAX
        # traces through the ETDRK coefficient computation with each sample's
        # individual nu value.
        def step_one(ic, nu):
            stepper = ex.stepper.Burgers(
                num_spatial_dims=cfg.num_spatial_dims,
                domain_extent=cfg.domain_extent,
                num_points=cfg.num_points,
                dt=cfg.dt,
                diffusivity=nu,
                convection_scale=cfg.convection_scale,
                single_channel=cfg.single_channel,
            )
            return ex.RepeatedStepper(stepper, cfg.num_steps)(ic)

        def rollout_one(ic, nu):
            stepper = ex.stepper.Burgers(
                num_spatial_dims=cfg.num_spatial_dims,
                domain_extent=cfg.domain_extent,
                num_points=cfg.num_points,
                dt=cfg.dt,
                diffusivity=nu,
                convection_scale=cfg.convection_scale,
                single_channel=cfg.single_channel,
            )
            return ex.rollout(stepper, cfg.num_steps, include_init=True)(ic)

        if cfg.store_trajectory:
            trajectories = jax.vmap(rollout_one)(ics, nus)  # (N, T+1, C, *spatial)
            finals = trajectories[:, -1]
        else:
            finals = jax.vmap(step_one)(ics, nus)  # (N, C, *spatial)
            trajectories = None

        # --- Convert to PyTorch ---
        ics_pt = jax_to_torch(ics, device=cfg.torch_device)
        finals_pt = jax_to_torch(finals, device=cfg.torch_device)
        nus_pt = jax_to_torch(nus, device=cfg.torch_device)  # (N,)

        # Observation noise for inverse problems — corrupts the final-state
        # field (the model input in the inverse direction) with additive
        # Gaussian noise.
        if problem == 'inverse' and cfg.obs_noise_std > 0.0:
            finals_pt = finals_pt + cfg.obs_noise_std * torch.randn_like(finals_pt)

        data = {
            'initial': ics_pt,
            'final': finals_pt,
            'diffusivity': nus_pt,
        }

        if trajectories is not None:
            data['trajectory'] = jax_to_torch(trajectories, device=cfg.torch_device)

        # --- Metadata ---
        stats = {
            'initial': compute_normalization_stats(ics_pt),
            'final': compute_normalization_stats(finals_pt),
            'diffusivity': {
                'min': float(nus_pt.min()),
                'max': float(nus_pt.max()),
                'mean': float(nus_pt.mean()),
            },
        }

        metadata = {
            'stats': stats,
            'config': cfg.to_dict(),
        }

        spatial_str = 'x'.join([str(cfg.num_points)] * cfg.num_spatial_dims)
        print(
            f"Generated Burgers {cfg.num_spatial_dims}D dataset: "
            f"{n} samples, {spatial_str} grid, "
            f"{cfg.num_steps} steps (dt={cfg.dt}), "
            f"ν ~ LogUniform({cfg.diffusivity_min:.0e}, {cfg.diffusivity_max:.0e})"
        )

        return PDEDataset(data, problem=problem, metadata=metadata)
