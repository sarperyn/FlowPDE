r"""
Burgers Equation Data Generation
==================================

Generates initial-condition → final-state pairs (and optionally full
trajectories) for the Burgers equation using Exponax's ETDRK stepper
on periodic domains.

The Burgers equation:

.. math::

    \frac{\partial u}{\partial t} + u \cdot \nabla u = \nu \nabla^2 u

Smooth sine/cosine initial conditions are evolved forward in time using the
spectral stepper.

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

from dataclasses import dataclass
from typing import Optional, Literal

import jax
import exponax as ex

from .base import GenerationConfig, PDEDataset
from .generator import (
    ExponaxDatasetGenerator,
    log_uniform,
)
from .utilities import sample_sine_fields


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
        ic_num_terms: Number of sine/cosine terms per initial condition.
        ic_max_mode: Largest integer wavenumber sampled per dimension.
        store_trajectory: If True, keep the full rollout trajectory
            in the dataset (key ``'trajectory'``).
    """
    num_spatial_dims: int = 1
    domain_extent: float = 1.0
    dt: float = 0.001
    num_steps: int = 400
    diffusivity_min: float = 1e-2
    diffusivity_max: float = 1e-2
    convection_scale: float = 1.0
    single_channel: bool = False
    ic_num_terms: int = 3
    ic_max_mode: int = 4
    store_trajectory: bool = False


class BurgersGenerator(ExponaxDatasetGenerator):
    """
    Generate Burgers equation datasets using Exponax.

    Workflow:
        1. Create simple smooth sine/cosine initial conditions
        2. Step forward using ``exponax.stepper.Burgers`` (optionally via
           ``exponax.rollout`` for trajectories)
        3. Convert JAX arrays → PyTorch tensors
        4. Wrap in a ``PDEDataset``

    Args:
        config: A ``BurgersConfig`` instance.  Keyword arguments are
                forwarded to ``BurgersConfig`` if *config* is None.
    """

    config_cls = BurgersConfig

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
        cfg, n, s = self.resolve_run(num_samples, seed)
        self.validate_problem(problem)

        key = jax.random.PRNGKey(s)
        field_key, nu_key = jax.random.split(key, 2)

        nus = log_uniform(
            nu_key,
            n=n,
            min_value=cfg.diffusivity_min,
            max_value=cfg.diffusivity_max,
        )
        ics = sample_sine_fields(
            field_key,
            n=n,
            num_spatial_dims=cfg.num_spatial_dims,
            num_points=cfg.num_points,
            domain_extent=cfg.domain_extent,
            num_terms=cfg.ic_num_terms,
            max_mode=cfg.ic_max_mode,
        )

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

        data = self.to_torch_data({
            'initial': ics,
            'final': finals,
            'diffusivity': nus,
            'trajectory': trajectories,
        })
        self.apply_observation_augmentation(
            data,
            observation_key='final',
            problem=problem,
            n=n,
            seed=s,
        )

        self.print_summary(
            "Burgers",
            n,
            details=(
                f"{cfg.num_steps} steps (dt={cfg.dt}), "
                f"nu ~ LogUniform({cfg.diffusivity_min:.0e}, "
                f"{cfg.diffusivity_max:.0e})"
            ),
        )
        return self.wrap_dataset(data, problem=problem)
