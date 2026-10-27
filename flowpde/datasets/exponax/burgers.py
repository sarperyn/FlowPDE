"""Burgers datasets from Exponax random Fourier fields and its native stepper."""

from dataclasses import dataclass
from typing import Literal, Optional

import exponax as ex
import jax

from .base import GenerationConfig, PDEDataset
from .generator import ExponaxDatasetGenerator


@dataclass
class BurgersConfig(GenerationConfig):
    """Native Burgers and RandomTruncatedFourierSeries options.

    ``diffusivity`` is fixed for the whole dataset. ``ic_cutoff`` and
    ``ic_max_one`` are passed directly to Exponax's field generator.
    ``store_trajectory`` retains the initial state and every subsequent step.
    """

    num_spatial_dims: int = 1
    dt: float = 0.1
    num_steps: int = 400
    diffusivity: float = 0.1
    convection_scale: float = 1.0
    single_channel: bool = False
    conservative: bool = False
    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0
    ic_cutoff: int = 5
    ic_max_one: bool = False
    store_trajectory: bool = False


class BurgersGenerator(ExponaxDatasetGenerator):
    """Wrap Exponax's Burgers solver as a PyTorch dataset.

    Initial fields come from ``ex.ic.RandomTruncatedFourierSeries``; vector
    Burgers uses ``ex.ic.RandomMultiChannelICGenerator``. No per-sample
    viscosity, custom fields, observation noise, or masking is applied.
    ``problem='inverse'`` simply swaps the clean initial/final pair.
    """

    config_cls = BurgersConfig

    def generate(
        self,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        problem: Literal['forward', 'inverse'] = 'forward',
    ) -> PDEDataset:
        """Generate initial/final pairs, optionally retaining full trajectories."""
        cfg, n, s = self.resolve_run(num_samples, seed)
        self.validate_problem(problem)
        if cfg.num_steps < 1:
            raise ValueError("num_steps must be positive")

        ic_generator = ex.ic.RandomTruncatedFourierSeries(
            cfg.num_spatial_dims, cutoff=cfg.ic_cutoff, max_one=cfg.ic_max_one,
        )
        if not cfg.single_channel:
            ic_generator = ex.ic.RandomMultiChannelICGenerator(
                (ic_generator,) * cfg.num_spatial_dims,
            )
        initials = ex.build_ic_set(
            ic_generator, num_points=cfg.num_points, num_samples=n,
            key=jax.random.PRNGKey(s),
        )
        stepper = ex.stepper.Burgers(
            num_spatial_dims=cfg.num_spatial_dims,
            domain_extent=cfg.domain_extent,
            num_points=cfg.num_points,
            dt=cfg.dt,
            diffusivity=cfg.diffusivity,
            convection_scale=cfg.convection_scale,
            single_channel=cfg.single_channel,
            conservative=cfg.conservative,
            order=cfg.order,
            dealiasing_fraction=cfg.dealiasing_fraction,
            num_circle_points=cfg.num_circle_points,
            circle_radius=cfg.circle_radius,
        )
        if cfg.store_trajectory:
            trajectories = jax.vmap(
                ex.rollout(stepper, cfg.num_steps, include_init=True)
            )(initials)
            finals = trajectories[:, -1]
        else:
            trajectories = None
            finals = jax.vmap(ex.RepeatedStepper(stepper, cfg.num_steps))(initials)

        data = self.to_torch_data({
            'initial': initials, 'final': finals, 'trajectory': trajectories,
        })
        self.print_summary("Burgers", n, details=f"{cfg.num_steps} steps (dt={cfg.dt}), nu={cfg.diffusivity}")
        return self.wrap_dataset(data, problem=problem)
