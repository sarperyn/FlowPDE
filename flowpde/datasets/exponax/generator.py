"""Reusable building blocks for Exponax-backed dataset generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import exponax as ex
import jax
import jax.numpy as jnp
import torch

from .base import GenerationConfig, PDEDataset
from .converters import (
    apply_spectral_cutoff,
    compute_normalization_stats,
    gaussian_bump_ic_batch,
    jax_to_torch,
)


@dataclass
class FourierFieldConfig:
    """Settings for the default Exponax random-field sampler."""

    cutoff_min: int = 0
    cutoff_max: int = 8
    max_one: bool = False
    amplitude_min: float = 1.0
    amplitude_max: float = 1.0
    normalize: bool = False
    random_cutoff: bool = True
    gaussian_bump_prob: float = 0.0
    gaussian_bump_max_bumps: int = 5


def log_uniform(key, n: int, min_value: float, max_value: float):
    """Draw ``n`` positive scalars from a log-uniform distribution."""
    values = jax.random.uniform(
        key,
        shape=(n,),
        minval=jnp.log(min_value),
        maxval=jnp.log(max_value),
    )
    return jnp.exp(values)


def normalize_per_sample(fields):
    """Normalize every sample to max absolute value 1."""

    def normalize(field):
        max_abs = jnp.abs(field).max()
        return field / jnp.maximum(max_abs, 1e-6)

    return jax.vmap(normalize)(fields)


def sample_fourier_fields(
    key,
    *,
    n: int,
    cfg: GenerationConfig,
    field: FourierFieldConfig,
):
    """Generate a batch of Exponax random fields with common controls."""
    ic_gen = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=cfg.num_spatial_dims,
        cutoff=field.cutoff_max,
        max_one=field.max_one,
    )

    key, amp_key, cutoff_key, bump_key, mix_key = jax.random.split(key, 5)
    fields = ex.build_ic_set(
        ic_gen,
        num_points=cfg.num_points,
        num_samples=n,
        key=key,
    )

    if field.random_cutoff:
        cutoffs = jax.random.randint(
            cutoff_key,
            shape=(n,),
            minval=field.cutoff_min,
            maxval=field.cutoff_max + 1,
        )
        fields = jax.vmap(
            lambda f, c: apply_spectral_cutoff(f, c, cfg.num_spatial_dims)
        )(fields, cutoffs)

    if field.gaussian_bump_prob > 0.0 and fields.shape[1] == 1:
        bump_fields = gaussian_bump_ic_batch(
            bump_key,
            n=n,
            num_points=cfg.num_points,
            domain_extent=cfg.domain_extent,
            num_spatial_dims=cfg.num_spatial_dims,
            max_bumps=field.gaussian_bump_max_bumps,
        )
        mix_mask = jax.random.uniform(mix_key, (n,)) < field.gaussian_bump_prob
        mask = mix_mask.reshape((n,) + (1,) * (fields.ndim - 1))
        fields = jnp.where(mask, bump_fields, fields)

    if field.normalize:
        fields = normalize_per_sample(fields)

    if field.amplitude_min != 1.0 or field.amplitude_max != 1.0:
        amp_shape = (n,) + (1,) * (fields.ndim - 1)
        amplitudes = jax.random.uniform(
            amp_key,
            shape=amp_shape,
            minval=field.amplitude_min,
            maxval=field.amplitude_max,
        )
        fields = fields * amplitudes

    return fields


class ExponaxDatasetGenerator:
    """Base class for concise Exponax problem generators."""

    config: GenerationConfig
    dataset_cls = PDEDataset

    def __init__(self, config: Optional[GenerationConfig] = None, **kwargs):
        if config is None:
            config = self.config_cls(**kwargs)
        self.config = config

    def resolve_run(self, num_samples: Optional[int], seed: Optional[int]):
        cfg = self.config
        n = num_samples or cfg.num_samples
        s = seed if seed is not None else cfg.seed
        return cfg, n, s

    def to_torch_data(self, arrays: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            key: jax_to_torch(value, device=self.config.torch_device)
            for key, value in arrays.items()
            if value is not None
        }

    def apply_inverse_observations(
        self,
        data: Dict[str, torch.Tensor],
        *,
        observation_key: str,
        problem: Literal["forward", "inverse"],
        n: int,
        seed: int,
    ) -> None:
        cfg = self.config
        if problem != "inverse":
            return

        if cfg.obs_noise_std > 0.0:
            data[observation_key] = (
                data[observation_key]
                + cfg.obs_noise_std * torch.randn_like(data[observation_key])
            )

        if cfg.obs_mask_fraction < 1.0:
            spatial_shape = (cfg.num_points,) * cfg.num_spatial_dims
            mask_gen = torch.Generator().manual_seed(seed)
            obs_mask = (
                torch.rand((n, 1) + spatial_shape, generator=mask_gen)
                < cfg.obs_mask_fraction
            ).float().to(device=cfg.torch_device)
            data[observation_key] = data[observation_key] * obs_mask
            data["obs_mask"] = obs_mask

    def wrap_dataset(
        self,
        data: Dict[str, torch.Tensor],
        *,
        problem: Literal["forward", "inverse"],
        extra_stats: Optional[Dict[str, Dict[str, float]]] = None,
        **dataset_kwargs,
    ):
        stats = {
            key: compute_normalization_stats(value)
            for key, value in data.items()
            if torch.is_tensor(value) and key != "obs_mask"
        }
        if extra_stats:
            stats.update(extra_stats)

        metadata = {
            "stats": stats,
            "config": self.config.to_dict(),
        }
        return self.dataset_cls(
            data,
            problem=problem,
            metadata=metadata,
            **dataset_kwargs,
        )

    def print_summary(self, name: str, n: int, details: str = "") -> None:
        cfg = self.config
        spatial = "x".join([str(cfg.num_points)] * cfg.num_spatial_dims)
        suffix = f", {details}" if details else ""
        print(
            f"Generated {name} {cfg.num_spatial_dims}D dataset: "
            f"{n} samples, {spatial} grid{suffix}"
        )
