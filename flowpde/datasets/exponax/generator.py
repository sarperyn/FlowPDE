"""Reusable building blocks for Exponax-backed dataset generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import exponax as ex
import jax
import jax.numpy as jnp
import torch

from .base import GenerationConfig, PDEDataset
from .utilities import compute_normalization_stats, jax_to_torch

logger = logging.getLogger(__name__)


@dataclass
class FourierFieldConfig:
    """Settings for smooth fixed-cutoff Exponax random fields."""

    cutoff: int = 8
    max_one: bool = False
    amplitude_min: float = 1.0
    amplitude_max: float = 1.0
    normalize: bool = False


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
        cutoff=field.cutoff,
        max_one=field.max_one,
    )

    key, amp_key = jax.random.split(key, 2)
    fields = ex.build_ic_set(
        ic_gen,
        num_points=cfg.num_points,
        num_samples=n,
        key=key,
    )

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
        n = cfg.num_samples if num_samples is None else num_samples
        if n < 1:
            raise ValueError("num_samples must be positive")
        s = seed if seed is not None else cfg.seed
        return cfg, n, s

    def validate_problem(self, problem: str) -> None:
        if problem not in {"forward", "inverse"}:
            raise ValueError("problem must be 'forward' or 'inverse'")

    def to_torch_data(self, arrays: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            key: jax_to_torch(value, device=self.config.torch_device)
            for key, value in arrays.items()
            if value is not None
        }

    def wrap_dataset(
        self,
        data: Dict[str, torch.Tensor],
        *,
        problem: Literal["forward", "inverse"],
        extra_stats: Optional[Dict[str, Dict[str, float]]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
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
            "grid": {"spacing": self.config.domain_extent / self.config.num_points,
                     "periodic": True, "endpoint_included": False},
        }
        # Generation diagnostics (e.g. solver residuals) belong in metadata,
        # not in `data`: they are not fields and must not be normalized.
        if extra_metadata:
            metadata.update(extra_metadata)
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
        logger.info(
            "Generated %s %dD dataset: %d samples, %s grid%s",
            name, cfg.num_spatial_dims, n, spatial, suffix,
        )
