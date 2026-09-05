"""
Exponax Dataset Utilities
=========================

Utility helpers for Exponax-backed datasets, including JAX↔PyTorch
conversions and normalization stats.
"""

import numpy as np
import torch
from typing import Any, Union, Optional


DeviceType = Union[str, torch.device]


def jax_to_torch(
    jax_array: "Any",
    device: DeviceType = 'cpu',
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Convert a JAX array to a PyTorch tensor.

    Uses NumPy as an intermediate format for maximum compatibility
    across different JAX/PyTorch device configurations.

    Args:
        jax_array: JAX array to convert
        device: Target PyTorch device ('cpu', 'cuda', etc.)
        dtype: Target PyTorch dtype (default: torch.float32)

    Returns:
        PyTorch tensor on the specified device
    """
    np_array = np.array(jax_array, copy=True)
    tensor = torch.from_numpy(np_array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    tensor = tensor.to(device=device)
    return tensor


def compute_normalization_stats(tensor: torch.Tensor) -> dict:
    """
    Compute normalization statistics for a tensor.

    Args:
        tensor: Input tensor of any shape

    Returns:
        Dictionary with 'mean', 'std', 'min', and 'max' values.
    """
    return {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
    }


def sample_sine_fields(
    key,
    *,
    n: int,
    num_spatial_dims: int,
    num_points: int,
    domain_extent: float,
    num_terms: int,
    max_mode: int,
):
    """Generate simple smooth sine fields with random modes, weights, and phases."""
    if num_terms < 1:
        raise ValueError("num_terms must be at least 1")
    if max_mode < 1:
        raise ValueError("max_mode must be at least 1")

    import jax
    import jax.numpy as jnp

    coords = [
        jnp.linspace(0.0, domain_extent, num_points, endpoint=False)
        for _ in range(num_spatial_dims)
    ]
    grids = jnp.meshgrid(*coords, indexing='ij') if num_spatial_dims > 1 else coords
    stacked_grid = jnp.stack(grids, axis=0)

    _, mode_key, coeff_key, phase_key = jax.random.split(key, 4)
    modes = jax.random.randint(
        mode_key,
        shape=(n, num_terms, num_spatial_dims),
        minval=1,
        maxval=max_mode + 1,
    )
    coeffs = jax.random.normal(coeff_key, shape=(n, num_terms))
    phases = jax.random.uniform(
        phase_key,
        shape=(n, num_terms),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    def sample_one(sample_modes, sample_coeffs, sample_phases):
        field = jnp.zeros([num_points] * num_spatial_dims)
        for i in range(num_terms):
            angle = (
                2.0
                * jnp.pi
                * (
                    sample_modes[i].reshape(
                        (num_spatial_dims,) + (1,) * num_spatial_dims
                    )
                    * stacked_grid
                ).sum(axis=0)
                / domain_extent
                + sample_phases[i]
            )
            field = field + sample_coeffs[i] * jnp.sin(angle)
        field = field / jnp.sqrt(float(num_terms))
        return field[jnp.newaxis]

    return jax.vmap(sample_one)(modes, coeffs, phases)
