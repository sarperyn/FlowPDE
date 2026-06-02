"""
Exponax Dataset Utilities
=========================

Utility helpers for Exponax-backed datasets, including JAX↔PyTorch
conversions, normalization stats, spectral filtering, and IC synthesis.
"""

import numpy as np
import torch
from typing import Union, Optional


DeviceType = Union[str, torch.device]


def jax_to_torch(
    jax_array,
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


def torch_to_jax(tensor: torch.Tensor, dtype: Optional[str] = 'float32'):
    """
    Convert a PyTorch tensor to a JAX array.

    Args:
        tensor: PyTorch tensor to convert
        dtype: Target JAX dtype as string (default: 'float32')

    Returns:
        JAX array
    """
    import jax.numpy as jnp

    np_array = tensor.detach().cpu().numpy()
    jax_array = jnp.array(np_array)
    if dtype is not None:
        jax_array = jax_array.astype(dtype)
    return jax_array


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


def apply_spectral_cutoff(field, cutoff, num_spatial_dims: int):
    """
    Zero out Fourier modes above ``cutoff`` in all spatial dimensions.

    Generates a spatially filtered version of *field* by masking every
    mode with |k| > cutoff along each spatial axis.  Safe to use inside
    ``jax.vmap`` when *cutoff* is a per-sample traced scalar.

    The last spatial axis uses a real FFT (rfft), so its frequencies are
    non-negative (0 … n//2).  All preceding spatial axes use full FFTs
    with |k| = |round(freq * n)|.

    Args:
        field: JAX array of shape ``(C, *spatial)``.
        cutoff: Integer cutoff threshold; modes with |k| > cutoff are
                zeroed.  May be a JAX-traced scalar (e.g. from vmap).
        num_spatial_dims: Number of trailing spatial dimensions in
                *field* (1 for 1-D problems, 2 for 2-D, etc.).

    Returns:
        Filtered JAX array with the same dtype and shape as *field*.
    """
    import jax.numpy as jnp

    spatial_axes = tuple(range(1, 1 + num_spatial_dims))
    spatial_shape = field.shape[1:1 + num_spatial_dims]

    # Forward real FFT along all spatial axes simultaneously
    field_hat = jnp.fft.rfftn(field, axes=spatial_axes)

    # Build one boolean mask per spatial axis, then AND them together
    masks = []
    for i, (ax, n_pts) in enumerate(zip(spatial_axes, spatial_shape)):
        if i < num_spatial_dims - 1:
            # Full FFT axis: frequencies 0, 1, …, n//2, -(n//2-1), …, -1
            freqs = jnp.abs(jnp.fft.fftfreq(n_pts) * n_pts)
        else:
            # Real FFT axis: non-negative frequencies 0, 1, …, n//2
            freqs = jnp.arange(n_pts // 2 + 1, dtype=jnp.float32)

        reshape = [1] * field_hat.ndim
        reshape[ax] = freqs.shape[0]
        masks.append((freqs <= cutoff).reshape(reshape))

    mask = masks[0]
    for m in masks[1:]:
        mask = mask & m

    return jnp.fft.irfftn(field_hat * mask, s=spatial_shape, axes=spatial_axes).real


def gaussian_bump_ic_batch(
    key,
    n: int,
    num_points: int,
    domain_extent: float,
    num_spatial_dims: int,
    max_bumps: int = 5,
):
    """
    Generate a batch of random Gaussian-bump fields.

    Each sample is a superposition of ``k ~ Uniform{1, max_bumps}``
    randomly placed, randomly sized, and randomly signed isotropic Gaussian
    bumps on a uniform spatial grid.  The result is normalized to
    ``max|field| = 1`` so that the caller's amplitude scaling is the sole
    amplitude control.  Safe to use with ``jax.vmap``.

    Args:
        key: JAX PRNG key.
        n: Number of samples to generate.
        num_points: Grid resolution per spatial dimension.
        domain_extent: Physical size of the periodic domain.
        num_spatial_dims: Number of spatial dimensions (1, 2, or 3).
        max_bumps: Maximum number of Gaussian bumps per sample.

    Returns:
        JAX array of shape ``(n, 1, *([num_points] * num_spatial_dims))``.
    """
    import jax
    import jax.numpy as jnp

    def single(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)

        # Number of active bumps encoded as a static-shape mask
        n_bumps = jax.random.randint(k1, (), minval=1, maxval=max_bumps + 1)
        active = (jnp.arange(max_bumps) < n_bumps).astype(jnp.float32)  # (B,)

        # Random bump parameters
        centers = jax.random.uniform(
            k2,
            (max_bumps, num_spatial_dims),
            minval=0.1 * domain_extent,
            maxval=0.9 * domain_extent,
        )  # (B, D)
        widths = jax.random.uniform(
            k3,
            (max_bumps,),
            minval=0.05 * domain_extent,
            maxval=0.25 * domain_extent,
        )  # (B,)
        signs = jnp.where(
            jax.random.uniform(k4, (max_bumps,)) > 0.5, 1.0, -1.0
        )  # (B,)

        # Coordinate arrays for the spatial grid
        coords = [
            jnp.linspace(0.0, domain_extent, num_points, endpoint=False)
            for _ in range(num_spatial_dims)
        ]
        grids = jnp.meshgrid(*coords, indexing='ij') if num_spatial_dims > 1 else coords

        # Accumulate bumps (loop unrolled at trace time)
        field = jnp.zeros([num_points] * num_spatial_dims)
        for b in range(max_bumps):
            dist_sq = jnp.zeros_like(grids[0])
            for d in range(num_spatial_dims):
                dist_sq = dist_sq + ((grids[d] - centers[b, d]) / widths[b]) ** 2
            field = field + signs[b] * jnp.exp(-0.5 * dist_sq) * active[b]

        # Normalize to max|field| = 1
        field = field / jnp.maximum(jnp.abs(field).max(), 1e-6)
        return field[jnp.newaxis]  # (1, *spatial)

    sample_keys = jax.random.split(key, n)
    return jax.vmap(single)(sample_keys)  # (n, 1, *spatial)
