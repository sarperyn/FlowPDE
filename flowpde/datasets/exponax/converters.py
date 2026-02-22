"""
JAX ↔ PyTorch Conversion Utilities
===================================

Provides efficient conversion between JAX arrays and PyTorch tensors,
handling device placement and dtype conversion.
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
        Dictionary with 'mean' and 'std' values
    """
    return {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
    }
