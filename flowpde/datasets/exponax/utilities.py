"""
Exponax Dataset Utilities
=========================

Utility helpers for Exponax-backed datasets, including JAX↔PyTorch
conversions and normalization stats.
"""

from typing import Any, Optional, Union

import numpy as np
import torch

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
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("Cannot compute statistics for empty or nonfinite PDE data")
    return {
        'mean': tensor.mean().item(),
        'std': tensor.std(correction=0 if tensor.numel() == 1 else 1).item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
    }
