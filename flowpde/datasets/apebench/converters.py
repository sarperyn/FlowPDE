"""
JAX ↔ PyTorch Conversion Utilities
===================================

Provides efficient conversion between JAX arrays and PyTorch tensors,
handling device placement and dtype conversion.
"""

import numpy as np
import torch
from typing import Union, Optional, Literal

# Type alias for device specification
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
        jax_array: JAX array to convert (jax.numpy.ndarray)
        device: Target PyTorch device ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: Target PyTorch dtype (default: torch.float32)
        
    Returns:
        PyTorch tensor on the specified device
        
    Example:
        >>> import jax.numpy as jnp
        >>> jax_arr = jnp.ones((10, 3, 64))
        >>> torch_tensor = jax_to_torch(jax_arr, device='cuda')
        >>> torch_tensor.shape
        torch.Size([10, 3, 64])
    """
    # Convert to NumPy first (this handles JAX device → CPU transfer)
    np_array = np.asarray(jax_array)
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(np_array)
    
    # Apply dtype conversion if specified
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    
    # Move to target device
    tensor = tensor.to(device=device)
    
    return tensor


def torch_to_jax(
    tensor: torch.Tensor,
    dtype: Optional[str] = 'float32',
):
    """
    Convert a PyTorch tensor to a JAX array.
    
    Args:
        tensor: PyTorch tensor to convert
        dtype: Target JAX dtype as string (default: 'float32')
        
    Returns:
        JAX array
        
    Note:
        This function requires JAX to be installed.
        Import is done lazily to avoid import errors when JAX is not available.
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for torch_to_jax conversion. "
            "Install with: pip install jax jaxlib"
        )
    
    # Move to CPU and convert to NumPy
    np_array = tensor.detach().cpu().numpy()
    
    # Convert to JAX array
    jax_array = jnp.array(np_array)
    
    # Apply dtype if specified
    if dtype is not None:
        jax_array = jax_array.astype(dtype)
    
    return jax_array


def check_jax_available() -> bool:
    """Check if JAX is available for import."""
    try:
        import jax
        return True
    except ImportError:
        return False


def check_apebench_available() -> bool:
    """Check if APEBench is available for import."""
    try:
        import apebench
        return True
    except ImportError:
        return False


def get_jax_device(device: Literal['cpu', 'gpu', 'auto'] = 'auto'):
    """
    Get a JAX device for computation.
    
    Args:
        device: Device type - 'cpu', 'gpu', or 'auto'
                'auto' will use GPU if available, else CPU
                
    Returns:
        JAX device object
    """
    try:
        import jax
    except ImportError:
        raise ImportError(
            "JAX is required. Install with: pip install jax jaxlib"
        )
    
    if device == 'auto':
        # Check if GPU is available
        try:
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) > 0:
                return gpu_devices[0]
        except RuntimeError:
            pass
        return jax.devices('cpu')[0]
    elif device == 'gpu':
        try:
            return jax.devices('gpu')[0]
        except RuntimeError:
            raise RuntimeError("No GPU device available for JAX")
    else:  # cpu
        return jax.devices('cpu')[0]


def reshape_trajectory_for_flowpde(
    trajectory: torch.Tensor,
    include_init: bool = True,
) -> dict:
    """
    Reshape APEBench trajectory format to FlowPDE convention.
    
    APEBench format: (N, T+1, C, *spatial)
    FlowPDE format: separate tensors for initial, final, trajectory
    
    Args:
        trajectory: Full trajectory tensor from APEBench
        include_init: Whether trajectory includes initial condition at t=0
        
    Returns:
        Dictionary with 'initial', 'final', 'trajectory' tensors
    """
    if include_init:
        initial = trajectory[:, 0]      # (N, C, *spatial)
        final = trajectory[:, -1]       # (N, C, *spatial)
        traj = trajectory[:, 1:]        # (N, T, C, *spatial) excluding init
    else:
        initial = trajectory[:, 0]
        final = trajectory[:, -1]
        traj = trajectory
    
    return {
        'initial': initial,
        'final': final,
        'trajectory': traj,
    }


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
