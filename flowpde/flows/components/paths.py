"""
Path Interpolation Strategies for Flow Matching.

Defines how to interpolate between noise (x_0) and data (x_1)
and compute the corresponding target velocity fields.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch
from torch import Tensor


class PathInterpolant(ABC):
    """
    Abstract base class for interpolation paths in flow matching.
    
    A path defines:
    1. How to interpolate: x_t = interpolate(x_0, x_1, t)
    2. The target velocity: v_t = velocity(x_0, x_1, t)
    
    The velocity is the derivative dx_t/dt that the model learns to predict.
    """
    
    @abstractmethod
    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Compute interpolated point x_t on the path.
        
        Args:
            x_0: Starting point (noise), shape (B, *)
            x_1: Ending point (data), shape (B, *)
            t: Time in [0, 1], shape (B, 1)
        
        Returns:
            x_t: Interpolated point, shape (B, *)
        """
        pass
    
    @abstractmethod
    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Compute target velocity v_t = dx_t/dt.
        
        Args:
            x_0: Starting point (noise), shape (B, *)
            x_1: Ending point (data), shape (B, *)
            t: Time in [0, 1], shape (B, 1)
        
        Returns:
            v_t: Target velocity, shape (B, *)
        """
        pass
    
    def __call__(
        self, 
        x_0: Tensor, 
        x_1: Tensor, 
        t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute both interpolation and velocity.
        
        Returns:
            (x_t, v_t): Interpolated point and target velocity
        """
        x_t = self.interpolate(x_0, x_1, t)
        v_t = self.velocity(x_0, x_1, t)
        return x_t, v_t


class LinearPath(PathInterpolant):
    """
    Linear interpolation path (straight lines).
    
    Path: x_t = (1 - t) * x_0 + t * x_1
    Velocity: v_t = x_1 - x_0 (constant along path)
    
    This is the standard path used in:
    - Flow Matching (Lipman et al., 2023)
    - Rectified Flow (Liu et al., 2023)
    
    Properties:
    - Straight lines from noise to data
    - Constant velocity (simplest to learn)
    - x_0 at t=0, x_1 at t=1
    """
    
    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """x_t = (1 - t) * x_0 + t * x_1"""
        t_expanded = self._expand_t(t, x_0.dim())
        return (1 - t_expanded) * x_0 + t_expanded * x_1
    
    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """v_t = x_1 - x_0 (constant)"""
        return x_1 - x_0
    
    def _expand_t(self, t: Tensor, ndim: int) -> Tensor:
        """Expand t for broadcasting: (B, 1) -> (B, 1, 1, ..., 1)"""
        return t.view(-1, *([1] * (ndim - 1)))


class OTConditionalPath(PathInterpolant):
    """
    Optimal Transport Conditional Flow path.
    
    Path: x_t = t * x_1 + (1 - t) * x_0 + σ * sqrt(t * (1-t)) * ε
    Velocity: v_t = x_1 - x_0 (ignoring noise term derivative)
    
    This is the OT-CFM path from Tong et al. (2023).
    When σ = 0, equivalent to LinearPath.
    
    Args:
        sigma: Noise scale (default: 0.0)
    """
    
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma
    
    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """x_t = t * x_1 + (1 - t) * x_0 + σ * sqrt(t * (1-t)) * ε"""
        t_expanded = self._expand_t(t, x_0.dim())
        
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        
        if self.sigma > 0:
            noise = torch.randn_like(x_0)
            noise_scale = self.sigma * torch.sqrt(t_expanded * (1 - t_expanded))
            x_t = x_t + noise_scale * noise
        
        return x_t
    
    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        v_t = x_1 - x_0
        
        Note: The noise term has a time-dependent coefficient, but
        the marginal target velocity is still x_1 - x_0.
        """
        return x_1 - x_0
    
    def _expand_t(self, t: Tensor, ndim: int) -> Tensor:
        """Expand t for broadcasting."""
        return t.view(-1, *([1] * (ndim - 1)))


# =============================================================================
# Factory Function
# =============================================================================

_PATH_REGISTRY = {
    'linear': LinearPath,
    'ot_conditional': OTConditionalPath,
    'conditional_optimal_transport': OTConditionalPath,  # Alias
    'ot': OTConditionalPath,  # Alias
}


def get_path(path: Union[str, PathInterpolant], **kwargs: Any) -> PathInterpolant:
    """
    Get a path interpolant by name or return if already an instance.
    
    Args:
        path: Path name ('linear', 'ot_conditional') or PathInterpolant instance
        **kwargs: Additional arguments for path constructor
    
    Returns:
        PathInterpolant instance
    
    Examples:
        >>> path = get_path('linear')
        >>> path = get_path('ot_conditional', sigma=0.01)
        >>> path = get_path(LinearPath())  # Pass-through
    """
    if isinstance(path, PathInterpolant):
        return path
    
    if path not in _PATH_REGISTRY:
        raise ValueError(
            f"Unknown path: '{path}'. "
            f"Available: {list(_PATH_REGISTRY.keys())}"
        )
    
    return _PATH_REGISTRY[path](**kwargs)
