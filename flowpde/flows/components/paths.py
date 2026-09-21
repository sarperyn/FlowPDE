"""
Path Interpolation Strategies for Flow Matching.

Defines how to interpolate between noise (x_0) and data (x_1)
and compute the corresponding target velocity fields.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

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
    r"""
    Optimal-transport conditional flow path (Tong et al., 2023).

    Two noise schedules are available, and both regress the *exact* derivative
    of the path the model is shown.

    ``schedule='constant'`` (default) — the OT-CFM path of Tong et al.:

        x_t = t * x_1 + (1 - t) * x_0 + sigma * eps
        v_t = x_1 - x_0

    The noise term does not depend on t, so the chord *is* the conditional
    velocity.  This is what makes sigma > 0 usable: the regression target
    stays bounded everywhere.

    ``schedule='bridge'`` — a Brownian-bridge tube that pins both endpoints:

        x_t = t * x_1 + (1 - t) * x_0 + sigma * sqrt(t(1-t)) * eps
        v_t = x_1 - x_0 + sigma * (1 - 2t) / (2 * sqrt(t(1-t))) * eps

    Here the tube width is time dependent, so the chord alone is *not* the
    derivative: the second term is required, or the model regresses a target
    that does not match its own input.  That term diverges as t approaches 0
    or 1 (it is clamped, but stays large), which is why the constant schedule
    is the default.

    With sigma = 0 both schedules reduce exactly to `LinearPath`.

    Args:
        sigma: Noise scale (default: 0.0).
        schedule: `'constant'` (default) or `'bridge'`.
    """

    _SCHEDULES = ("constant", "bridge")

    # Floor on t(1-t) inside the bridge derivative, so the endpoints give a
    # large-but-finite target instead of an infinity.
    _BRIDGE_FLOOR = 1e-6

    def __init__(self, sigma: float = 0.0, schedule: str = "constant"):
        if schedule not in self._SCHEDULES:
            raise ValueError(
                f"Unknown schedule: '{schedule}'. "
                f"Available: {list(self._SCHEDULES)}"
            )
        self.sigma = sigma
        self.schedule = schedule

    def _expand_t(self, t: Tensor, ndim: int) -> Tensor:
        """Expand t for broadcasting."""
        return t.view(-1, *([1] * (ndim - 1)))

    def _tube_width(self, t_expanded: Tensor) -> Union[float, Tensor]:
        r"""$\sigma_t$, the width of the conditional tube at time t."""
        if self.schedule == "constant":
            return self.sigma
        return self.sigma * torch.sqrt(
            (t_expanded * (1 - t_expanded)).clamp(min=0.0)
        )

    def _tube_width_derivative(self, t_expanded: Tensor) -> Union[float, Tensor]:
        r"""$d\sigma_t/dt$, the term the chord alone leaves out."""
        if self.schedule == "constant":
            return 0.0
        denominator = 2 * torch.sqrt(
            (t_expanded * (1 - t_expanded)).clamp(min=self._BRIDGE_FLOOR)
        )
        return self.sigma * (1 - 2 * t_expanded) / denominator

    def interpolate(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """x_t on the conditional path.

        Args:
            noise: The epsilon to use.  Pass the same one to `velocity()` so
                the target is the derivative of *this* sample path; `__call__`
                does that for you.
        """
        t_expanded = self._expand_t(t, x_0.dim())
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0

        if self.sigma > 0:
            noise = torch.randn_like(x_0) if noise is None else noise
            x_t = x_t + self._tube_width(t_expanded) * noise

        return x_t

    def velocity(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Exact derivative dx_t/dt of the path defined by `interpolate`."""
        v_t = x_1 - x_0

        if self.sigma > 0 and self.schedule != "constant":
            t_expanded = self._expand_t(t, x_0.dim())
            noise = torch.randn_like(x_0) if noise is None else noise
            v_t = v_t + self._tube_width_derivative(t_expanded) * noise

        return v_t

    def __call__(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Interpolated point and its target velocity, from a single epsilon.

        Drawing epsilon twice would hand the model an input from one sample
        path and the derivative of a different one.
        """
        noise = torch.randn_like(x_0) if self.sigma > 0 else None
        return (
            self.interpolate(x_0, x_1, t, noise=noise),
            self.velocity(x_0, x_1, t, noise=noise),
        )


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
