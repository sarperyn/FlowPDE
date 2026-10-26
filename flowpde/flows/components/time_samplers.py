"""
Time Sampling Strategies for Flow Matching.

Defines distributions for sampling time t ∈ [0, 1] during training.
Different sampling strategies can improve training dynamics.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Union

import torch
from torch import Tensor


class TimeSampler(ABC):
    """
    Abstract base class for time sampling distributions.

    Different time distributions can affect training:
    - Uniform: Standard, simple
    - Logit-normal: Default parameters concentrate samples toward the middle
    - Beta: Flexible, can emphasize different regions
    """

    def get_config(self):
        return {"type": type(self).__name__, **vars(self)}

    @abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Sample time values in [0, 1].

        Args:
            batch_size: Number of samples
            device: Device for tensor

        Returns:
            Time tensor of shape (batch_size, 1)
        """
        pass

    def __call__(self, batch_size: int, device: torch.device) -> Tensor:
        """Alias for sample()."""
        return self.sample(batch_size, device)


class UniformSampler(TimeSampler):
    """
    Uniform time sampling: t ~ U(low, high).

    Standard sampling strategy for flow matching.

    Args:
        low: Lower bound (default: 0.0)
        high: Upper bound (default: 1.0)
    """

    def __init__(self, low: float = 0.0, high: float = 1.0):
        if not 0 <= low < high <= 1:
            raise ValueError("Uniform time bounds must satisfy 0 <= low < high <= 1")
        self.low = low
        self.high = high

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample t ~ U(low, high)."""
        return torch.rand(batch_size, 1, device=device) * (self.high - self.low) + self.low


class LogitNormalSampler(TimeSampler):
    """
    Logit-normal time sampling: t = sigmoid(z) where z ~ N(mean, std²).

    With mean=0 and std=1 this distribution emphasizes the middle of the
    interval. Larger std can shift mass toward the endpoints.

    Used in Rectified Flow and some diffusion models.

    Args:
        mean: Mean of the normal distribution (default: 0.0)
        std: Standard deviation (default: 1.0)
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("Logit-normal mean must be finite and std positive")
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample t = sigmoid(z) where z ~ N(mean, std²)."""
        z = torch.randn(batch_size, 1, device=device) * self.std + self.mean
        t = torch.sigmoid(z)
        return t


class BetaSampler(TimeSampler):
    """
    Beta distribution time sampling: t ~ Beta(alpha, beta).

    Flexible distribution that can emphasize different regions:
    - alpha=beta=1: Uniform
    - alpha=beta=0.5: U-shaped (emphasizes boundaries)
    - alpha=beta>1: Concentrated around 0.5

    Args:
        alpha: First shape parameter (default: 1.0)
        beta: Second shape parameter (default: 1.0)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        if not math.isfinite(alpha) or not math.isfinite(beta) or alpha <= 0 or beta <= 0:
            raise ValueError("Beta parameters must be finite and positive")

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample t ~ Beta(alpha, beta)."""
        # Sample and move to device
        dist = torch.distributions.Beta(
            torch.tensor(float(self.alpha), device=device), torch.tensor(float(self.beta), device=device)
        )
        return dist.sample((batch_size, 1))


# =============================================================================
# Factory Function
# =============================================================================

_SAMPLER_REGISTRY = {
    'uniform': UniformSampler,
    'logit_normal': LogitNormalSampler,
    'logitnormal': LogitNormalSampler,  # Alias
    'beta': BetaSampler,
}


def get_time_sampler(
    sampler: Union[str, TimeSampler],
    **kwargs: Any
) -> TimeSampler:
    """
    Get a time sampler by name or return if already an instance.

    Args:
        sampler: Sampler name ('uniform', 'logit_normal', 'beta') or instance
        **kwargs: Additional arguments for sampler constructor

    Returns:
        TimeSampler instance

    Examples:
        >>> sampler = get_time_sampler('uniform')
        >>> sampler = get_time_sampler('logit_normal', std=0.5)
        >>> sampler = get_time_sampler('beta', alpha=0.5, beta=0.5)
    """
    if isinstance(sampler, TimeSampler):
        return sampler

    if isinstance(sampler, dict):
        config = dict(sampler)
        name = config.pop("type")
        classes = {cls.__name__: cls for cls in _SAMPLER_REGISTRY.values()}
        return classes[name](**config)

    if sampler not in _SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown time sampler: '{sampler}'. "
            f"Available: {list(_SAMPLER_REGISTRY.keys())}"
        )

    return _SAMPLER_REGISTRY[sampler](**kwargs)
