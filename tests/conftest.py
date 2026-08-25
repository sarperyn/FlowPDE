"""Shared fixtures for the FlowPDE test suite."""

import pytest
import torch
from torch import nn

from flowpde.flows import NeuralODEFlow
from flowpde.objectives import FlowMatchingObjective


@pytest.fixture(autouse=True)
def deterministic():
    """Seed every test so failures are reproducible."""
    torch.manual_seed(1234)


@pytest.fixture
def device():
    return torch.device("cpu")


class ConstantVelocity(nn.Module):
    """Velocity field ``v(x, f, t) = c``, independent of x, f and t.

    Its flow is exactly straight, which makes several quantities analytically
    known and therefore testable.
    """

    def __init__(self, value: float = 0.7, dim: int = 8):
        super().__init__()
        self.register_buffer("c", torch.full((1, dim), value))
        # A trainable parameter so optimizers and EMA have something to track.
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, f, t):
        return self.c.expand_as(x) * self.scale


class LinearVelocity(nn.Module):
    """Small trainable velocity field with the standard ``(x, f, t)`` signature."""

    def __init__(self, dim: int = 8, condition_dim: int = None):
        super().__init__()
        condition_dim = condition_dim if condition_dim is not None else dim
        self.net = nn.Linear(dim + condition_dim + 1, dim)

    def forward(self, x, f, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], 1)
        return self.net(torch.cat([x, f, t], dim=1))


@pytest.fixture
def constant_velocity_model():
    return ConstantVelocity()


@pytest.fixture
def linear_velocity_model():
    return LinearVelocity()


@pytest.fixture
def batch():
    """A minimal ``{'target', 'input'}`` batch matching the toy models."""
    return {
        "target": torch.randn(6, 8),
        "input": torch.randn(6, 8),
    }


@pytest.fixture
def objective(linear_velocity_model):
    flow = NeuralODEFlow(
        linear_velocity_model, target_key="target", condition_key="input"
    )
    return FlowMatchingObjective(flow, target_key="target", condition_key="input")
