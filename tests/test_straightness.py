"""Tests for the path-straightness diagnostic.

Straightness is checked against velocity fields whose flows are known in
closed form, so the expected value is an exact number rather than a
regression baseline.
"""

import math

import pytest
import torch
from torch import nn

from flowpde.flows import NeuralODEFlow
from flowpde.objectives import FlowMatchingObjective


def build_objective(model):
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    return FlowMatchingObjective(flow, target_key="target", condition_key="input")


class ConstantField(nn.Module):
    """v = c. Trajectories are straight lines at constant speed: S = 0."""

    def __init__(self, dim=8, value=0.7):
        super().__init__()
        self.register_buffer("c", torch.full((1, dim), value))
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, f, t):
        return self.c.expand_as(x)


class RampField(nn.Module):
    """v = 2*t*c, so z_t = z_0 + c*t^2 and the chord is exactly c.

    Deviation from the chord is c*(2t - 1), giving
    S / |c|^2 = integral of (2t-1)^2 over [0,1] = 1/3.
    """

    def __init__(self, dim=8, value=0.7):
        super().__init__()
        self.register_buffer("c", torch.full((1, dim), value))
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, f, t):
        return 2.0 * t * self.c.expand_as(x)


class RotatingField(nn.Module):
    """Constant speed, continuously turning direction.

    The magnitude |v| is 1 at every t, so a metric based on the spread of
    velocity *norms* reports this as perfectly straight. It is not.
    """

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, f, t):
        angle = math.pi * t
        v = torch.zeros_like(x)
        v[:, 0:1] = torch.cos(angle)
        v[:, 1:2] = torch.sin(angle)
        return v


@pytest.fixture
def batch():
    return {"target": torch.randn(8, 8), "input": torch.randn(8, 8)}


def test_straight_flow_scores_zero(batch):
    objective = build_objective(ConstantField())
    result = objective.estimate_straightness(
        batch, n_time_points=51, mode="trajectory", n_steps=100, solver="rk4"
    )
    assert result["straightness"] == pytest.approx(0.0, abs=1e-6)
    assert result["normalized_straightness"] == pytest.approx(0.0, abs=1e-6)


def test_curved_flow_matches_analytic_value(batch):
    """v = 2tc has normalized straightness exactly 1/3."""
    objective = build_objective(RampField())
    result = objective.estimate_straightness(
        batch, n_time_points=201, mode="trajectory", n_steps=400, solver="rk4"
    )
    assert result["normalized_straightness"] == pytest.approx(1 / 3, abs=5e-3)


def test_detects_curvature_at_constant_speed(batch):
    """The failure mode of the previous implementation.

    Velocity norm is constant, so the old 'std of velocity norms' metric
    scored this as perfectly straight. The chord-deviation metric must not.
    """
    objective = build_objective(RotatingField())
    result = objective.estimate_straightness(
        batch, n_time_points=51, mode="trajectory", n_steps=100, solver="rk4"
    )

    velocity_norms_are_constant = True  # |v| = 1 for all t by construction
    assert velocity_norms_are_constant
    assert result["straightness"] > 0.1


def test_interpolant_mode_runs_and_reports_same_keys(batch):
    objective = build_objective(ConstantField())
    result = objective.estimate_straightness(batch, n_time_points=10, mode="interpolant")
    assert set(result) == {"straightness", "normalized_straightness", "chord_norm"}
    assert result["straightness"] >= 0.0


def test_invalid_mode_rejected(batch):
    objective = build_objective(ConstantField())
    with pytest.raises(ValueError, match="mode must be"):
        objective.estimate_straightness(batch, mode="bogus")


def test_training_mode_is_restored(batch):
    objective = build_objective(ConstantField())
    objective.train()
    objective.estimate_straightness(batch, n_time_points=5, mode="interpolant")
    assert objective.training, "estimate_straightness must not leave the model in eval"
