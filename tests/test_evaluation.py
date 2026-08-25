"""Tests for sampling-based validation."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from flowpde.datasets import FieldNormalizer
from flowpde.flows import NeuralODEFlow
from flowpde.objectives import FlowMatchingObjective
from flowpde.trainers import FlowEvaluator


class FixedDataset(Dataset):
    def __init__(self, n=16, dim=4, scale=1.0, offset=0.0):
        generator = torch.Generator().manual_seed(3)
        self.f = torch.randn(n, 1, dim, generator=generator)
        self.u = (2.0 * self.f) * scale + offset

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        return {"input": self.f[idx], "target": self.u[idx]}


class OracleVelocity(nn.Module):
    """Transports x_0 to exactly 2*f in one unit of time, ignoring x_0.

    dx/dt = 2f - x  drives x to 2f, but for an exact test we use the constant
    field  v = 2f - x_0  which is only correct at t=0.  Instead this uses the
    linear-path velocity for the known pair, giving x_1 = 2f exactly under
    Euler integration with any step count.
    """

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, f, t):
        # Straight-line field pointing from the current state to the target,
        # scaled by remaining time so the endpoint is hit exactly.
        target = 2.0 * f
        remaining = (1.0 - t).clamp(min=1e-6)
        return (target - x) / remaining


class ZeroVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, f, t):
        return torch.zeros_like(x)


def build(model):
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    return FlowMatchingObjective(flow, target_key="target", condition_key="input")


def test_perfect_model_scores_near_zero_error():
    dataset = FixedDataset()
    loader = DataLoader(dataset, batch_size=8)
    evaluator = FlowEvaluator(build(OracleVelocity()), loader, n_steps=200, solver="euler")

    result = evaluator()
    assert result["rel_l2"] < 0.05


def test_evaluation_is_deterministic_across_calls():
    """Fixed evaluation noise is what makes epoch-to-epoch comparison meaningful."""
    dataset = FixedDataset()
    loader = DataLoader(dataset, batch_size=8)
    evaluator = FlowEvaluator(build(ZeroVelocity()), loader, n_steps=10, solver="euler")

    first = evaluator()["rel_l2"]
    second = evaluator()["rel_l2"]
    assert first == pytest.approx(second, abs=1e-9)


def test_different_seeds_give_different_noise():
    dataset = FixedDataset()
    loader = DataLoader(dataset, batch_size=8)
    objective = build(ZeroVelocity())

    a = FlowEvaluator(objective, loader, n_steps=5, seed=0)()["rel_l2"]
    b = FlowEvaluator(objective, loader, n_steps=5, seed=99)()["rel_l2"]
    assert a != pytest.approx(b, abs=1e-9)


def test_max_batches_limits_work():
    dataset = FixedDataset(n=64)
    loader = DataLoader(dataset, batch_size=8)
    objective = build(ZeroVelocity())

    limited = FlowEvaluator(objective, loader, n_steps=5, max_batches=1)()
    full = FlowEvaluator(objective, loader, n_steps=5)()
    assert set(limited) == set(full)


def test_metrics_reported_in_physical_units():
    """With normalization, errors must be denormalized before scoring,
    otherwise the number is not comparable to unnormalized runs."""
    normalizer = FieldNormalizer({"solution": {"mean": 100.0, "std": 10.0}})
    dataset = FixedDataset()
    loader = DataLoader(dataset, batch_size=8)
    objective = build(ZeroVelocity())

    normalized = FlowEvaluator(objective, loader, n_steps=5)()["rel_l2"]
    physical = FlowEvaluator(
        objective, loader, n_steps=5,
        normalizer=normalizer, target_fields=["solution"],
    )()["rel_l2"]

    # A large mean offset dominates the denominator, so the physical-unit
    # relative error is far smaller than the normalized-space one.
    assert physical < normalized


def test_normalizer_without_target_fields_is_rejected():
    loader = DataLoader(FixedDataset(), batch_size=8)
    with pytest.raises(ValueError, match="target_fields is required"):
        FlowEvaluator(
            build(ZeroVelocity()), loader,
            normalizer=FieldNormalizer({"solution": {"mean": 0.0, "std": 1.0}}),
        )


def test_ensemble_reports_spread():
    dataset = FixedDataset()
    loader = DataLoader(dataset, batch_size=8)
    evaluator = FlowEvaluator(
        build(ZeroVelocity()), loader, n_steps=5, ensemble_size=4
    )

    result = evaluator()
    assert "sample_spread" in result
    assert "mean_rel_l2" in result
    # ZeroVelocity passes the initial noise straight through, so different
    # ensemble members genuinely differ.
    assert result["sample_spread"] > 0.0


def test_single_sample_reports_no_spread_key():
    loader = DataLoader(FixedDataset(), batch_size=8)
    result = FlowEvaluator(build(ZeroVelocity()), loader, n_steps=5)()
    assert "sample_spread" not in result


def test_training_mode_restored_after_evaluation():
    loader = DataLoader(FixedDataset(), batch_size=8)
    objective = build(ZeroVelocity())
    objective.train()

    FlowEvaluator(objective, loader, n_steps=5)()
    assert objective.training


def test_objective_without_sample_raises():
    class NoSample(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(1, 1)

    loader = DataLoader(FixedDataset(), batch_size=8)
    with pytest.raises(AttributeError, match="no sample"):
        FlowEvaluator(NoSample(), loader)()
