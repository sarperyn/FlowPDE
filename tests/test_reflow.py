"""Tests for reflow and the source-distribution component.

The load-bearing test here is ``test_reflow_pairs_are_preserved_during_training``:
the deleted implementation stored the generated noise but never used it, so
training resampled independently and the procedure straightened nothing. That
failure was silent — losses looked fine.
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from flowpde.flows import BatchSource, GaussianSource, NeuralODEFlow, get_source
from flowpde.models.mlp import MLP
from flowpde.objectives import FlowMatchingObjective
from flowpde.trainers import ReflowDataset, generate_reflow_pairs, reflow


# Source distributions


def test_gaussian_source_shape_and_scale():
    source = GaussianSource(std=3.0)
    out = source.sample((512, 4), torch.device("cpu"))
    assert out.shape == (512, 4)
    assert 2.5 < out.std().item() < 3.5


def test_batch_source_returns_precomputed_value():
    source = BatchSource(key="x_0")
    stored = torch.randn(4, 6)
    out = source.sample((4, 6), torch.device("cpu"), {"x_0": stored})
    assert torch.equal(out, stored)


def test_batch_source_falls_back_without_batch():
    """At inference there is no batch, so sampling must still work."""
    source = BatchSource(key="x_0")
    out = source.sample((4, 6), torch.device("cpu"), batch=None)
    assert out.shape == (4, 6)


def test_batch_source_strict_rejects_missing_key():
    """A missing key means reflow would silently resample noise."""
    source = BatchSource(key="x_0", strict=True)
    with pytest.raises(KeyError, match="silently disable path straightening"):
        source.sample((4, 6), torch.device("cpu"), {"target": torch.randn(4, 6)})


def test_batch_source_non_strict_falls_back():
    source = BatchSource(key="x_0", strict=False)
    out = source.sample((4, 6), torch.device("cpu"), {"target": torch.randn(4, 6)})
    assert out.shape == (4, 6)


def test_batch_source_rejects_shape_mismatch():
    source = BatchSource(key="x_0")
    with pytest.raises(ValueError, match="shape"):
        source.sample((4, 6), torch.device("cpu"), {"x_0": torch.randn(4, 99)})


def test_get_source_registry_and_passthrough():
    assert isinstance(get_source("gaussian"), GaussianSource)
    assert isinstance(get_source("batch"), BatchSource)
    instance = GaussianSource()
    assert get_source(instance) is instance
    with pytest.raises(ValueError, match="Unknown source"):
        get_source("nope")


# Objective integration


def build_objective(source="gaussian", coupling="independent", dim=4):
    model = MLP(input_dim=dim, condition_dim=dim, hidden_dim=32, num_layers=1)
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    return FlowMatchingObjective(
        flow, source=source, coupling=coupling,
        target_key="target", condition_key="input",
    )


def test_objective_defaults_to_gaussian_source():
    assert isinstance(build_objective().source, GaussianSource)


def test_reflow_pairs_are_preserved_during_training():
    """The bug that made the old reflow a no-op.

    With BatchSource the loss must be computed from the stored x_0. Since a
    linear path with a fixed (x_0, x_1) pair has target velocity exactly
    x_1 - x_0, we can detect resampling: an independently drawn x_0 would give
    a different target and hence a different loss on every call.
    """
    torch.manual_seed(0)
    objective = build_objective(source=BatchSource())

    batch = {
        "x_0": torch.randn(8, 4),
        "target": torch.randn(8, 4),
        "input": torch.randn(8, 4),
    }

    # Deterministic except for the time sample, so pin that too.
    objective.time_sampler = _FixedTime(0.5)

    losses = [objective.compute_loss(batch).item() for _ in range(5)]
    assert max(losses) - min(losses) < 1e-9, (
        "loss varies across calls, so x_0 is being resampled rather than "
        "read from the batch"
    )


class _FixedTime:
    def __init__(self, value):
        self.value = value

    def __call__(self, batch_size, device):
        return torch.full((batch_size, 1), self.value, device=device)


def test_gaussian_source_does_vary_across_calls():
    """Control for the test above: with fresh noise the loss must move."""
    torch.manual_seed(0)
    objective = build_objective(source="gaussian")
    objective.time_sampler = _FixedTime(0.5)
    batch = {"target": torch.randn(8, 4), "input": torch.randn(8, 4)}

    losses = [objective.compute_loss(batch).item() for _ in range(5)]
    assert max(losses) - min(losses) > 1e-6


def test_coupling_is_skipped_when_source_defines_pairing():
    """Minibatch-OT would reorder the reflow pairs and destroy them."""
    objective = build_objective(source=BatchSource(), coupling="minibatch_ot")
    x_0 = torch.randn(8, 4)
    batch = {"x_0": x_0, "target": torch.randn(8, 4), "input": torch.randn(8, 4)}

    assert objective._source_defines_pairing(batch) is True
    # Without a stored x_0 the coupling applies as usual.
    assert objective._source_defines_pairing({"target": x_0}) is False


def test_source_recorded_in_config():
    config = build_objective(source=BatchSource(key="z")).get_config()
    assert config["source"]["type"] == "BatchSource"
    assert config["source"]["key"] == "z"


# Reflow


class ConditionDataset(Dataset):
    def __init__(self, n=16, dim=4):
        generator = torch.Generator().manual_seed(0)
        self.f = torch.randn(n, dim, generator=generator)
        self.u = torch.randn(n, dim, generator=generator)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        return {"input": self.f[idx], "target": self.u[idx]}


def test_generate_reflow_pairs_shapes_and_keys():
    objective = build_objective()
    loader = DataLoader(ConditionDataset(), batch_size=8)

    pairs = generate_reflow_pairs(objective, loader, n_steps=10, solver="euler")

    assert isinstance(pairs, ReflowDataset)
    assert len(pairs) == 16
    sample = pairs[0]
    assert set(sample) == {"x_0", "target", "input"}
    assert sample["x_0"].shape == sample["target"].shape


def test_generated_targets_match_the_ode_from_stored_source():
    """The pairing must be exactly (z, ODE(z)), not (z, something else)."""
    objective = build_objective()
    loader = DataLoader(ConditionDataset(n=8), batch_size=8)

    pairs = generate_reflow_pairs(objective, loader, n_steps=20, solver="euler")

    replayed = objective.sample(
        condition=pairs.condition, n_steps=20, solver="euler", x_init=pairs.x_0
    )
    assert torch.allclose(replayed, pairs.x_1, atol=1e-4)


def test_reflow_pairs_are_reproducible_with_seed():
    objective = build_objective()
    loader = DataLoader(ConditionDataset(), batch_size=8)

    first = generate_reflow_pairs(objective, loader, n_steps=5, seed=7)
    second = generate_reflow_pairs(objective, loader, n_steps=5, seed=7)
    assert torch.allclose(first.x_0, second.x_0)
    assert torch.allclose(first.x_1, second.x_1)


def test_reflow_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="Mismatched lengths"):
        ReflowDataset(torch.randn(4, 2), torch.randn(3, 2), torch.randn(4, 2))


def test_reflow_straightens_trajectories(tmp_path):
    """The end the whole procedure exists for.

    Reflow retrains on the model's own transport map, so its trajectories
    should become measurably closer to straight lines.
    """
    torch.manual_seed(0)
    objective = build_objective()
    dataset = ConditionDataset(n=64)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Give the model a non-trivial (and non-straight) starting point.
    optimizer = torch.optim.Adam(objective.model.parameters(), lr=3e-3)
    for _ in range(60):
        for batch in loader:
            optimizer.zero_grad()
            objective.compute_loss(batch).backward()
            optimizer.step()

    probe = next(iter(DataLoader(dataset, batch_size=32)))
    before = objective.estimate_straightness(
        probe, n_time_points=21, mode="trajectory", n_steps=50
    )["normalized_straightness"]

    reflow(
        objective,
        loader,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=3e-3),
        num_iterations=1,
        epochs_per_iteration=60,
        n_steps=50,
        batch_size=16,
        save_dir=str(tmp_path),
        print_stats_interval=1000,
        seed=0,
    )

    after = objective.estimate_straightness(
        probe, n_time_points=21, mode="trajectory", n_steps=50
    )["normalized_straightness"]

    assert after < before, f"reflow should straighten paths ({before:.4f} -> {after:.4f})"


def test_reflow_restores_the_original_source(tmp_path):
    """Sampling behaviour must be unchanged after reflow returns."""
    objective = build_objective()
    original = objective.source
    loader = DataLoader(ConditionDataset(), batch_size=8)

    reflow(
        objective, loader,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3),
        num_iterations=1, epochs_per_iteration=2, n_steps=5,
        save_dir=str(tmp_path), print_stats_interval=1000,
    )

    assert objective.source is original
    assert isinstance(objective.source, GaussianSource)


def test_reflow_reports_history(tmp_path):
    objective = build_objective()
    loader = DataLoader(ConditionDataset(), batch_size=8)

    history = reflow(
        objective, loader,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3),
        num_iterations=2, epochs_per_iteration=2, n_steps=5,
        save_dir=str(tmp_path), print_stats_interval=1000,
    )

    assert len(history) == 2
    assert [record["iteration"] for record in history] == [1, 2]
    assert all(record["num_pairs"] == 16 for record in history)
