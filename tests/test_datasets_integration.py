"""Integration tests over the real Exponax generators.

These exercise the full path a training run takes — generate, normalize,
batch, train — which unit tests with synthetic tensors cannot cover.  They
run the solvers, so they are kept small and marked ``slow``.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from flowpde.datasets import (
    BurgersGenerator,
    DarcyGenerator,
    FieldNormalizer,
    PoissonGenerator,
)
from flowpde.flows import NeuralODEFlow
from flowpde.models.mlp import MLP
from flowpde.objectives import FlowMatchingObjective
from flowpde.trainers import FlowEvaluator, Trainer

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def poisson_splits():
    generator = PoissonGenerator(num_spatial_dims=2, num_points=8)
    train = generator.generate(num_samples=24, seed=0)
    val = generator.generate(num_samples=8, seed=101)
    return train, val


def test_poisson_normalization_uses_train_statistics(poisson_splits):
    train, val = poisson_splits
    normalizer = FieldNormalizer.from_dataset(train)
    train.set_normalizer(normalizer)
    val.set_normalizer(normalizer)

    targets = torch.stack([train[i]["target"] for i in range(len(train))])
    assert abs(targets.mean().item()) < 1e-4
    assert abs(targets.std().item() - 1.0) < 1e-3

    # Both splits must be transformed by the *same* numbers.
    assert val.normalizer is train.normalizer


def test_forward_and_inverse_share_one_normalizer():
    """Field-keyed statistics stay correct when the problem direction flips."""
    generator = PoissonGenerator(num_spatial_dims=1, num_points=16)
    forward = generator.generate(num_samples=8, seed=0, problem="forward")
    inverse = generator.generate(num_samples=8, seed=0, problem="inverse")

    normalizer = FieldNormalizer.from_dataset(forward)
    forward.set_normalizer(normalizer)
    inverse.set_normalizer(normalizer)

    assert forward.target_fields == ["solution"]
    assert inverse.target_fields == ["source"]

    # The same raw field is standardized identically regardless of its role.
    assert torch.allclose(forward[0]["input"], inverse[0]["target"], atol=1e-6)


def test_inverse_problem_with_masking_appends_mask_channel():
    generator = PoissonGenerator(
        num_spatial_dims=1, num_points=16, obs_mask_fraction=0.5, obs_noise_std=0.1
    )
    dataset = generator.generate(num_samples=8, seed=0, problem="inverse")
    dataset.set_normalizer(FieldNormalizer.from_dataset(dataset))

    sample = dataset[0]
    assert sample["input"].shape[0] == 2, "observation + mask channel"
    mask = sample["input"][1]
    assert torch.all((mask == 0) | (mask == 1)), "mask must not be normalized"
    assert 0 < mask.mean().item() < 1


def test_darcy_target_fields_match_problem_mode():
    generator = DarcyGenerator(num_spatial_dims=2, num_points=8)
    forward = generator.generate(num_samples=4, seed=0, problem="forward")
    assert forward.target_fields == ["solution"]
    assert forward.input_fields == ["kappa", "source"]

    normalizer = FieldNormalizer.from_dataset(forward)
    forward.set_normalizer(normalizer)
    sample = forward[0]
    assert sample["input"].shape[0] == 2
    assert sample["target"].shape[0] == 1


def test_darcy_multi_field_target_denormalizes_channelwise():
    """inverse_mode='both' concatenates kappa and source into one target."""
    generator = DarcyGenerator(num_spatial_dims=2, num_points=8)
    dataset = generator.generate(
        num_samples=4, seed=0, problem="inverse", inverse_mode="both"
    )
    normalizer = FieldNormalizer.from_dataset(dataset)
    dataset.set_normalizer(normalizer)

    assert dataset.target_fields == ["kappa", "source"]

    normalized = torch.stack([dataset[i]["target"] for i in range(len(dataset))])
    restored = normalizer.denormalize_channels(dataset.target_fields, normalized)

    raw_kappa = dataset.get_raw_data()["kappa"]
    assert torch.allclose(restored[:, 0:1], raw_kappa, atol=1e-4)


def test_burgers_generates_initial_final_pairs():
    generator = BurgersGenerator(num_spatial_dims=1, num_points=32, num_steps=5)
    dataset = generator.generate(num_samples=8, seed=0)
    assert dataset.target_fields == ["final"]
    assert dataset.input_fields == ["initial"]

    dataset.set_normalizer(FieldNormalizer.from_dataset(dataset, fields=["initial", "final"]))
    sample = dataset[0]
    assert sample["input"].shape == sample["target"].shape


def test_end_to_end_poisson_training_improves_sampled_error(poisson_splits, tmp_path):
    """The full pipeline: generate, normalize, train with EMA, validate by
    sampling, and select on the sampled error."""
    train, val = poisson_splits
    normalizer = FieldNormalizer.from_dataset(train)
    train.set_normalizer(normalizer)
    val.set_normalizer(normalizer)

    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(val, batch_size=8)

    dim = 8 * 8
    model = MLP(input_dim=dim, condition_dim=dim, hidden_dim=64, num_layers=2)
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    objective = FlowMatchingObjective(flow, target_key="target", condition_key="input")

    evaluator = FlowEvaluator(
        objective, val_loader, n_steps=20, solver="euler",
        normalizer=normalizer, target_fields=val.target_fields,
    )
    before = evaluator()["rel_l2"]

    trainer = Trainer(
        objective,
        torch.optim.Adam(model.parameters(), lr=3e-3),
        device="cpu",
        ema_decay=0.995,
        validator=evaluator,
        monitor="rel_l2",
        val_interval=10,
        checkpoint_extra={"normalizer_state": normalizer.state_dict()},
    )
    trainer.train(
        train_loader, epochs=60, print_stats_interval=1000,
        save_dir=str(tmp_path), save_interval=1000,
    )

    after = evaluator()["rel_l2"]
    assert after < before, f"sampled error should improve ({before:.4f} -> {after:.4f})"

    checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
    restored = FieldNormalizer.from_state_dict(checkpoint["normalizer_state"])
    assert restored.stats.keys() == normalizer.stats.keys()
