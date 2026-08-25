"""Tests for field normalization."""

import pytest
import torch

from flowpde.datasets import FieldNormalizer
from flowpde.datasets.exponax.base import PDEDataset


def make_dataset(scale: float = 5.0, offset: float = 2.0, n: int = 32):
    """A PDEDataset with deliberately non-unit scale, like real PDE fields."""
    source = torch.randn(n, 1, 8, 8) * scale + offset
    solution = torch.randn(n, 1, 8, 8) * (scale / 3) - offset
    stats = {
        "source": {"mean": source.mean().item(), "std": source.std().item()},
        "solution": {"mean": solution.mean().item(), "std": solution.std().item()},
    }
    return PDEDataset(
        {"source": source, "solution": solution},
        problem="forward",
        metadata={"stats": stats},
    )


def test_normalize_denormalize_roundtrip():
    normalizer = FieldNormalizer({"solution": {"mean": 2.0, "std": 4.0}})
    raw = torch.randn(10, 3) * 4.0 + 2.0
    restored = normalizer.denormalize("solution", normalizer.normalize("solution", raw))
    assert torch.allclose(restored, raw, atol=1e-5)


def test_normalize_produces_unit_scale():
    dataset = make_dataset()
    normalizer = FieldNormalizer.from_dataset(dataset)
    dataset.set_normalizer(normalizer)

    targets = torch.stack([dataset[i]["target"] for i in range(len(dataset))])
    assert abs(targets.mean().item()) < 1e-4
    assert abs(targets.std().item() - 1.0) < 1e-3


def test_unknown_field_passes_through_unchanged():
    """Auxiliary channels such as obs_mask must not be rescaled."""
    normalizer = FieldNormalizer({"solution": {"mean": 1.0, "std": 2.0}})
    mask = torch.randint(0, 2, (4, 1, 8, 8)).float()
    assert torch.equal(normalizer.normalize("obs_mask", mask), mask)


def test_zero_std_field_does_not_divide_by_zero():
    normalizer = FieldNormalizer({"constant": {"mean": 3.0, "std": 0.0}})
    out = normalizer.normalize("constant", torch.full((4,), 3.0))
    assert torch.isfinite(out).all()


def test_test_split_uses_train_statistics():
    """Sharing the train normalizer is what prevents evaluation leakage."""
    train = make_dataset(scale=5.0, offset=2.0)
    test = make_dataset(scale=50.0, offset=-30.0)

    normalizer = FieldNormalizer.from_dataset(train)
    train.set_normalizer(normalizer)
    test.set_normalizer(normalizer)

    test_targets = torch.stack([test[i]["target"] for i in range(len(test))])
    # Refitting would force std to 1; using train stats must not.
    assert abs(test_targets.std().item() - 1.0) > 0.5


def test_obs_mask_channel_stays_binary_after_normalization():
    n = 8
    source = torch.randn(n, 1, 8, 8) * 4 + 1
    solution = torch.randn(n, 1, 8, 8) * 4 + 1
    mask = (torch.rand(n, 1, 8, 8) < 0.5).float()
    stats = {
        "source": {"mean": 1.0, "std": 4.0},
        "solution": {"mean": 1.0, "std": 4.0},
    }
    dataset = PDEDataset(
        {"source": source, "solution": solution, "obs_mask": mask},
        problem="inverse",
        metadata={"stats": stats},
    )
    dataset.set_normalizer(FieldNormalizer(stats))

    sample = dataset[0]
    assert sample["input"].shape[0] == 2, "mask should be appended as a channel"
    mask_channel = sample["input"][1]
    assert torch.all((mask_channel == 0) | (mask_channel == 1))


def test_denormalize_channels_for_multi_field_targets():
    """Darcy's inverse_mode='both' target concatenates two fields."""
    normalizer = FieldNormalizer(
        {
            "kappa": {"mean": 10.0, "std": 2.0},
            "source": {"mean": -1.0, "std": 0.5},
        }
    )
    kappa = torch.randn(4, 1, 8, 8)
    source = torch.randn(4, 1, 8, 8)
    stacked = torch.cat([kappa, source], dim=1)

    restored = normalizer.denormalize_channels(["kappa", "source"], stacked)
    assert torch.allclose(restored[:, 0:1], kappa * 2.0 + 10.0, atol=1e-5)
    assert torch.allclose(restored[:, 1:2], source * 0.5 - 1.0, atol=1e-5)


def test_denormalize_channels_rejects_uneven_split():
    normalizer = FieldNormalizer(
        {"a": {"mean": 0.0, "std": 1.0}, "b": {"mean": 0.0, "std": 1.0}}
    )
    with pytest.raises(ValueError, match="evenly"):
        normalizer.denormalize_channels(["a", "b"], torch.randn(2, 3, 4, 4))


def test_state_dict_roundtrip():
    original = FieldNormalizer(
        {"solution": {"mean": 1.5, "std": 0.25}, "source": {"mean": -3.0, "std": 7.0}}
    )
    restored = FieldNormalizer.from_state_dict(original.state_dict())
    assert restored.stats == original.stats

    raw = torch.randn(5, 4)
    assert torch.allclose(
        restored.normalize("solution", raw), original.normalize("solution", raw)
    )


def test_from_dataset_requires_statistics():
    dataset = PDEDataset(
        {"source": torch.randn(4, 1, 4, 4), "solution": torch.randn(4, 1, 4, 4)},
        problem="forward",
    )
    with pytest.raises(ValueError, match="no normalization statistics"):
        FieldNormalizer.from_dataset(dataset)


def test_target_fields_track_problem_direction():
    """Field names, not roles, key the statistics — so flipping the problem
    direction reuses the same normalizer correctly."""
    forward = make_dataset()
    assert forward.target_fields == ["solution"]
    assert forward.input_fields == ["source"]

    inverse = PDEDataset(
        forward.data, problem="inverse", metadata=forward.metadata
    )
    assert inverse.target_fields == ["source"]
    assert inverse.input_fields == ["solution"]
