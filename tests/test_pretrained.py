"""Tests for pretrained checkpoints: recipes, save/load round trip, downloads."""

import json

import pytest
import torch

from flowpde.datasets import FieldNormalizer
from flowpde.pretrained import (
    build_objective,
    list_pretrained,
    load_pretrained,
    save_pretrained,
)
from flowpde.pretrained import checkpoint as checkpoint_module

RECIPE = {
    "model": {
        "name": "ConvNet",
        "kwargs": {
            "spatial_dim": 1,
            "spatial_size": 8,
            "hidden_channels": 8,
            "num_blocks": 2,
            "solution_channels": 1,
            "condition_channels": 1,
        },
    },
    "conditioner": "concat",
    "flow": {"target_key": "target", "condition_key": "input"},
    "objective": {"name": "flow_matching", "path": "linear", "time_sampler": "uniform"},
}


def _trained_objective():
    torch.manual_seed(0)
    objective = build_objective(RECIPE)
    batch = {"input": torch.randn(4, 1, 8), "target": torch.randn(4, 1, 8)}
    objective.compute_loss(batch).backward()  # records the flow's target dim
    return objective


@pytest.fixture
def saved(tmp_path):
    objective = _trained_objective()
    normalizer = FieldNormalizer({"final": {"mean": 1.0, "std": 2.0}})
    path = save_pretrained(
        tmp_path / "toy.pt",
        objective,
        RECIPE,
        target_shape=(1, 8),
        normalizer=normalizer,
        data={"generator": "BurgersGenerator", "config": {}, "problem": "forward",
              "target_fields": ["final"]},
        metrics={"rel_l2": 0.5},
        info={"scale": "test"},
    )
    return objective, path


def test_build_objective_from_recipe():
    objective = build_objective(RECIPE)
    assert type(objective.model).__name__ == "ConvNet"
    assert objective.flow.target_key == "target"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"model": {"name": "Transformer"}}, "Unknown model"),
        ({"conditioner": "film"}, "Unknown conditioner"),
        ({"objective": {"name": "maximum_likelihood"}}, "Unknown objective"),
        ({"objective": {"name": "flow_matching", "pathh": "linear"}}, "pathh"),
    ],
)
def test_invalid_recipe_is_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        build_objective({**RECIPE, **change})


def test_round_trip_restores_weights_and_metadata(saved):
    objective, path = saved
    model = load_pretrained(path)

    for name, tensor in objective.model.state_dict().items():
        assert torch.equal(model.objective.model.state_dict()[name], tensor)
    assert model.target_shape == [1, 8]
    assert model.metrics == {"rel_l2": 0.5}
    assert model.info == {"scale": "test"}
    assert model.normalizer.state_dict() == FieldNormalizer(
        {"final": {"mean": 1.0, "std": 2.0}}
    ).state_dict()
    assert not model.objective.training


def test_sample_shape_and_physical_units(saved):
    _, path = saved
    model = load_pretrained(path)
    condition = torch.randn(3, 1, 8)

    torch.manual_seed(1)
    physical = model.sample(condition, n_steps=2)
    torch.manual_seed(1)
    normalized = model.sample(condition, n_steps=2, physical_units=False)

    assert physical.shape == (3, 1, 8)
    torch.testing.assert_close(physical, normalized * 2.0 + 1.0)


def test_file_loads_with_weights_only(saved):
    _, path = saved
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["format"] == "flowpde-pretrained"


def test_metadata_is_stored_as_plain_values(tmp_path):
    """str subclasses like torch.__version__ would break weights_only loading."""
    path = save_pretrained(
        tmp_path / "plain.pt", _trained_objective(), RECIPE, target_shape=(1, 8),
        info={"torch": torch.__version__, "shape": (1, 8)},
        metrics={"rel_l2": torch.tensor(0.25)},
    )
    model = load_pretrained(path)
    assert model.info == {"torch": str(torch.__version__), "shape": [1, 8]}
    assert model.metrics == {"rel_l2": 0.25}


def test_training_checkpoint_is_rejected_with_hint(tmp_path):
    path = tmp_path / "best_model.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(ValueError, match="Trainer.load_checkpoint"):
        load_pretrained(path)


def test_newer_format_version_is_rejected(saved, tmp_path):
    _, path = saved
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["format_version"] = 999
    newer = tmp_path / "newer.pt"
    torch.save(checkpoint, newer)
    with pytest.raises(ValueError, match="Upgrade"):
        load_pretrained(newer)


def test_download_caches_and_verifies_checksum(saved, tmp_path):
    _, path = saved
    sha = checkpoint_module._sha256(path)
    cache = tmp_path / "cache"

    model = load_pretrained(path.as_uri(), sha256=sha, cache_dir=cache)
    assert model.target_shape == [1, 8]
    assert (cache / "toy.pt").is_file()
    assert not list(cache.glob("*.part"))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_pretrained(path.as_uri(), sha256="0" * 64, cache_dir=tmp_path / "other")
    assert not list((tmp_path / "other").iterdir())


def test_registry_names_resolve(saved, tmp_path, monkeypatch):
    _, path = saved
    entry = {"url": path.as_uri(), "sha256": checkpoint_module._sha256(path)}
    monkeypatch.setattr(checkpoint_module, "registry", lambda: {"toy": entry})

    assert list_pretrained() == ["toy"]
    model = load_pretrained("toy", cache_dir=tmp_path / "cache")
    assert model.metrics == {"rel_l2": 0.5}


def test_unknown_name_lists_what_exists():
    with pytest.raises(FileNotFoundError, match="neither a registered"):
        load_pretrained("no_such_model")


def test_shipped_registry_is_valid_json():
    entries = checkpoint_module.registry()
    assert isinstance(entries, dict)
    for name, entry in entries.items():
        assert {"url", "sha256"} <= set(entry), name
        assert len(entry["sha256"]) == 64, name
    json.dumps(entries)
