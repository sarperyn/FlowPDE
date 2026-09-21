"""Tests for the training loop: EMA integration, validation-based selection,
and checkpoint contents."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from flowpde.datasets import FieldNormalizer
from flowpde.flows import NeuralODEFlow
from flowpde.objectives import FlowMatchingObjective
from flowpde.trainers import FlowEvaluator, Trainer


class ToyDataset(Dataset):
    """A deterministic linear map input -> target, learnable by a tiny model."""

    def __init__(self, n=32, dim=4):
        generator = torch.Generator().manual_seed(0)
        self.f = torch.randn(n, 1, dim, generator=generator)
        self.u = 2.0 * self.f + 0.5

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        return {"input": self.f[idx], "target": self.u[idx]}


class TinyVelocity(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim + 1, 32), nn.SiLU(), nn.Linear(32, dim)
        )

    def forward(self, x, f, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x, f, t.expand(x.shape[0], 1)], dim=1))


@pytest.fixture
def setup():
    dataset = ToyDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = TinyVelocity()
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    objective = FlowMatchingObjective(flow, target_key="target", condition_key="input")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return dataset, loader, objective, optimizer


def test_training_runs_and_reduces_loss(setup, tmp_path):
    _, loader, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, device="cpu")
    trainer.train(loader, epochs=12, print_stats_interval=100, save_dir=str(tmp_path), save_interval=100)

    losses = trainer.history["train_loss"]
    assert len(losses) == 12
    assert min(losses[-3:]) < losses[0], "loss should decrease over training"


def test_ema_updates_once_per_step(setup, tmp_path):
    dataset, loader, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.99)
    epochs = 3
    trainer.train(loader, epochs=epochs, print_stats_interval=100, save_dir=str(tmp_path), save_interval=100)

    expected = epochs * len(loader)
    assert trainer.ema.num_updates == expected


def test_validation_metric_drives_model_selection(setup, tmp_path):
    """A validator that reports a fixed schedule: the best checkpoint must be
    written at the epoch with the lowest metric, not the lowest train loss."""
    _, loader, objective, optimizer = setup

    scores = [0.9, 0.5, 0.7, 0.3, 0.8]
    calls = {"n": 0}

    def validator():
        value = scores[calls["n"]]
        calls["n"] += 1
        return {"rel_l2": value}

    trainer = Trainer(
        objective, optimizer, device="cpu", validator=validator, monitor="rel_l2"
    )
    trainer.train(loader, epochs=5, print_stats_interval=100, save_dir=str(tmp_path), save_interval=100)

    assert calls["n"] == 5
    assert trainer.best_metric == pytest.approx(0.3)

    checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
    assert checkpoint["epoch"] == 3, "best checkpoint should come from the epoch scoring 0.3"


def test_monitor_mode_max(setup, tmp_path):
    _, loader, objective, optimizer = setup
    scores = iter([0.1, 0.9, 0.4])

    trainer = Trainer(
        objective,
        optimizer,
        device="cpu",
        validator=lambda: {"coverage": next(scores)},
        monitor="coverage",
        monitor_mode="max",
    )
    trainer.train(loader, epochs=3, print_stats_interval=100, save_dir=str(tmp_path), save_interval=100)
    assert trainer.best_metric == pytest.approx(0.9)


def test_unknown_monitor_key_raises(setup, tmp_path):
    _, loader, objective, optimizer = setup
    trainer = Trainer(
        objective, optimizer, device="cpu",
        validator=lambda: {"rel_l2": 0.5}, monitor="not_a_metric",
    )
    with pytest.raises(KeyError, match="not_a_metric"):
        trainer.train(loader, epochs=1, print_stats_interval=100, save_dir=str(tmp_path), save_interval=100)


def test_val_interval_controls_frequency(setup, tmp_path):
    _, loader, objective, optimizer = setup
    calls = {"n": 0}

    def validator():
        calls["n"] += 1
        return {"rel_l2": 1.0}

    trainer = Trainer(objective, optimizer, device="cpu", validator=validator, val_interval=3)
    trainer.train(loader, epochs=6, print_stats_interval=100, save_dir=str(tmp_path), save_interval=100)
    # Epochs 3 and 6 (1-indexed); the final epoch always validates.
    assert calls["n"] == 2


def test_checkpoint_carries_normalizer_and_ema_state(setup, tmp_path):
    _, loader, objective, optimizer = setup
    normalizer = FieldNormalizer({"solution": {"mean": 1.0, "std": 2.0}})

    trainer = Trainer(
        objective, optimizer, device="cpu", ema_decay=0.99,
        checkpoint_extra={"normalizer_state": normalizer.state_dict()},
    )
    trainer.train(loader, epochs=2, print_stats_interval=100, save_dir=str(tmp_path), save_interval=1)

    checkpoint = torch.load(tmp_path / "latest_checkpoint.pt", weights_only=False)
    assert "ema_state" in checkpoint
    assert "normalizer_state" in checkpoint

    # The normalizer must survive a round trip so inference can undo training
    # preprocessing without re-deriving statistics.
    restored = FieldNormalizer.from_state_dict(checkpoint["normalizer_state"])
    assert restored.stats["solution"]["std"] == pytest.approx(2.0)


def test_checkpoint_stores_ema_weights_as_model_state(setup, tmp_path):
    """Under EMA the averaged weights are what should be deployed."""
    _, loader, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.9)
    trainer.train(loader, epochs=2, print_stats_interval=100, save_dir=str(tmp_path), save_interval=1)

    checkpoint = torch.load(tmp_path / "latest_checkpoint.pt", weights_only=False)
    saved = checkpoint["model_state"]
    for name, shadow in trainer.ema.shadow.items():
        assert torch.allclose(saved[name], shadow, atol=1e-6)

    # ...and the live training weights were restored afterwards.
    live = dict(trainer.model.named_parameters())
    differs = any(
        not torch.allclose(live[name].detach(), shadow, atol=1e-6)
        for name, shadow in trainer.ema.shadow.items()
    )
    assert differs, "live weights should not have been overwritten by EMA"


def test_scheduler_is_optional(setup, tmp_path):
    """save_model previously crashed when scheduler was None."""
    _, loader, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, scheduler=None, device="cpu")
    trainer.train(loader, epochs=1, print_stats_interval=100, save_dir=str(tmp_path), save_interval=1)
    checkpoint = torch.load(tmp_path / "latest_checkpoint.pt", weights_only=False)
    assert checkpoint["scheduler_state"] is None


def test_end_to_end_with_real_evaluator(setup, tmp_path):
    """Training with a genuine sampling-based validator must lower the
    sampled-solution error, not merely the training loss."""
    dataset, loader, objective, optimizer = setup

    evaluator = FlowEvaluator(objective, loader, n_steps=20, solver="euler", max_batches=2)
    before = evaluator()["rel_l2"]

    trainer = Trainer(objective, optimizer, device="cpu", validator=evaluator, monitor="rel_l2")
    trainer.train(loader, epochs=40, print_stats_interval=100, save_dir=str(tmp_path), save_interval=100)

    after = evaluator()["rel_l2"]
    assert after < before, f"sampled error should improve ({before:.4f} -> {after:.4f})"


def test_checkpoint_keeps_raw_weights_for_resuming(setup, tmp_path):
    """Under EMA, 'model_state' holds the averaged weights -- so the live ones
    have to be saved separately or the run cannot be resumed at all.

    'optimizer_state' contains moments accumulated against the raw trajectory;
    pairing them with EMA weights on resume silently corrupts training.
    """
    _, loader, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.9)
    trainer.train(
        loader, epochs=3, print_stats_interval=100,
        save_dir=str(tmp_path), save_interval=3,
    )

    live = {name: t.clone() for name, t in trainer.model.state_dict().items()}
    checkpoint = torch.load(tmp_path / "latest_checkpoint.pt", weights_only=False)

    assert "raw_model_state" in checkpoint
    for name, tensor in live.items():
        assert torch.allclose(checkpoint["raw_model_state"][name], tensor)

    # The deployable-weights contract that inference code relies on is kept.
    shadow = checkpoint["ema_state"]["shadow"]
    for name, tensor in shadow.items():
        assert torch.allclose(checkpoint["model_state"][name], tensor)
    assert not torch.allclose(
        checkpoint["model_state"]["net.0.weight"], live["net.0.weight"]
    )


def test_load_checkpoint_resumes_exactly(setup, tmp_path):
    """Resuming restores the raw weights, the optimizer and the EMA shadow."""
    _, loader, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.9)
    trainer.train(
        loader, epochs=3, print_stats_interval=100,
        save_dir=str(tmp_path), save_interval=3,
    )
    live = {name: t.clone() for name, t in trainer.model.state_dict().items()}

    fresh_model = TinyVelocity()
    fresh_flow = NeuralODEFlow(fresh_model, target_key="target", condition_key="input")
    fresh_objective = FlowMatchingObjective(
        fresh_flow, target_key="target", condition_key="input"
    )
    resumed = Trainer(
        fresh_objective,
        torch.optim.Adam(fresh_model.parameters(), lr=1e-3),
        device="cpu",
        ema_decay=0.9,
    )
    resumed.load_checkpoint(str(tmp_path / "latest_checkpoint.pt"))

    for name, tensor in live.items():
        assert torch.allclose(resumed.model.state_dict()[name], tensor)
    for name, tensor in trainer.ema.shadow.items():
        assert torch.allclose(resumed.ema.shadow[name], tensor)
    assert resumed.ema.num_updates == trainer.ema.num_updates
    assert (
        resumed.optimizer.state_dict()["state"][0]["step"]
        == optimizer.state_dict()["state"][0]["step"]
    )


def test_load_checkpoint_for_inference_gives_ema_weights(setup, tmp_path):
    """resume_training=False loads the deployable averaged weights."""
    _, loader, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.9)
    trainer.train(
        loader, epochs=3, print_stats_interval=100,
        save_dir=str(tmp_path), save_interval=3,
    )
    checkpoint = torch.load(tmp_path / "latest_checkpoint.pt", weights_only=False)

    trainer.load_checkpoint(
        str(tmp_path / "latest_checkpoint.pt"), resume_training=False
    )
    for name, tensor in checkpoint["model_state"].items():
        assert torch.allclose(trainer.model.state_dict()[name], tensor)


def test_load_checkpoint_rejects_legacy_ema_checkpoint(setup, tmp_path):
    """An old checkpoint has no raw weights; say so instead of resuming wrong."""
    _, loader, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, device="cpu", ema_decay=0.9)
    trainer.train(
        loader, epochs=2, print_stats_interval=100,
        save_dir=str(tmp_path), save_interval=2,
    )

    path = tmp_path / "latest_checkpoint.pt"
    legacy = torch.load(path, weights_only=False)
    legacy.pop("raw_model_state")
    torch.save(legacy, path)

    with pytest.raises(KeyError, match="raw training weights"):
        trainer.load_checkpoint(str(path))

    # ...but it is still perfectly usable for inference.
    trainer.load_checkpoint(str(path), resume_training=False)


def test_device_default_does_not_assume_cuda(setup):
    """A hard-coded 'cuda' default crashes every CPU-only machine."""
    from flowpde.utils import resolve_device

    _, _, objective, optimizer = setup
    trainer = Trainer(objective, optimizer)

    assert trainer.device == resolve_device()
    assert trainer.device in {"cuda", "cpu"}
    # An explicit choice is always honoured.
    assert Trainer(objective, optimizer, device="cpu").device == "cpu"


def test_amp_uses_the_trainers_own_device(setup):
    """torch.cuda.amp made use_amp=True a silent no-op off CUDA."""
    import warnings

    _, loader, objective, optimizer = setup
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = Trainer(objective, optimizer, device="cpu", use_amp=True)
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]

    assert trainer._amp_device_type == "cpu"
    assert trainer.scaler.is_enabled()
    # It must actually run a step, not just construct.
    assert "loss" in trainer.step(next(iter(loader)))


def test_amp_rejects_a_backend_without_autocast(setup):
    _, _, objective, optimizer = setup
    with pytest.raises(ValueError, match="use_amp=True is not supported"):
        Trainer(objective, optimizer, device="meta", use_amp=True)


def test_saving_without_a_directory_is_an_explicit_error(setup):
    """save_dir is only set by train(); _save used to raise AttributeError."""
    _, _, objective, optimizer = setup
    trainer = Trainer(objective, optimizer, device="cpu")

    with pytest.raises(RuntimeError, match="No save directory"):
        trainer._save("best_model.pt", epoch=0, epoch_loss=0.0)
