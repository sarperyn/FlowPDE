"""End-to-end tests for the benchmark scripts at the smoke scale."""

import json

import pytest
import torch
from torch.utils.data import DataLoader

import flowpde
from benchmarks import publish, report, run
from benchmarks.tasks import SCALES, TASKS, resolve_task
from flowpde.pretrained import load_pretrained


@pytest.fixture(autouse=True)
def quiet():
    flowpde.set_verbosity("WARNING")
    yield
    flowpde.set_verbosity("INFO")


@pytest.mark.parametrize("name", sorted(TASKS))
@pytest.mark.parametrize("scale", SCALES)
def test_every_task_resolves_at_every_scale(name, scale):
    task = resolve_task(name, scale)
    assert task["scale"] == scale
    assert {"data", "splits", "model", "training", "evaluation"} <= set(task)


def test_unknown_task_is_rejected():
    with pytest.raises(KeyError, match="Available"):
        resolve_task("navier_stokes")


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(TASKS))
def test_smoke_run_exports_a_loadable_model(name, tmp_path):
    assert run.main([name, "--scale", "smoke", "--device", "cpu", "--out", str(tmp_path)]) == 0

    run_dir = tmp_path / "smoke" / name
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["task"] == name
    assert metrics["test"]["rel_l2"] > 0

    model = load_pretrained(run_dir / f"{name}.pt")
    test = model.make_dataset(num_samples=2, seed=999)
    batch = next(iter(DataLoader(test, batch_size=2)))
    samples = model.sample(batch["input"], n_steps=2)
    assert samples.shape == batch["target"].shape
    assert torch.isfinite(samples).all()


@pytest.mark.slow
def test_report_and_publish(tmp_path, capsys):
    run.main(["burgers_forward", "--scale", "smoke", "--device", "cpu", "--out", str(tmp_path)])

    page = tmp_path / "benchmarks.md"
    page.write_text(f"intro\n{report.START}\nold\n{report.END}\noutro\n")
    report.main([str(tmp_path), "--write", str(page)])
    text = page.read_text()
    assert "`burgers_forward`" in text and "old" not in text and text.endswith("outro\n")

    registry = tmp_path / "registry.json"
    assert publish.main([str(tmp_path), "--tag", "t", "--registry", str(registry)]) == 1
    assert publish.main(
        [str(tmp_path), "--tag", "t", "--registry", str(registry), "--allow-smoke"]
    ) == 0
    entry = json.loads(registry.read_text())["burgers_forward_convnet"]
    assert entry["url"].endswith("/releases/download/t/burgers_forward_convnet.pt")
    assert len(entry["sha256"]) == 64
