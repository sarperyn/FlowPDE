"""Train and evaluate FlowPDE benchmark tasks, and export pretrained models.

Examples:
    python -m benchmarks.run --list
    python -m benchmarks.run burgers_forward --scale quick
    python -m benchmarks.run --all --scale full --device cuda --out runs/

For each task this generates the train/val/test splits, fits the normalizer on
the training split, trains with EMA and validation-based model selection, then
scores the selected (averaged) weights on the full test split in physical
units.  It writes, under ``<out>/<task>/``:

- ``<task>.pt``     pretrained file for `flowpde.pretrained.load_pretrained`
- ``metrics.json``  test metrics, timings and the exact settings used
- ``train/``        Trainer checkpoints and loss curves
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

import flowpde
from flowpde.datasets import FieldNormalizer
from flowpde.pretrained import build_objective, data_spec, save_pretrained
from flowpde.trainers import FlowEvaluator, Trainer

from .tasks import SCALES, TASKS, resolve_task

logger = logging.getLogger("flowpde.benchmarks")


def pick_device(requested: str) -> str:
    """``auto`` picks CUDA, then Apple MPS, then CPU; anything else is used as given."""
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(f"--device {requested}: CUDA is not available on this machine.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("--device mps: Apple MPS is not available on this machine.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_splits(task: Dict[str, Any]):
    """Generate train/val/test with the task's fixed sizes and seeds."""
    from flowpde import datasets  # needs the `data` extra

    generator = getattr(datasets, task["data"]["generator"])(**task["data"]["config"])
    kwargs = {"problem": task["problem"]}
    if task.get("inverse_mode"):
        kwargs["inverse_mode"] = task["inverse_mode"]
    splits = task["splits"]
    out = {}
    for split in ("train", "val", "test"):
        out[split] = generator.generate(
            num_samples=splits[split], seed=splits[f"{split}_seed"], **kwargs
        )
    return generator, out


def build_recipe(task: Dict[str, Any], sample: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Complete the task's model settings with shapes read off the data."""
    target, condition = sample["target"], sample["input"]
    spatial_dim = target.dim() - 1
    kwargs = {
        "spatial_dim": spatial_dim,
        "spatial_size": int(target.shape[-1]),
        "solution_channels": int(target.shape[0]),
        "condition_channels": int(condition.shape[0]),
        **task["model"].get("kwargs", {}),
    }
    return {
        "model": {"name": task["model"]["name"], "kwargs": kwargs},
        "conditioner": task["conditioner"],
        "flow": {"target_key": "target", "condition_key": "input"},
        "objective": dict(task["objective"]),
    }


def time_sampling(objective, loader, n_steps: int, solver: str, device: str) -> float:
    """Seconds per sample for one batch, after a warm-up call."""
    batch = next(iter(loader))
    condition = batch["input"].to(device)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    objective.sample(condition, n_steps=n_steps, solver=solver)
    sync()
    start = time.perf_counter()
    objective.sample(condition, n_steps=n_steps, solver=solver)
    sync()
    return (time.perf_counter() - start) / condition.shape[0]


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_task(name: str, scale: str, device: str, out: Path, seed: int, amp: bool) -> Dict[str, Any]:
    task = resolve_task(name, scale)
    run_dir = out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s (%s scale) on %s -> %s", name, scale, device, run_dir)

    t0 = time.perf_counter()
    generator, splits = generate_splits(task)
    data_time = time.perf_counter() - t0

    normalizer = FieldNormalizer.from_dataset(splits["train"])
    for dataset in splits.values():
        dataset.set_normalizer(normalizer)
    target_fields = splits["train"].target_fields

    recipe = build_recipe(task, splits["train"][0])
    torch.manual_seed(seed)
    objective = build_objective(recipe)
    n_params = sum(p.numel() for p in objective.model.parameters())

    training, evaluation = task["training"], task["evaluation"]
    batch_size = training["batch_size"]
    generator_rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True,
                              generator=generator_rng)
    val_loader = DataLoader(splits["val"], batch_size=batch_size)
    test_loader = DataLoader(splits["test"], batch_size=batch_size)

    def evaluator(loader, max_batches):
        return FlowEvaluator(
            objective, loader,
            n_steps=evaluation["n_steps"], solver=evaluation["solver"],
            max_batches=max_batches, normalizer=normalizer,
            target_fields=target_fields, metrics=evaluation["metrics"],
            seed=evaluation["seed"],
        )

    optimizer = torch.optim.AdamW(
        objective.model.parameters(),
        lr=training["learning_rate"], weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"], eta_min=training["min_learning_rate"]
    )
    trainer = Trainer(
        objective, optimizer, scheduler,
        device=device,
        gradient_clip=training["gradient_clip"],
        use_amp=amp,
        ema_decay=training["ema_decay"],
        validator=evaluator(val_loader, training["val_max_batches"]),
        val_interval=training["val_interval"],
        monitor="rel_l2",
        checkpoint_extra={"normalizer_state": normalizer.state_dict()},
    )

    t0 = time.perf_counter()
    trainer.train(
        train_loader,
        epochs=training["epochs"],
        print_stats_interval=max(1, training["epochs"] // 20),
        save_dir=str(run_dir / "train"),
        save_interval=training["epochs"],
    )
    train_time = time.perf_counter() - t0

    # Score the weights model selection picked: the EMA average at the best
    # validation epoch.
    trainer.load_checkpoint(str(run_dir / "train" / "best_model.pt"), resume_training=False)
    objective.eval()
    test_metrics = evaluator(test_loader, None)()
    seconds_per_sample = time_sampling(
        objective, test_loader, evaluation["n_steps"], evaluation["solver"], device
    )

    target_shape = list(splits["train"][0]["target"].shape)
    summary = {
        "task": name,
        "scale": scale,
        "description": task["description"],
        "pde": task["pde"],
        "problem": task["problem"],
        "report": task.get("report"),
        "grid": "x".join([str(target_shape[-1])] * (len(target_shape) - 1)),
        "backbone": recipe["model"]["name"],
        "parameters": n_params,
        "train_samples": task["splits"]["train"],
        "test_samples": task["splits"]["test"],
        "epochs": training["epochs"],
        "test": test_metrics,
        "sampling": {
            "solver": evaluation["solver"],
            "n_steps": evaluation["n_steps"],
            "seconds_per_sample": seconds_per_sample,
        },
        "timing": {"data_seconds": data_time, "train_seconds": train_time},
        "environment": {
            "device": device,
            "device_name": torch.cuda.get_device_name() if device.startswith("cuda")
            else platform.processor() or platform.machine(),
            "torch": str(torch.__version__),
            "flowpde": flowpde.__version__,
            "python": platform.python_version(),
            "git_commit": git_commit(),
            "seed": seed,
            "amp": amp,
        },
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "settings": task,
    }

    save_pretrained(
        run_dir / f"{name}.pt",
        objective,
        recipe,
        target_shape=target_shape,
        normalizer=normalizer,
        data=data_spec(generator, task["problem"], task.get("inverse_mode"),
                       task["splits"], splits["train"]),
        metrics={**test_metrics, "seconds_per_sample": seconds_per_sample},
        info={k: summary[k] for k in ("task", "scale", "description", "epochs",
                                      "parameters", "date", "environment")},
    )
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "%s: test rel_l2 %.4f | %.1f s training | %.3g s/sample",
        name, test_metrics["rel_l2"], train_time, seconds_per_sample,
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Train and evaluate FlowPDE benchmark tasks.",
    )
    parser.add_argument("tasks", nargs="*", help="task names (see --list)")
    parser.add_argument("--all", action="store_true", help="run every task")
    parser.add_argument("--list", action="store_true", help="list tasks and exit")
    parser.add_argument("--scale", choices=SCALES, default="quick",
                        help="full = report settings (GPU); quick = laptop; smoke = CI")
    parser.add_argument("--device", default="auto",
                        help="auto (CUDA > MPS > CPU), cpu, mps, cuda, cuda:1, ...")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/runs"),
                        help="output root; results go to <out>/<scale>/<task>/")
    parser.add_argument("--seed", type=int, default=42, help="model and shuffling seed")
    parser.add_argument("--amp", action="store_true",
                        help="automatic mixed precision (CUDA recommended)")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(TASKS):
            print(f"{name:18s} {TASKS[name]['description']}")
        return 0
    names = sorted(TASKS) if args.all else args.tasks
    if not names:
        parser.error("name at least one task, or pass --all (see --list)")
    for name in names:
        if name not in TASKS:
            parser.error(f"unknown task '{name}'; available: {', '.join(sorted(TASKS))}")

    device = pick_device(args.device)
    if device == "mps":
        # Let the few operators MPS lacks fall back to the CPU instead of failing.
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    out = args.out / args.scale
    for name in names:
        run_task(name, args.scale, device, out, args.seed, args.amp)
    print(f"\nDone. Build the table with: python -m benchmarks.report {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
