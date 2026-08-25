"""Microbenchmark MLE training steps for the Darcy objective ablation."""

from __future__ import annotations

import argparse
import time

import torch

from experiments.common.config import deep_update, load_yaml
from experiments.common.models import build_objective
from experiments.common.training import build_optimizer
from experiments.common.utils import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--grid", type=int, default=64, help="Spatial grid size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Benchmark batch size.")
    parser.add_argument("--steps", type=int, default=3, help="Measured optimizer steps.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup optimizer steps.")
    parser.add_argument("--variant", default="mle_hutchinson_1", help="MLE variant name.")
    parser.add_argument("--device", default=None, help="Override device.")
    return parser.parse_args()


def variant_objective_config(base_config, variant_config):
    """Merge base objective settings with variant-specific objective settings."""
    return deep_update(base_config["objective"], variant_config.get("objective", {}))


def synchronize(device: str) -> None:
    """Synchronize CUDA timers when needed."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))
    device = args.device or resolve_device(config.get("device", "auto"))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")

    variant_config = config["variants"][args.variant]
    objective_config = variant_objective_config(config, variant_config)
    model_config = deep_update(
        config["model"],
        {
            "spatial_size": args.grid,
            "spatial_dim": 2,
            "solution_channels": 1,
            "condition_channels": 2,
        },
    )
    objective = build_objective(
        model_config=model_config,
        objective_config=objective_config,
        conditioner_name=variant_config.get("conditioner", "concat"),
        backbone=model_config.get("backbone", "unet"),
    ).to(device)
    optimizer = build_optimizer(objective.model.parameters(), config["training"])

    batch = {
        "input": torch.randn(args.batch_size, 2, args.grid, args.grid, device=device),
        "target": torch.randn(args.batch_size, 1, args.grid, args.grid, device=device),
    }

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    times = []
    total_steps = args.warmup + args.steps
    for step in range(total_steps):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad()
        loss = objective.compute_loss(batch)
        loss.backward()
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            times.append(elapsed)
        print(
            f"step={step + 1}/{total_steps} "
            f"loss={loss.item():.6f} seconds={elapsed:.3f}",
            flush=True,
        )

    mean_seconds = sum(times) / len(times)
    batches_per_epoch = config["data"]["splits"]["train"] / args.batch_size
    epoch_seconds = mean_seconds * batches_per_epoch
    print("\nBenchmark summary")
    print(f"device: {device}")
    print(f"variant: {args.variant}")
    print(f"grid: {args.grid}x{args.grid}")
    print(f"batch_size: {args.batch_size}")
    print(f"mean_seconds_per_step: {mean_seconds:.3f}")
    print(f"estimated_seconds_per_epoch: {epoch_seconds:.1f}")
    print(f"estimated_hours_for_config_epochs: {epoch_seconds * config['training']['epochs'] / 3600:.2f}")
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"peak_cuda_memory_mb: {peak_mb:.1f}")


if __name__ == "__main__":
    main()
