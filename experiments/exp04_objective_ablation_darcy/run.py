"""Run Darcy objective ablations: flow matching versus maximum likelihood."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from experiments.common.config import deep_update, load_yaml, save_yaml
from experiments.common.data import build_darcy_splits, build_loaders
from experiments.common.metrics import build_flow_evaluator
from experiments.common.models import build_objective, count_parameters
from experiments.common.training import build_trainer
from experiments.common.utils import ensure_dir, resolve_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--variants", nargs="+", default=None, help="Variant subset.")
    parser.add_argument("--quick", action="store_true", help="Use quick overrides.")
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=None,
        help="Override Darcy grid size for both data.generator.num_points and model.spatial_size.",
    )
    return parser.parse_args()


def apply_quick_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply optional quick-run overrides."""
    quick = config.get("quick")
    if not quick:
        return config
    merged = deep_update(config, quick)
    merged.pop("quick", None)
    return merged


def apply_spatial_size_override(config: Dict[str, Any], spatial_size: int | None) -> Dict[str, Any]:
    """Override grid size in both the data generator and model config."""
    if spatial_size is None:
        return config
    if spatial_size < 4:
        raise ValueError("--spatial-size must be at least 4 for the Darcy grid.")
    return deep_update(
        config,
        {
            "data": {"generator": {"num_points": spatial_size}},
            "model": {"spatial_size": spatial_size},
        },
    )


def write_summary(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write a CSV summary for all completed variants."""
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def variant_objective_config(
    base_config: Dict[str, Any],
    variant_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge base objective settings with variant-specific objective settings."""
    return deep_update(base_config["objective"], variant_config.get("objective", {}))


def evaluate_nll(
    objective,
    data_loader,
    max_batches: int | None,
    device: str,
) -> Dict[str, float]:
    """Evaluate normalized negative log likelihood on normalized targets."""
    was_training = objective.training
    objective.eval()
    total_nll = 0.0
    total_nll_per_dim = 0.0
    total_seconds = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        target = batch["target"].to(device)
        condition = batch["input"].to(device)
        flat_dim = int(target[0].numel())
        start = time.perf_counter()
        log_prob = objective.flow.log_prob(target, condition)
        elapsed = time.perf_counter() - start
        nll = -log_prob.mean()
        total_nll += nll.item()
        total_nll_per_dim += (nll / flat_dim).item()
        total_seconds += elapsed
        num_batches += 1

    if was_training:
        objective.train()
    if num_batches == 0:
        return {
            "test_nll": float("nan"),
            "test_nll_per_dim": float("nan"),
            "nll_seconds_per_batch": float("nan"),
        }
    return {
        "test_nll": total_nll / num_batches,
        "test_nll_per_dim": total_nll_per_dim / num_batches,
        "nll_seconds_per_batch": total_seconds / num_batches,
    }


def run_variant(
    name: str,
    variant_config: Dict[str, Any],
    base_config: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """Train and evaluate one objective variant."""
    run_seed = int(base_config["seed"]) + int(variant_config.get("seed_offset", 0))
    seed_everything(run_seed)

    output_root = ensure_dir(base_config["output_dir"])
    run_dir = ensure_dir(output_root / name)
    objective_config = variant_objective_config(base_config, variant_config)
    model_config = base_config["model"]
    backbone = model_config.get("backbone", "unet")
    save_yaml(
        {
            **base_config,
            "active_variant": name,
            "active_variant_config": variant_config,
            "active_objective_config": objective_config,
            "device": device,
        },
        run_dir / "resolved_config.yaml",
    )

    train_ds, val_ds, test_ds, normalizer = build_darcy_splits(base_config["data"])
    train_loader, val_loader, test_loader = build_loaders(
        train_ds,
        val_ds,
        test_ds,
        base_config["data"]["loader"],
    )

    objective = build_objective(
        model_config=model_config,
        objective_config=objective_config,
        conditioner_name=variant_config.get("conditioner", "concat"),
        backbone=backbone,
    )
    parameter_count = count_parameters(objective.model)
    val_eval = build_flow_evaluator(
        objective=objective,
        data_loader=val_loader,
        eval_config=base_config["evaluation"],
        normalizer=normalizer,
        target_fields=val_ds.target_fields,
        max_batches=base_config["evaluation"].get("val_max_batches"),
    )
    trainer = build_trainer(
        objective=objective,
        train_config=base_config["training"],
        device=device,
        validator=val_eval,
        checkpoint_extra={
            "normalizer_state": normalizer.state_dict(),
            "variant": name,
            "objective": objective_config["name"],
            "trace_estimator": objective_config.get("trace_estimator"),
            "n_trace_samples": objective_config.get("n_trace_samples"),
            "parameter_count": parameter_count,
        },
    )

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    trainer.train(
        data_loader=train_loader,
        epochs=base_config["training"]["epochs"],
        print_stats_interval=base_config["training"].get("print_interval", 1),
        save_dir=str(run_dir),
        save_interval=base_config["training"].get("save_interval", 10),
    )
    train_seconds = time.perf_counter() - train_start
    peak_memory_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    test_eval = build_flow_evaluator(
        objective=objective,
        data_loader=test_loader,
        eval_config=base_config["evaluation"],
        normalizer=normalizer,
        target_fields=test_ds.target_fields,
        max_batches=base_config["evaluation"].get("test_max_batches"),
    )
    with torch.no_grad():
        test_metrics = test_eval()

    nll_metrics = evaluate_nll(
        objective=objective,
        data_loader=test_loader,
        max_batches=base_config["evaluation"].get("nll_max_batches"),
        device=device,
    )

    result = {
        "variant": name,
        "objective": objective_config["name"],
        "trace_estimator": objective_config.get("trace_estimator"),
        "n_trace_samples": objective_config.get("n_trace_samples"),
        "seed": run_seed,
        "parameter_count": parameter_count,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **nll_metrics,
        "best_train_loss": trainer.best_loss,
        "best_val_metric": trainer.best_metric,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / base_config["training"]["epochs"],
        "peak_cuda_memory_mb": peak_memory_mb,
    }
    save_json(result, run_dir / "metrics.json")
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.quick:
        config = apply_quick_overrides(config)
    config = apply_spatial_size_override(config, args.spatial_size)

    selected = args.variants or list(config["variants"].keys())
    unknown = [name for name in selected if name not in config["variants"]]
    if unknown:
        raise KeyError(f"Unknown variant(s): {unknown}")

    device = resolve_device(config.get("device", "auto"))
    rows = []
    for name in selected:
        print(f"\n=== Running variant: {name} on {device} ===")
        rows.append(run_variant(name, config["variants"][name], config, device))

    summary_path = Path(config["output_dir"]) / "summary.csv"
    write_summary(rows, summary_path)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
