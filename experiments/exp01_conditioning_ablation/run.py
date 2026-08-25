"""Run Darcy conditioner ablations with UNet flow matching."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch

from experiments.common.config import deep_update, load_yaml, save_yaml
from experiments.common.data import build_darcy_splits, build_loaders
from experiments.common.metrics import build_flow_evaluator
from experiments.common.models import build_unet_objective
from experiments.common.training import build_trainer
from experiments.common.utils import ensure_dir, resolve_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional subset of variant names to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the config's quick overrides for a smoke test.",
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


def run_variant(
    name: str,
    variant_config: Dict[str, Any],
    base_config: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """Train and evaluate one conditioner variant."""
    run_seed = int(base_config["seed"]) + int(variant_config.get("seed_offset", 0))
    seed_everything(run_seed)

    output_root = ensure_dir(base_config["output_dir"])
    run_dir = ensure_dir(output_root / name)
    save_yaml(
        {
            **base_config,
            "active_variant": name,
            "active_variant_config": variant_config,
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

    objective = build_unet_objective(
        model_config=base_config["model"],
        objective_config=base_config["objective"],
        conditioner_name=variant_config["conditioner"],
    )
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
            "conditioner": variant_config["conditioner"],
        },
    )
    trainer.train(
        data_loader=train_loader,
        epochs=base_config["training"]["epochs"],
        print_stats_interval=base_config["training"].get("print_interval", 1),
        save_dir=str(run_dir),
        save_interval=base_config["training"].get("save_interval", 10),
    )

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

    result = {
        "variant": name,
        "conditioner": variant_config["conditioner"],
        "seed": run_seed,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "best_train_loss": trainer.best_loss,
        "best_val_metric": trainer.best_metric,
    }
    save_json(result, run_dir / "metrics.json")
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.quick:
        config = apply_quick_overrides(config)

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

