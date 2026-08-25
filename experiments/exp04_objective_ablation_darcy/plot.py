"""Create figures for the Darcy objective ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.common.config import deep_update, load_yaml
from experiments.common.data import build_darcy_splits, build_loaders
from experiments.common.models import build_objective
from experiments.common.utils import ensure_dir, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--results-dir", default=None, help="Override results directory.")
    parser.add_argument("--output", default=None, help="Output PDF path.")
    parser.add_argument("--variants", nargs="+", default=None, help="Subset of variants.")
    parser.add_argument("--quick", action="store_true", help="Use quick-run overrides.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override ODE steps.")
    parser.add_argument("--solver", default=None, help="Override ODE solver.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit test batches.")
    parser.add_argument("--sample-index", type=int, default=None, help="Inference sample index.")
    return parser.parse_args()


def apply_quick_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply optional quick-run overrides."""
    quick = config.get("quick")
    if not quick:
        return config
    merged = deep_update(config, quick)
    merged.pop("quick", None)
    return merged


def variant_objective_config(
    base_config: Dict[str, Any],
    variant_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge base objective settings with variant-specific objective settings."""
    return deep_update(base_config["objective"], variant_config.get("objective", {}))


def load_summary(path: Path) -> List[Dict[str, Any]]:
    """Load the run summary CSV."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value in {"", "None"}:
                row[key] = np.nan
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def load_checkpoint_objective(
    config: Dict[str, Any],
    variant: str,
    checkpoint_path: Path,
    device: str,
):
    """Build an objective and load a saved model state."""
    variant_config = config["variants"][variant]
    objective_config = variant_objective_config(config, variant_config)
    objective = build_objective(
        model_config=config["model"],
        objective_config=objective_config,
        conditioner_name=variant_config.get("conditioner", "concat"),
        backbone=config["model"].get("backbone", "unet"),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    objective.model.load_state_dict(checkpoint["model_state"])
    objective.to(device)
    objective.eval()
    return objective


def relative_l2_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Return per-sample relative L2 errors."""
    diff = (pred - target).flatten(start_dim=1).norm(dim=1)
    denom = target.flatten(start_dim=1).norm(dim=1).clamp(min=eps)
    return diff / denom


def sample_predictions(
    objectives: Dict[str, Any],
    loader,
    normalizer,
    target_fields: List[str],
    device: str,
    n_steps: int,
    solver: str,
    max_batches: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample all models on identical test batches and identical noise."""
    examples: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            target = batch["target"].to(device)
            condition = batch["input"].to(device)
            batch_size = target.shape[0]
            target_shape = target.shape[1:]
            flat_dim = int(target[0].numel())
            generator = torch.Generator().manual_seed(seed + 1009 * batch_idx)
            x_init = torch.randn(batch_size, flat_dim, generator=generator).to(device)

            target_phys = normalizer.denormalize_channels(target_fields, target)
            condition_phys = torch.cat(
                [
                    normalizer.denormalize("kappa", condition[:, 0:1]),
                    normalizer.denormalize("source", condition[:, 1:2]),
                ],
                dim=1,
            )
            batch_predictions = {}
            for name, objective in objectives.items():
                sampler = objective.sample if hasattr(objective, "sample") else objective.flow.sample
                sample = sampler(
                    condition=condition,
                    n_steps=n_steps,
                    solver=solver,
                    x_init=x_init.clone(),
                ).reshape(batch_size, *target_shape)
                pred_phys = normalizer.denormalize_channels(target_fields, sample)
                batch_predictions[name] = pred_phys

            for sample_idx in range(batch_size):
                example = {
                    "index": batch_idx * loader.batch_size + sample_idx,
                    "condition": condition_phys[sample_idx].cpu(),
                    "target": target_phys[sample_idx].cpu(),
                }
                best_error = float("inf")
                for name, pred_phys in batch_predictions.items():
                    pred = pred_phys[sample_idx].cpu()
                    rel_l2 = relative_l2_per_sample(
                        pred.unsqueeze(0), example["target"].unsqueeze(0)
                    ).item()
                    example[name] = pred
                    example[f"{name}_rel_l2"] = rel_l2
                    best_error = min(best_error, rel_l2)
                example["best_rel_l2"] = best_error
                examples.append(example)
    if not examples:
        raise RuntimeError("Test loader produced no examples.")
    return examples


def choose_example(
    examples: List[Dict[str, Any]],
    sample_index: int | None = None,
) -> Dict[str, Any]:
    """Choose a requested sample or a representative default."""
    if sample_index is None:
        ordered = sorted(examples, key=lambda item: item["best_rel_l2"])
        return ordered[len(ordered) // 2]
    for example in examples:
        if example["index"] == sample_index:
            return example
    raise ValueError(f"sample_index={sample_index} was not sampled.")


def imshow_field(ax, field: np.ndarray, title: str, cmap: str, symmetric: bool = False):
    """Image helper with compact colorbar."""
    vmin = vmax = None
    if symmetric:
        limit = np.percentile(np.abs(field), 99)
        vmin, vmax = -limit, limit
    image = ax.imshow(field, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6, length=2)


def make_tradeoff_figure(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot quality, likelihood, and runtime tradeoffs."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    variants = [str(row["variant"]) for row in rows]
    colors = ["#276fbf", "#c44e52", "#7b2cbf", "#2b9348"]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(variants)}
    x = np.arange(len(variants))

    fig = plt.figure(figsize=(11.5, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    ax_quality = fig.add_subplot(grid[0, 0])
    rel_l2 = [row["test_rel_l2"] for row in rows]
    ax_quality.bar(x, rel_l2, color=[color_map[name] for name in variants])
    ax_quality.set_yscale("log")
    ax_quality.set_xticks(x, variants, rotation=25, ha="right")
    ax_quality.set_ylabel("Rel. L2")
    ax_quality.set_title("A. Sample quality", loc="left", fontweight="bold")

    ax_h1 = fig.add_subplot(grid[0, 1])
    h1 = [row["test_h1"] for row in rows]
    ax_h1.bar(x, h1, color=[color_map[name] for name in variants])
    ax_h1.set_yscale("log")
    ax_h1.set_xticks(x, variants, rotation=25, ha="right")
    ax_h1.set_ylabel("H1 error")
    ax_h1.set_title("B. Gradient-sensitive error", loc="left", fontweight="bold")

    ax_nll = fig.add_subplot(grid[0, 2])
    nll = [row["test_nll_per_dim"] for row in rows]
    ax_nll.bar(x, nll, color=[color_map[name] for name in variants])
    ax_nll.set_xticks(x, variants, rotation=25, ha="right")
    ax_nll.set_ylabel("NLL / dim")
    ax_nll.set_title("C. Likelihood", loc="left", fontweight="bold")

    ax_time = fig.add_subplot(grid[1, 0])
    seconds = [row["seconds_per_epoch"] for row in rows]
    ax_time.bar(x, seconds, color=[color_map[name] for name in variants])
    ax_time.set_yscale("log")
    ax_time.set_xticks(x, variants, rotation=25, ha="right")
    ax_time.set_ylabel("Seconds / epoch")
    ax_time.set_title("D. Training cost", loc="left", fontweight="bold")

    ax_nll_time = fig.add_subplot(grid[1, 1])
    nll_seconds = [row["nll_seconds_per_batch"] for row in rows]
    ax_nll_time.bar(x, nll_seconds, color=[color_map[name] for name in variants])
    ax_nll_time.set_yscale("log")
    ax_nll_time.set_xticks(x, variants, rotation=25, ha="right")
    ax_nll_time.set_ylabel("Seconds / batch")
    ax_nll_time.set_title("E. Likelihood eval cost", loc="left", fontweight="bold")

    ax_scatter = fig.add_subplot(grid[1, 2])
    ax_scatter.scatter(seconds, rel_l2, s=55, color=[color_map[name] for name in variants])
    for row in rows:
        ax_scatter.annotate(
            str(row["variant"]),
            (row["seconds_per_epoch"], row["test_rel_l2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Seconds / epoch")
    ax_scatter.set_ylabel("Rel. L2")
    ax_scatter.set_title("F. Quality/cost frontier", loc="left", fontweight="bold")
    ax_scatter.grid(alpha=0.2, lw=0.5)

    fig.suptitle("Flow matching versus maximum likelihood on Darcy", fontsize=12, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_inference_example_figure(
    variants: List[str],
    example: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot one Darcy test sample's inference result for every objective."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(12.0, 2.4 + 2.0 * len(variants)), constrained_layout=True)
    grid = fig.add_gridspec(len(variants) + 1, 4, height_ratios=[1.0, *([1.0] * len(variants))])

    condition = example["condition"].numpy()
    target = example["target"][0].numpy()
    imshow_field(fig.add_subplot(grid[0, 0]), condition[0], "kappa", "viridis")
    imshow_field(fig.add_subplot(grid[0, 1]), condition[1], "source", "coolwarm", symmetric=True)
    imshow_field(fig.add_subplot(grid[0, 2]), target, "ground truth u", "magma")
    ax_blank = fig.add_subplot(grid[0, 3])
    ax_blank.axis("off")
    ax_blank.set_title(f"test sample #{example['index']}", fontsize=8, pad=3)

    for row_idx, name in enumerate(variants, start=1):
        prediction = example[name][0].numpy()
        abs_error = np.abs(prediction - target)
        signed_error = prediction - target
        imshow_field(
            fig.add_subplot(grid[row_idx, 0]),
            prediction,
            f"{name}\nrel. L2={example[f'{name}_rel_l2']:.4f}",
            "magma",
        )
        imshow_field(fig.add_subplot(grid[row_idx, 1]), abs_error, "|error|", "inferno")
        imshow_field(fig.add_subplot(grid[row_idx, 2]), signed_error, "signed error", "coolwarm", symmetric=True)
        ax_overlay = fig.add_subplot(grid[row_idx, 3])
        center = target.shape[0] // 2
        x = np.arange(target.shape[1])
        ax_overlay.plot(x, target[center], color="black", lw=1.8, label="truth")
        ax_overlay.plot(x, prediction[center], lw=1.5, label=name)
        ax_overlay.set_xlim(x.min(), x.max())
        ax_overlay.set_title("centerline", fontsize=8, pad=3)
        ax_overlay.set_xticks([])
        ax_overlay.grid(alpha=0.2, lw=0.5)
        ax_overlay.legend(frameon=False, fontsize=6)

    fig.suptitle("Objective ablation one-sample inference comparison", fontsize=12, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.quick:
        config = apply_quick_overrides(config)
    results_dir = Path(args.results_dir or config["output_dir"])
    output_path = Path(args.output or results_dir / "figures" / "objective_ablation_darcy.pdf")
    selected = args.variants or list(config["variants"].keys())

    rows = [row for row in load_summary(results_dir / "summary.csv") if row["variant"] in selected]
    if not rows:
        raise RuntimeError("No selected variants found in summary.csv.")
    make_tradeoff_figure(rows, output_path)

    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    train_ds, val_ds, test_ds, normalizer = build_darcy_splits(config["data"])
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, config["data"]["loader"])
    objectives = {
        variant: load_checkpoint_objective(
            config=config,
            variant=variant,
            checkpoint_path=results_dir / variant / "best_model.pt",
            device=device,
        )
        for variant in selected
    }
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = config["evaluation"].get("plot_max_batches")
    examples = sample_predictions(
        objectives=objectives,
        loader=test_loader,
        normalizer=normalizer,
        target_fields=test_ds.target_fields,
        device=device,
        n_steps=args.n_steps or config["evaluation"].get("n_steps", 50),
        solver=args.solver or config["evaluation"].get("solver", "euler"),
        max_batches=max_batches,
        seed=config["evaluation"].get("seed", 0),
    )
    inference_output = output_path.with_name("inference_example_objective_darcy.pdf")
    make_inference_example_figure(selected, choose_example(examples, args.sample_index), inference_output)

    print(f"Saved figure to {output_path}")
    print(f"Saved PNG to {output_path.with_suffix('.png')}")
    print(f"Saved inference example to {inference_output}")
    print(f"Saved inference example PNG to {inference_output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
