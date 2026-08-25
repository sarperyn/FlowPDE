"""Create comparison figures for the Darcy backbone ablation."""

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
from experiments.common.models import build_objective, count_parameters
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
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Test sample index for the inference-example figure.",
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


def variant_model_config(
    base_config: Dict[str, Any],
    variant_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge base model settings with variant-specific backbone settings."""
    return deep_update(base_config["model"], variant_config.get("model", {}))


def load_checkpoint_objective(
    config: Dict[str, Any],
    variant: str,
    checkpoint_path: Path,
    device: str,
):
    """Build an objective and load a saved model state."""
    variant_config = config["variants"][variant]
    objective = build_objective(
        model_config=variant_model_config(config, variant_config),
        objective_config=config["objective"],
        conditioner_name=variant_config.get("conditioner", "concat"),
        backbone=variant_config["backbone"],
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


def h1_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Return per-sample relative H1 errors for 2D fields."""
    diff = pred - target

    def norm_sq(x):
        value = x.pow(2).flatten(start_dim=1).sum(dim=1)
        grad_x = (x[..., 1:, :] - x[..., :-1, :]).pow(2).flatten(start_dim=1).sum(dim=1)
        grad_y = (x[..., :, 1:] - x[..., :, :-1]).pow(2).flatten(start_dim=1).sum(dim=1)
        return value + grad_x + grad_y

    return norm_sq(diff).sqrt() / norm_sq(target).sqrt().clamp(min=eps)


def mae_per_sample(pred: torch.Tensor, target: torch.Tensor):
    """Return per-sample mean absolute error."""
    return (pred - target).abs().flatten(start_dim=1).mean(dim=1)


def rel_max_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Return per-sample relative max error normalized by target std."""
    max_err = (pred - target).abs().flatten(start_dim=1).max(dim=1).values
    scale = target.flatten(start_dim=1).std(dim=1).clamp(min=eps)
    return max_err / scale


def region_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample relative L2 error restricted to a binary spatial mask."""
    diff_sq = ((pred - target) * mask).pow(2).flatten(start_dim=1).sum(dim=1)
    target_sq = (target * mask).pow(2).flatten(start_dim=1).sum(dim=1).clamp(min=eps)
    return diff_sq.sqrt() / target_sq.sqrt()


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
) -> Dict[str, Any]:
    """Sample all models on identical test batches and identical noise."""
    rows: List[Dict[str, float]] = []
    examples: List[Dict[str, Any]] = []
    predictions = {name: [] for name in objectives}
    targets = []
    conditions = []

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
            targets.append(target_phys.cpu())
            conditions.append(condition_phys.cpu())

            source_abs = condition_phys[:, 1:2].abs()
            source_threshold = torch.quantile(source_abs.flatten(start_dim=1), 0.8, dim=1)
            source_mask = source_abs >= source_threshold[:, None, None, None]
            kappa = condition_phys[:, 0:1]
            kappa_threshold = torch.quantile(kappa.flatten(start_dim=1), 0.8, dim=1)
            high_kappa_mask = kappa >= kappa_threshold[:, None, None, None]

            batch_predictions = {}
            metric_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
            for name, objective in objectives.items():
                samples = objective.sample(
                    condition=condition,
                    n_steps=n_steps,
                    solver=solver,
                    x_init=x_init.clone(),
                )
                pred = samples.reshape(batch_size, *target_shape)
                pred_phys = normalizer.denormalize_channels(target_fields, pred)
                predictions[name].append(pred_phys.cpu())
                batch_predictions[name] = pred_phys
                metric_tensors[name] = {
                    "rel_l2": relative_l2_per_sample(pred_phys, target_phys),
                    "h1": h1_per_sample(pred_phys, target_phys),
                    "mae": mae_per_sample(pred_phys, target_phys),
                    "rel_max": rel_max_per_sample(pred_phys, target_phys),
                    "source_region_rel_l2": region_rel_l2(
                        pred_phys, target_phys, source_mask
                    ),
                    "high_kappa_rel_l2": region_rel_l2(
                        pred_phys, target_phys, high_kappa_mask
                    ),
                }

            for sample_idx in range(batch_size):
                global_idx = batch_idx * loader.batch_size + sample_idx
                row: Dict[str, float] = {"index": float(global_idx)}
                example = {
                    "index": global_idx,
                    "condition": condition_phys[sample_idx].cpu(),
                    "target": target_phys[sample_idx].cpu(),
                }
                best_error = float("inf")
                for name, pred_phys in batch_predictions.items():
                    for metric_name, values in metric_tensors[name].items():
                        row[f"{name}_{metric_name}"] = float(values[sample_idx].item())
                    example[name] = pred_phys[sample_idx].cpu()
                    rel_l2 = row[f"{name}_rel_l2"]
                    example[f"{name}_rel_l2"] = rel_l2
                    best_error = min(best_error, rel_l2)
                example["best_rel_l2"] = best_error
                rows.append(row)
                examples.append(example)

    if not targets:
        raise RuntimeError("Test loader produced no batches.")

    return {
        "rows": rows,
        "examples": examples,
        "predictions": {
            name: torch.cat(chunks, dim=0)
            for name, chunks in predictions.items()
        },
        "target": torch.cat(targets, dim=0),
        "condition": torch.cat(conditions, dim=0),
    }


def compute_diagnostics(
    rows: List[Dict[str, float]],
    variants: List[str],
    parameter_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Aggregate per-sample metrics for each variant."""
    metric_names = [
        "rel_l2",
        "h1",
        "mae",
        "rel_max",
        "source_region_rel_l2",
        "high_kappa_rel_l2",
    ]
    diagnostics = []
    for variant in variants:
        diagnostics.append(
            {
                "variant": variant,
                "parameter_count": parameter_counts[variant],
                **{
                    metric: float(np.mean([row[f"{variant}_{metric}"] for row in rows]))
                    for metric in metric_names
                },
            }
        )
    return diagnostics


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def choose_representative_example(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose a sample near the median best-model endpoint error."""
    ordered = sorted(examples, key=lambda item: item["best_rel_l2"])
    return ordered[len(ordered) // 2]


def choose_example(
    examples: List[Dict[str, Any]],
    sample_index: int | None = None,
) -> Dict[str, Any]:
    """Choose a requested sample or a representative default."""
    if sample_index is None:
        return choose_representative_example(examples)
    for example in examples:
        if example["index"] == sample_index:
            return example
    available = (examples[0]["index"], examples[-1]["index"])
    raise ValueError(
        f"sample_index={sample_index} was not sampled. Available sampled "
        f"indices span {available[0]}..{available[1]}; increase --max-batches "
        "or choose an index in that range."
    )


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


def make_aggregate_figure(
    diagnostics: List[Dict[str, Any]],
    example: Dict[str, Any],
    output_path: Path,
) -> None:
    """Build the main backbone comparison figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    variants = [row["variant"] for row in diagnostics]
    colors = ["#276fbf", "#2b9348", "#c44e52", "#7b2cbf", "#6c757d"]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(variants)}

    fig = plt.figure(figsize=(11.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 8, height_ratios=[1.0, 1.2])

    ax_bar = fig.add_subplot(grid[0, 0:2])
    x_pos = np.arange(len(variants))
    rel_l2 = [row["rel_l2"] for row in diagnostics]
    ax_bar.bar(x_pos, rel_l2, color=[color_map[name] for name in variants])
    ax_bar.set_yscale("log")
    ax_bar.set_xticks(x_pos, variants, rotation=25, ha="right")
    ax_bar.set_ylabel("Rel. L2, log scale")
    ax_bar.set_title("A. Endpoint error", loc="left", fontweight="bold")
    for idx, row in enumerate(diagnostics):
        ax_bar.text(
            idx,
            rel_l2[idx] * 1.2,
            f"{row['parameter_count'] / 1e6:.2f}M",
            ha="center",
            fontsize=7,
        )

    ax_cost = fig.add_subplot(grid[0, 2:4])
    params = np.array([row["parameter_count"] for row in diagnostics])
    ax_cost.scatter(params, rel_l2, s=45, color=[color_map[name] for name in variants])
    for row in diagnostics:
        ax_cost.annotate(
            row["variant"],
            (row["parameter_count"], row["rel_l2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax_cost.set_xscale("log")
    ax_cost.set_yscale("log")
    ax_cost.set_xlabel("Trainable parameters")
    ax_cost.set_ylabel("Rel. L2")
    ax_cost.set_title("B. Accuracy vs. size", loc="left", fontweight="bold")
    ax_cost.grid(alpha=0.2, lw=0.5)

    ax_diagnostics = fig.add_subplot(grid[0, 4:8])
    metric_names = ["h1", "rel_max", "source_region_rel_l2", "high_kappa_rel_l2"]
    metric_labels = ["H1", "Rel. max", "High |source|", "High kappa"]
    metric_x = np.arange(len(metric_names))
    width = 0.18
    offsets = np.linspace(-0.27, 0.27, len(variants))
    for offset, name in zip(offsets, variants):
        row = next(item for item in diagnostics if item["variant"] == name)
        ax_diagnostics.bar(
            metric_x + offset,
            [row[key] for key in metric_names],
            width,
            label=name,
            color=color_map[name],
        )
    ax_diagnostics.set_yscale("log")
    ax_diagnostics.set_xticks(metric_x, metric_labels)
    ax_diagnostics.set_ylabel("Error, log scale")
    ax_diagnostics.set_title("C. Darcy diagnostics", loc="left", fontweight="bold")
    ax_diagnostics.legend(frameon=False, fontsize=6, ncols=2)

    condition = example["condition"].numpy()
    target = example["target"][0].numpy()
    best_variant = min(variants, key=lambda name: example[f"{name}_rel_l2"])
    best_pred = example[best_variant][0].numpy()
    best_err = np.abs(best_pred - target)

    axes = [fig.add_subplot(grid[1, i]) for i in range(8)]
    imshow_field(axes[0], condition[0], "D. kappa", "viridis")
    imshow_field(axes[1], condition[1], "source", "coolwarm", symmetric=True)
    imshow_field(axes[2], target, "target u", "magma")
    for ax, name in zip(axes[3:7], variants):
        imshow_field(ax, example[name][0].numpy(), f"{name}", "magma")
    imshow_field(axes[7], best_err, f"|best error|\n{best_variant}", "inferno")
    axes[0].set_ylabel(f"representative test sample #{example['index']}", fontsize=8)

    fig.suptitle(
        "Backbone choice in Darcy conditional flow matching",
        fontsize=12,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_inference_example_figure(
    variants: List[str],
    example: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot one Darcy test sample's inference result for every trained model."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    n_variants = len(variants)
    fig = plt.figure(figsize=(12.0, 2.4 + 2.1 * n_variants), constrained_layout=True)
    grid = fig.add_gridspec(n_variants + 1, 4, height_ratios=[1.0, *([1.0] * n_variants)])

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
            f"{name} prediction\nrel. L2={example[f'{name}_rel_l2']:.4f}",
            "magma",
        )
        imshow_field(
            fig.add_subplot(grid[row_idx, 1]),
            abs_error,
            f"{name} |error|",
            "inferno",
        )
        imshow_field(
            fig.add_subplot(grid[row_idx, 2]),
            signed_error,
            f"{name} signed error",
            "coolwarm",
            symmetric=True,
        )
        ax_overlay = fig.add_subplot(grid[row_idx, 3])
        center = target.shape[0] // 2
        x = np.arange(target.shape[1])
        ax_overlay.plot(x, target[center], color="black", lw=1.8, label="truth")
        ax_overlay.plot(x, prediction[center], lw=1.5, label=name)
        ax_overlay.set_xlim(x.min(), x.max())
        ax_overlay.set_title(f"{name} centerline", fontsize=8, pad=3)
        ax_overlay.set_xticks([])
        ax_overlay.grid(alpha=0.2, lw=0.5)
        ax_overlay.legend(frameon=False, fontsize=6)

    fig.suptitle(
        "Darcy one-sample inference comparison",
        fontsize=12,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.quick:
        config = apply_quick_overrides(config)
    results_dir = Path(args.results_dir or config["output_dir"])
    output_path = Path(
        args.output or results_dir / "figures" / "backbone_ablation_darcy.pdf"
    )

    selected = args.variants or list(config["variants"].keys())
    unknown = [name for name in selected if name not in config["variants"]]
    if unknown:
        raise KeyError(f"Unknown variant(s): {unknown}")

    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))

    train_ds, val_ds, test_ds, normalizer = build_darcy_splits(config["data"])
    _, _, test_loader = build_loaders(
        train_ds,
        val_ds,
        test_ds,
        config["data"]["loader"],
    )

    objectives = {
        variant: load_checkpoint_objective(
            config=config,
            variant=variant,
            checkpoint_path=results_dir / variant / "best_model.pt",
            device=device,
        )
        for variant in selected
    }
    parameter_counts = {
        variant: count_parameters(objective.model)
        for variant, objective in objectives.items()
    }

    n_steps = args.n_steps or config["evaluation"].get("n_steps", 50)
    solver = args.solver or config["evaluation"].get("solver", "euler")
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = config["evaluation"].get("plot_max_batches")

    sampled = sample_predictions(
        objectives=objectives,
        loader=test_loader,
        normalizer=normalizer,
        target_fields=test_ds.target_fields,
        device=device,
        n_steps=n_steps,
        solver=solver,
        max_batches=max_batches,
        seed=config["evaluation"].get("seed", 0),
    )
    diagnostics = compute_diagnostics(
        sampled["rows"],
        variants=selected,
        parameter_counts=parameter_counts,
    )
    figure_dir = ensure_dir(results_dir / "figures")
    write_csv(diagnostics, figure_dir / "darcy_diagnostics.csv")
    write_csv(sampled["rows"], figure_dir / "per_sample_test_errors.csv")
    example = choose_example(sampled["examples"], args.sample_index)
    make_aggregate_figure(diagnostics, example, output_path)
    inference_output = output_path.with_name("inference_example_darcy.pdf")
    make_inference_example_figure(selected, example, inference_output)
    print(f"Saved figure to {output_path}")
    print(f"Saved PNG to {output_path.with_suffix('.png')}")
    print(f"Saved inference example to {inference_output}")
    print(f"Saved inference example PNG to {inference_output.with_suffix('.png')}")
    print(f"Saved diagnostics to {figure_dir / 'darcy_diagnostics.csv'}")
    print(f"Saved per-sample errors to {figure_dir / 'per_sample_test_errors.csv'}")


if __name__ == "__main__":
    main()
