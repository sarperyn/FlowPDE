"""Create comparison figures for the Burgers backbone ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.common.config import deep_update, load_yaml
from experiments.common.data import build_burgers_splits, build_loaders
from experiments.common.models import build_objective, count_parameters
from experiments.common.utils import ensure_dir, resolve_device, seed_everything
from flowpde.utils.metrics import relative_l2_error_batch


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


def variant_model_config(
    base_config: Dict[str, Any],
    variant_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge base model settings with variant-specific backbone settings."""
    return deep_update(base_config["model"], variant_config.get("model", {}))


def apply_quick_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply optional quick-run overrides."""
    quick = config.get("quick")
    if not quick:
        return config
    merged = deep_update(config, quick)
    merged.pop("quick", None)
    return merged


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


def denormalize(normalizer, fields, tensor: torch.Tensor) -> torch.Tensor:
    """Map normalized target channels back to physical units."""
    return normalizer.denormalize_channels(fields, tensor)


def sample_predictions(
    objectives: Dict[str, Any],
    loader,
    normalizer,
    target_fields,
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

            target_physical = denormalize(normalizer, target_fields, target).cpu()
            condition_physical = normalizer.denormalize_channels(
                ["initial"], condition
            ).cpu()
            targets.append(target_physical)
            conditions.append(condition_physical)

            batch_row: Dict[str, Any] = {"batch": batch_idx}
            batch_predictions = {}
            for name, objective in objectives.items():
                sample = objective.sample(
                    condition=condition,
                    n_steps=n_steps,
                    solver=solver,
                    x_init=x_init.clone(),
                ).reshape(batch_size, *target_shape)
                sample_physical = denormalize(normalizer, target_fields, sample).cpu()
                predictions[name].append(sample_physical)
                batch_predictions[name] = sample_physical
                batch_row[f"{name}_rel_l2"] = relative_l2_error_batch(
                    sample_physical, target_physical
                ).mean().item()

            rows.append(batch_row)
            for sample_idx in range(batch_size):
                example = {
                    "index": batch_idx * batch_size + sample_idx,
                    "condition": condition_physical[sample_idx],
                    "target": target_physical[sample_idx],
                }
                best_error = float("inf")
                for name, sample_physical in batch_predictions.items():
                    pred = sample_physical[sample_idx]
                    rel_l2 = relative_l2_error_batch(
                        pred.unsqueeze(0), target_physical[sample_idx].unsqueeze(0)
                    ).item()
                    example[name] = pred
                    example[f"{name}_rel_l2"] = rel_l2
                    best_error = min(best_error, rel_l2)
                example["best_rel_l2"] = best_error
                examples.append(example)

    if not targets:
        raise RuntimeError("Test loader produced no batches.")

    stacked = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in predictions.items()
    }
    return {
        "rows": rows,
        "examples": examples,
        "predictions": stacked,
        "target": torch.cat(targets, dim=0),
        "condition": torch.cat(conditions, dim=0),
    }


def gradient_1d(x: torch.Tensor) -> torch.Tensor:
    """Centered periodic first derivative up to a constant grid spacing."""
    return 0.5 * (torch.roll(x, shifts=-1, dims=-1) - torch.roll(x, shifts=1, dims=-1))


def rel_l2_mean(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    return relative_l2_error_batch(pred, target, eps=eps).mean().item()


def shock_region_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.8,
    eps: float = 1e-8,
) -> float:
    """Relative L2 error on locations with largest target gradients."""
    grad = gradient_1d(target).abs()
    threshold = torch.quantile(grad.flatten(start_dim=1), quantile, dim=1)
    mask = grad >= threshold[:, None, None]
    diff_sq = ((pred - target) * mask).pow(2).flatten(start_dim=1).sum(dim=1)
    target_sq = (target * mask).pow(2).flatten(start_dim=1).sum(dim=1).clamp(min=eps)
    return (diff_sq.sqrt() / target_sq.sqrt()).mean().item()


def gradient_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return rel_l2_mean(gradient_1d(pred), gradient_1d(target))


def mass_drift(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.mean(dim=-1) - target.mean(dim=-1)).abs().mean().item()


def spectral_band_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    bands: Dict[str, tuple[int, int]],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute relative spectral magnitude errors over wavenumber bands."""
    pred_fft = torch.fft.rfft(pred[:, 0], dim=-1).abs()
    target_fft = torch.fft.rfft(target[:, 0], dim=-1).abs()
    out = {}
    for name, (lo, hi) in bands.items():
        pred_band = pred_fft[:, lo:hi]
        target_band = target_fft[:, lo:hi]
        num = (pred_band - target_band).pow(2).sum(dim=1).sqrt()
        den = target_band.pow(2).sum(dim=1).sqrt().clamp(min=eps)
        out[name] = (num / den).mean().item()
    return out


def compute_diagnostics(
    predictions: Dict[str, torch.Tensor],
    target: torch.Tensor,
    parameter_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Compute aggregate Burgers diagnostics for each variant."""
    bands = {"low": (1, 5), "mid": (5, 13), "high": (13, target.shape[-1] // 2 + 1)}
    rows = []
    for name, pred in predictions.items():
        spectral = spectral_band_errors(pred, target, bands)
        rows.append(
            {
                "variant": name,
                "parameter_count": parameter_counts[name],
                "rel_l2": rel_l2_mean(pred, target),
                "gradient_rel_l2": gradient_rel_l2(pred, target),
                "shock_rel_l2": shock_region_error(pred, target),
                "mass_drift": mass_drift(pred, target),
                **{f"spectral_{key}": value for key, value in spectral.items()},
            }
        )
    return rows


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
    """Choose a nontrivial example near the median best model error."""
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


def plot_line(ax, x: np.ndarray, fields: List[np.ndarray], labels: List[str], colors: List[str]) -> None:
    """Plot 1D fields with consistent styling."""
    for field, label, color in zip(fields, labels, colors):
        ax.plot(x, field, label=label, color=color, lw=1.7)
    ax.set_xlim(x.min(), x.max())
    ax.grid(alpha=0.2, lw=0.5)


def make_figure(
    diagnostics: List[Dict[str, Any]],
    example: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """Build the final multi-panel comparison figure."""
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

    fig = plt.figure(figsize=(11.5, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.15])

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

    ax_burgers = fig.add_subplot(grid[0, 4:6])
    width = 0.25
    metric_names = ["gradient_rel_l2", "shock_rel_l2", "mass_drift"]
    metric_labels = ["Gradient", "Shock", "Mass"]
    offsets = np.linspace(-width, width, len(variants))
    metric_x = np.arange(len(metric_names))
    for offset, name in zip(offsets, variants):
        row = next(item for item in diagnostics if item["variant"] == name)
        ax_burgers.bar(
            metric_x + offset,
            [row[key] for key in metric_names],
            width / max(len(variants), 1) * 3,
            label=name,
            color=color_map[name],
        )
    ax_burgers.set_yscale("log")
    ax_burgers.set_xticks(metric_x, metric_labels)
    ax_burgers.set_ylabel("Error, log scale")
    ax_burgers.set_title("C. Burgers diagnostics", loc="left", fontweight="bold")
    ax_burgers.legend(frameon=False, fontsize=6, ncols=2)

    x_grid = np.linspace(
        0.0,
        config["data"]["generator"]["domain_extent"],
        config["data"]["generator"]["num_points"],
        endpoint=False,
    )
    ax_lines = fig.add_subplot(grid[1, 0:4])
    fields = [
        example["condition"][0].numpy(),
        example["target"][0].numpy(),
        *[example[name][0].numpy() for name in variants],
    ]
    labels = ["initial", "target", *variants]
    line_colors = ["#555555", "#000000", *[color_map[name] for name in variants]]
    plot_line(ax_lines, x_grid, fields, labels, line_colors)
    ax_lines.set_xlabel("x")
    ax_lines.set_ylabel("u")
    ax_lines.set_title(
        f"D. Representative test sample #{example['index']}",
        loc="left",
        fontweight="bold",
    )
    ax_lines.legend(frameon=False, fontsize=7, ncols=3)

    ax_spec = fig.add_subplot(grid[1, 4:6])
    band_labels = ["low", "mid", "high"]
    band_x = np.arange(len(band_labels))
    for name in variants:
        row = next(item for item in diagnostics if item["variant"] == name)
        ax_spec.plot(
            band_x,
            [row[f"spectral_{band}"] for band in band_labels],
            marker="o",
            label=name,
            color=color_map[name],
            lw=1.7,
        )
    ax_spec.set_yscale("log")
    ax_spec.set_xticks(band_x, band_labels)
    ax_spec.set_xlabel("Frequency band")
    ax_spec.set_ylabel("Rel. spectral error")
    ax_spec.set_title("E. Spectral magnitude error", loc="left", fontweight="bold")
    ax_spec.grid(alpha=0.2, lw=0.5)

    fig.suptitle(
        "Backbone choice in Burgers conditional flow matching",
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
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot one test sample's inference result for every trained model."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ["#276fbf", "#2b9348", "#c44e52", "#7b2cbf", "#6c757d"]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(variants)}
    x_grid = np.linspace(
        0.0,
        config["data"]["generator"]["domain_extent"],
        config["data"]["generator"]["num_points"],
        endpoint=False,
    )

    n_variants = len(variants)
    fig = plt.figure(figsize=(12.0, 2.4 + 1.8 * n_variants), constrained_layout=True)
    grid = fig.add_gridspec(n_variants + 1, 2, height_ratios=[1.2, *([1.0] * n_variants)])

    ax_overlay = fig.add_subplot(grid[0, :])
    ax_overlay.plot(
        x_grid,
        example["condition"][0].numpy(),
        color="#777777",
        lw=1.4,
        ls="--",
        label="initial",
    )
    ax_overlay.plot(
        x_grid,
        example["target"][0].numpy(),
        color="#000000",
        lw=2.0,
        label="ground truth",
    )
    for name in variants:
        ax_overlay.plot(
            x_grid,
            example[name][0].numpy(),
            color=color_map[name],
            lw=1.6,
            label=f"{name} pred.",
        )
    ax_overlay.set_xlim(x_grid.min(), x_grid.max())
    ax_overlay.set_ylabel("u")
    ax_overlay.set_title(
        f"One-sample inference comparison, test sample #{example['index']}",
        loc="left",
        fontweight="bold",
    )
    ax_overlay.grid(alpha=0.2, lw=0.5)
    ax_overlay.legend(frameon=False, fontsize=7, ncols=min(6, len(variants) + 2))

    target = example["target"][0].numpy()
    max_abs_error = max(
        np.max(np.abs(example[name][0].numpy() - target)) for name in variants
    )
    for row_idx, name in enumerate(variants, start=1):
        prediction = example[name][0].numpy()
        error = prediction - target
        rel_l2 = example[f"{name}_rel_l2"]

        ax_pred = fig.add_subplot(grid[row_idx, 0])
        ax_pred.plot(x_grid, target, color="#000000", lw=1.8, label="ground truth")
        ax_pred.plot(
            x_grid,
            prediction,
            color=color_map[name],
            lw=1.7,
            label=f"{name} prediction",
        )
        ax_pred.set_xlim(x_grid.min(), x_grid.max())
        ax_pred.set_ylabel("u")
        ax_pred.set_title(f"{name}: prediction vs. ground truth", loc="left")
        ax_pred.text(
            0.99,
            0.92,
            f"rel. L2 = {rel_l2:.4f}",
            transform=ax_pred.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
        )
        ax_pred.grid(alpha=0.2, lw=0.5)
        ax_pred.legend(frameon=False, fontsize=7)

        ax_err = fig.add_subplot(grid[row_idx, 1])
        ax_err.axhline(0.0, color="#333333", lw=0.8)
        ax_err.fill_between(
            x_grid,
            error,
            0.0,
            color=color_map[name],
            alpha=0.35,
            linewidth=0,
        )
        ax_err.plot(x_grid, error, color=color_map[name], lw=1.3)
        ax_err.set_xlim(x_grid.min(), x_grid.max())
        ax_err.set_ylim(-1.05 * max_abs_error, 1.05 * max_abs_error)
        ax_err.set_ylabel("pred. - truth")
        ax_err.set_title(f"{name}: signed error", loc="left")
        ax_err.grid(alpha=0.2, lw=0.5)

    for ax in fig.axes[-2:]:
        ax.set_xlabel("x")

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
        args.output or results_dir / "figures" / "backbone_ablation_burgers.pdf"
    )

    selected = args.variants or list(config["variants"].keys())
    unknown = [name for name in selected if name not in config["variants"]]
    if unknown:
        raise KeyError(f"Unknown variant(s): {unknown}")

    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))

    train_ds, val_ds, test_ds, normalizer = build_burgers_splits(config["data"])
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
        predictions=sampled["predictions"],
        target=sampled["target"],
        parameter_counts=parameter_counts,
    )
    figure_dir = ensure_dir(results_dir / "figures")
    write_csv(diagnostics, figure_dir / "burgers_diagnostics.csv")
    write_csv(sampled["rows"], figure_dir / "batch_test_errors.csv")
    example = choose_example(sampled["examples"], args.sample_index)
    make_figure(diagnostics, example, config, output_path)
    inference_output = output_path.with_name("inference_example_burgers.pdf")
    make_inference_example_figure(selected, example, config, inference_output)
    print(f"Saved figure to {output_path}")
    print(f"Saved PNG to {output_path.with_suffix('.png')}")
    print(f"Saved inference example to {inference_output}")
    print(f"Saved inference example PNG to {inference_output.with_suffix('.png')}")
    print(f"Saved diagnostics to {figure_dir / 'burgers_diagnostics.csv'}")
    print(f"Saved batch errors to {figure_dir / 'batch_test_errors.csv'}")


if __name__ == "__main__":
    main()
