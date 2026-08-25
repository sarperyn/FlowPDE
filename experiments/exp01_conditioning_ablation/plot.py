"""Create paper-style comparison figures for the Darcy conditioning ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.common.config import load_yaml
from experiments.common.data import build_darcy_splits, build_loaders
from experiments.common.models import build_unet_objective
from experiments.common.utils import ensure_dir, resolve_device, seed_everything
from flowpde.datasets.normalization import FieldNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing null/ and concat/ run folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path. Defaults to <results-dir>/figures/conditioning_ablation.pdf.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on test batches for faster plotting.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override ODE sampling steps for the plot evaluation.",
    )
    parser.add_argument(
        "--solver",
        default=None,
        help="Override ODE solver for the plot evaluation.",
    )
    return parser.parse_args()


def load_checkpoint_objective(
    config: Dict[str, Any],
    variant: str,
    checkpoint_path: Path,
    device: str,
):
    """Build an objective and load a saved model state."""
    conditioner = config["variants"][variant]["conditioner"]
    objective = build_unet_objective(
        model_config=config["model"],
        objective_config=config["objective"],
        conditioner_name=conditioner,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    objective.model.load_state_dict(checkpoint["model_state"])
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


def sample_predictions(
    objectives: Dict[str, Any],
    loader,
    normalizer: FieldNormalizer,
    target_fields: List[str],
    device: str,
    n_steps: int,
    solver: str,
    max_batches: int | None,
    seed: int,
) -> Dict[str, Any]:
    """Sample both models on identical test batches and identical noise."""
    rows: List[Dict[str, float]] = []
    examples = []

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

        preds = {}
        for name, objective in objectives.items():
            samples = objective.sample(
                condition=condition,
                n_steps=n_steps,
                solver=solver,
                x_init=x_init,
            )
            pred = samples.reshape(batch_size, *target_shape)
            preds[name] = normalizer.denormalize_channels(target_fields, pred)

        rel_l2 = {
            name: relative_l2_per_sample(pred, target_phys).detach().cpu().numpy()
            for name, pred in preds.items()
        }
        h1 = {
            name: h1_per_sample(pred, target_phys).detach().cpu().numpy()
            for name, pred in preds.items()
        }
        mae = {
            name: mae_per_sample(pred, target_phys).detach().cpu().numpy()
            for name, pred in preds.items()
        }
        rel_max = {
            name: rel_max_per_sample(pred, target_phys).detach().cpu().numpy()
            for name, pred in preds.items()
        }

        for i in range(batch_size):
            global_idx = batch_idx * loader.batch_size + i
            rows.append(
                {
                    "index": float(global_idx),
                    "null_rel_l2": float(rel_l2["null"][i]),
                    "concat_rel_l2": float(rel_l2["concat"][i]),
                    "null_h1": float(h1["null"][i]),
                    "concat_h1": float(h1["concat"][i]),
                    "null_mae": float(mae["null"][i]),
                    "concat_mae": float(mae["concat"][i]),
                    "null_rel_max": float(rel_max["null"][i]),
                    "concat_rel_max": float(rel_max["concat"][i]),
                    "improvement": float(rel_l2["null"][i] / max(rel_l2["concat"][i], 1e-12)),
                }
            )
            examples.append(
                {
                    "index": global_idx,
                    "condition": condition_phys[i].detach().cpu(),
                    "target": target_phys[i].detach().cpu(),
                    "null": preds["null"][i].detach().cpu(),
                    "concat": preds["concat"][i].detach().cpu(),
                    "improvement": rows[-1]["improvement"],
                    "concat_rel_l2": rows[-1]["concat_rel_l2"],
                }
            )

    return {"rows": rows, "examples": examples}


def write_pairwise_csv(rows: List[Dict[str, float]], path: Path) -> None:
    """Save per-sample paired errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_representative_example(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose an example close to the median improvement ratio."""
    improvements = np.array([example["improvement"] for example in examples])
    median = np.median(improvements)
    idx = int(np.argmin(np.abs(improvements - median)))
    return examples[idx]


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


def make_figure(
    rows: List[Dict[str, float]],
    example: Dict[str, Any],
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

    null_l2 = np.array([row["null_rel_l2"] for row in rows])
    concat_l2 = np.array([row["concat_rel_l2"] for row in rows])
    improvement = null_l2 / np.maximum(concat_l2, 1e-12)
    win_rate = 100.0 * np.mean(concat_l2 < null_l2)

    fig = plt.figure(figsize=(11.5, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 7, height_ratios=[1.0, 1.2])

    ax_bar = fig.add_subplot(grid[0, 0:2])
    metrics = ["rel_l2", "h1", "rel_max"]
    labels = ["Rel. L2", "H1", "Rel. max"]
    x = np.arange(len(metrics))
    width = 0.36
    null_values = [np.mean([row[f"null_{metric}"] for row in rows]) for metric in metrics]
    concat_values = [np.mean([row[f"concat_{metric}"] for row in rows]) for metric in metrics]
    ax_bar.bar(x - width / 2, null_values, width, label="Null", color="#8c8c8c")
    ax_bar.bar(x + width / 2, concat_values, width, label="Concat", color="#2a6fbb")
    ax_bar.set_yscale("log")
    ax_bar.set_xticks(x, labels)
    ax_bar.set_ylabel("Error, log scale")
    ax_bar.set_title("A. Aggregate test metrics", loc="left", fontweight="bold")
    ax_bar.legend(frameon=False, fontsize=7)
    for i, (n_value, c_value) in enumerate(zip(null_values, concat_values)):
        ax_bar.text(i, max(n_value, c_value) * 1.25, f"{n_value / c_value:.1f}x", ha="center", fontsize=7)

    ax_scatter = fig.add_subplot(grid[0, 2:4])
    ax_scatter.scatter(null_l2, concat_l2, s=13, alpha=0.65, color="#2a6fbb", edgecolor="none")
    min_v = min(null_l2.min(), concat_l2.min()) * 0.8
    max_v = max(null_l2.max(), concat_l2.max()) * 1.2
    ax_scatter.plot([min_v, max_v], [min_v, max_v], color="black", lw=0.9, ls="--")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(min_v, max_v)
    ax_scatter.set_ylim(min_v, max_v)
    ax_scatter.set_xlabel("Null rel. L2")
    ax_scatter.set_ylabel("Concat rel. L2")
    ax_scatter.set_title("B. Paired test errors", loc="left", fontweight="bold")
    ax_scatter.text(
        0.04,
        0.95,
        f"concat wins {win_rate:.0f}%\nmedian gain {np.median(improvement):.1f}x",
        transform=ax_scatter.transAxes,
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )

    ax_cdf = fig.add_subplot(grid[0, 4:7])
    for values, label, color in [
        (null_l2, "Null", "#8c8c8c"),
        (concat_l2, "Concat", "#2a6fbb"),
    ]:
        sorted_values = np.sort(values)
        probs = np.linspace(0, 1, len(sorted_values), endpoint=False)
        ax_cdf.plot(sorted_values, probs, label=label, color=color, lw=2)
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("Per-sample rel. L2")
    ax_cdf.set_ylabel("Empirical CDF")
    ax_cdf.set_title("C. Error distribution", loc="left", fontweight="bold")
    ax_cdf.legend(frameon=False, fontsize=7)

    condition = example["condition"].numpy()
    target = example["target"][0].numpy()
    null_pred = example["null"][0].numpy()
    concat_pred = example["concat"][0].numpy()
    null_err = np.abs(null_pred - target)
    concat_err = np.abs(concat_pred - target)

    axes = [fig.add_subplot(grid[1, i]) for i in range(7)]
    imshow_field(axes[0], condition[0], "D. kappa", "viridis")
    imshow_field(axes[1], condition[1], "source", "coolwarm", symmetric=True)
    imshow_field(axes[2], target, "target u", "magma")
    imshow_field(axes[3], null_pred, "null pred.", "magma")
    imshow_field(axes[4], concat_pred, "concat pred.", "magma")
    imshow_field(axes[5], null_err, "|null error|", "inferno")
    imshow_field(axes[6], concat_err, "|concat error|", "inferno")
    axes[0].set_ylabel(
        f"representative test sample #{example['index']}\n"
        f"gain={example['improvement']:.1f}x",
        fontsize=8,
    )

    fig.suptitle(
        "Conditioning is the decisive ingredient for Darcy flow matching",
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
    results_dir = Path(args.results_dir or config["output_dir"])
    output_path = Path(
        args.output or results_dir / "figures" / "conditioning_ablation.pdf"
    )

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
        for variant in ("null", "concat")
    }

    n_steps = args.n_steps or config["evaluation"].get("n_steps", 50)
    solver = args.solver or config["evaluation"].get("solver", "euler")
    sampled = sample_predictions(
        objectives=objectives,
        loader=test_loader,
        normalizer=normalizer,
        target_fields=test_ds.target_fields,
        device=device,
        n_steps=n_steps,
        solver=solver,
        max_batches=args.max_batches,
        seed=config["evaluation"].get("seed", 0),
    )

    figure_dir = ensure_dir(results_dir / "figures")
    write_pairwise_csv(sampled["rows"], figure_dir / "paired_test_errors.csv")
    example = choose_representative_example(sampled["examples"])
    make_figure(sampled["rows"], example, output_path)
    print(f"Saved figure to {output_path}")
    print(f"Saved PNG to {output_path.with_suffix('.png')}")
    print(f"Saved paired errors to {figure_dir / 'paired_test_errors.csv'}")


if __name__ == "__main__":
    main()
