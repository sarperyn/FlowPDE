"""Figures for the inverse problem: posterior samples, calibration, and where
the data actually constrain the coefficient.

These read the ensembles cached by ``experiments.report_analysis.uq_suite``
(``results/analysis/ensembles/*.pt``) so that nothing is re-sampled here and
every panel refers to the same draw that produced the reported scores.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.report_figures.style import (
    ACCENT, ACCENT_LIGHT, GRAY, PALE, RULE, SERIES, TEXT_WIDTH, WARM,
    HEIGHT_SCALE, field_axes, grid, panel_label, save, use_report_style,
)

ANALYSIS = Path("results/analysis")
ENSEMBLES = ANALYSIS / "ensembles"

# Order here is the order panel (c) draws them in.
PRETTY = {
    "inverse_coefficient_5000_concat": r"$\kappa$, 5000, concat",
    "inverse_coefficient_5000_null": r"$\kappa$, 5000, null",
    "inverse_coefficient_3000_concat": r"$\kappa$, 3000, concat",
    "inverse_coefficient_3000_null": r"$\kappa$, 3000, null",
    "inverse_joint_5000_concat": r"$(\kappa,f)$, 5000, concat",
    "inverse_joint_5000_null": r"$(\kappa,f)$, 5000, null",
    "inverse_joint_3000_concat": r"$(\kappa,f)$, 3000, concat",
    "inverse_joint_3000_null": r"$(\kappa,f)$, 3000, null",
    "forward_darcy_convnet": "forward, ConvNet",
    "forward_darcy_unet": "forward, UNet",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def load_ensemble(label: str):
    return torch.load(ENSEMBLES / f"{label}.pt", weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# Posterior samples
# ─────────────────────────────────────────────────────────────────────────────

def figure_inverse_samples(index: int = 0, channel: int = 0) -> None:
    """What a posterior draw looks like, with and without the observation.

    The joint target is used rather than the coefficient-only one because the
    coefficient-only null run at 5000 samples has no saved checkpoint, and a
    prior/posterior comparison is only meaningful at matched training-set size.
    Channel 0 of the joint target is $\\kappa$.
    """
    use_report_style()

    rows = [
        ("inverse_joint_5000_null", "null (prior)"),
        ("inverse_joint_5000_concat", "concat (posterior)"),
    ]

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.55 * HEIGHT_SCALE))
    gs = fig.add_gridspec(2, 7, width_ratios=[0.50, 1, 1, 1, 1, 1, 1],
                          hspace=0.16, wspace=0.09,
                          left=0.085, right=0.995, top=0.86, bottom=0.02)

    for row, (label, name) in enumerate(rows):
        cached = load_ensemble(label)
        samples = cached["samples"][:, index, channel]
        target = cached["target"][index, channel]
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        ax = fig.add_subplot(gs[row, 0])
        ax.axis("off")
        head, _, tail = name.partition(" ")
        ax.text(0.92, 0.60, head, transform=ax.transAxes, fontsize=7.5,
                color=WARM if row == 0 else ACCENT, fontweight="bold",
                ha="right", va="center")
        ax.text(0.92, 0.40, tail, transform=ax.transAxes, fontsize=6.5,
                color=GRAY, ha="right", va="center")

        limit_low = float(min(target.min(), mean.min()))
        limit_high = float(max(target.max(), mean.max()))

        panels = [
            (target, r"truth $\kappa$", "viridis", (limit_low, limit_high)),
            (samples[0], "sample 1", "viridis", (limit_low, limit_high)),
            (samples[1], "sample 2", "viridis", (limit_low, limit_high)),
            (samples[2], "sample 3", "viridis", (limit_low, limit_high)),
            (mean, "ensemble mean", "viridis", (limit_low, limit_high)),
            (std, "ensemble std.", "magma", None),
        ]
        for col, (field, title, cmap, limits) in enumerate(panels, start=1):
            ax = fig.add_subplot(gs[row, col])
            kwargs = {}
            if limits is not None:
                kwargs = {"vmin": limits[0], "vmax": limits[1]}
            ax.imshow(np.asarray(field), cmap=cmap, origin="lower",
                      interpolation="nearest", **kwargs)
            field_axes(ax)
            if row == 0:
                ax.set_title(title, pad=3, color=GRAY, fontsize=7)

    save(fig, "fig_inverse_samples")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def figure_calibration() -> None:
    """Rank histogram, reliability curve and spread-skill, forward vs inverse."""
    use_report_style()

    hist_rows = read_csv(ANALYSIS / "uq_rank_histogram.csv")
    rel_rows = read_csv(ANALYSIS / "uq_reliability.csv")
    summary = {row["label"]: row for row in read_csv(ANALYSIS / "uq_summary.csv")}

    shown = [
        ("inverse_coefficient_5000_concat", ACCENT),
        ("forward_darcy_convnet", WARM),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.45 * HEIGHT_SCALE),
                             gridspec_kw={"wspace": 0.62, "bottom": 0.20,
                                          "right": 0.965,
                                          "width_ratios": [1.2, 0.95, 0.9]})

    # (a) rank histograms
    ax = axes[0]
    for label, colour in shown:
        selected = [r for r in hist_rows if r["label"] == label]
        selected.sort(key=lambda r: int(r["bin_index"]))
        frequency = np.array([float(r["frequency"]) for r in selected])
        centres = np.arange(len(frequency))
        ax.step(centres, frequency, where="mid", color=colour,
                label=PRETTY[label])
        uniform = float(selected[0]["uniform"])
    ax.axhline(uniform, color=GRAY, linewidth=0.7, linestyle=(0, (4, 3)))
    ax.text(0.02, uniform, "flat = calibrated", fontsize=6.2, color=GRAY,
            va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xlabel("rank of the truth among $K = 32$ members")
    ax.set_ylabel("frequency")
    ax.set_ylim(bottom=0)
    grid(ax)
    ax.legend(loc="upper center", fontsize=6.3, handletextpad=0.4)
    panel_label(ax, "(a)", dx=-0.22)

    # (b) reliability
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=0.7, linestyle=(0, (4, 3)))
    for label, colour in shown:
        selected = [r for r in rel_rows if r["label"] == label]
        selected.sort(key=lambda r: float(r["nominal"]))
        nominal = [float(r["nominal"]) for r in selected]
        empirical = [float(r["empirical"]) for r in selected]
        ax.plot(nominal, empirical, color=colour, label=PRETTY[label])
    ax.set_xlabel("nominal credible level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    grid(ax)
    ax.text(0.70, 0.09, "below: over-confident\nabove: over-dispersed",
            fontsize=6.0, color=GRAY, ha="center")
    panel_label(ax, "(b)", dx=-0.28)

    # (c) spread-skill
    ax = axes[2]
    labels = [label for label in PRETTY if label in summary]
    values = [float(summary[label]["spread_skill_corrected"]) for label in labels]
    colours = [WARM if label.startswith("forward") else ACCENT
               for label in labels]
    positions = np.arange(len(labels))
    ax.barh(positions, values, 0.62, color=colours, linewidth=0)
    ax.axvline(1.0, color=GRAY, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_yticks(positions)
    ax.set_yticklabels([PRETTY[label] for label in labels], fontsize=6.2)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("spread / skill, bias corrected")
    grid(ax, axis="x")
    ax.set_xlim(0, max(1.25, max(values) * 1.12))
    ax.text(1.0, -0.9, "calibrated", fontsize=6.2, color=GRAY, ha="center")
    panel_label(ax, "(c)", dx=-0.95, dy=1.10)

    save(fig, "fig_calibration")


# ─────────────────────────────────────────────────────────────────────────────
# Spatial identifiability
# ─────────────────────────────────────────────────────────────────────────────

def figure_identifiability(index: int = 0, n_bins: int = 12) -> None:
    r"""Posterior width against $|\nabla u|$.

    Section 2.2.2 argues that $\kappa$ enters $-\nabla\cdot(\kappa\nabla u) = f$
    only through the flux, so where $\nabla u$ is small the data barely
    constrain it.  That is a statement about the *spatial* structure of the
    posterior, and it is testable: bin every pixel of every test sample by the
    local $|\nabla u|$ and look at the posterior standard deviation there.
    """
    use_report_style()

    from flowpde.datasets.exponax.darcy import DarcyConfig, DarcyGenerator
    import yaml

    cached = load_ensemble("inverse_coefficient_5000_concat")
    samples = cached["samples"][:, :, 0]           # (K, B, H, W)
    std = samples.std(dim=0)                        # (B, H, W)
    n_samples = std.shape[0]

    run = Path("results/experiments/exp05_inverse_conditioning_ablation"
               "/coefficient_5000/concat")
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    generator = DarcyGenerator(config=DarcyConfig(**config["data"]["generator"]))
    forward = generator.generate(
        num_samples=config["data"]["splits"]["test"],
        seed=config["data"]["splits"]["test_seed"], problem="forward")

    solution = torch.stack([forward[i]["target"][0] for i in range(n_samples)])
    grad_y, grad_x = torch.gradient(solution, dim=(1, 2))
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.15 * HEIGHT_SCALE))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.1, 2.0],
                          wspace=0.30, left=0.02, right=0.985,
                          top=0.86, bottom=0.19)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.asarray(grad_mag[index]), cmap="magma", origin="lower")
    field_axes(ax)
    ax.set_title(r"$|\nabla u|$", color=GRAY, pad=3, fontsize=7.5)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.asarray(std[index]), cmap="magma", origin="lower")
    field_axes(ax)
    ax.set_title(r"posterior std. of $\kappa$", color=GRAY, pad=3, fontsize=7.5)
    fig.add_subplot(gs[0, 2]).axis("off")

    # Bin every pixel of every test sample by local flux magnitude.
    flat_grad = grad_mag.flatten().numpy()
    flat_std = std.flatten().numpy()
    quantiles = np.quantile(flat_grad, np.linspace(0, 1, n_bins + 1))
    centres, medians, lows, highs = [], [], [], []
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        selected = flat_std[(flat_grad >= lo) & (flat_grad < hi)]
        if selected.size == 0:
            continue
        centres.append(0.5 * (lo + hi))
        medians.append(np.median(selected))
        lows.append(np.quantile(selected, 0.25))
        highs.append(np.quantile(selected, 0.75))

    ax = fig.add_subplot(gs[0, 3])
    ax.fill_between(centres, lows, highs, color=PALE, alpha=0.55, linewidth=0)
    ax.plot(centres, medians, "o-", color=ACCENT, markersize=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"local flux magnitude $|\nabla u|$ (pixel decile)")
    ax.set_ylabel(r"posterior std. of $\kappa$")
    ax.set_title(r"all test pixels, median and interquartile range",
                 color=GRAY, pad=5)
    grid(ax)
    panel_label(ax, "(b)", dx=-0.15, dy=1.19)
    fig.text(0.012, 0.93, "(a)", fontsize=8.5, fontweight="bold", color=ACCENT)

    save(fig, "fig_identifiability")


if __name__ == "__main__":
    figure_inverse_samples()
    figure_calibration()
    figure_identifiability()
