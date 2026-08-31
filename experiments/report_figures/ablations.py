"""Figures for the controlled ablations: conditioning, backbone, objective.

All of these read result CSVs that the original runs wrote.  Where per-sample
errors were saved the figures show the *distribution* rather than its mean,
because with a single training run per cell the paired per-sample spread is the
only honest uncertainty available.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.report_figures.style import (
    ACCENT, ACCENT_LIGHT, BACKBONE_COLORS, BACKBONE_LABELS, GRAY, PALE, RULE,
    HEIGHT_SCALE, TEXT_WIDTH, WARM, ecdf, grid, panel_label, save, use_report_style,
)

RESULTS = Path("results/experiments")
EXP01 = RESULTS / "exp01_conditioning_ablation"
EXP02 = RESULTS / "exp02_backbone_ablation_burgers"
EXP03 = RESULTS / "exp03_backbone_ablation_darcy"
EXP04 = RESULTS / "exp04_objective_ablation_darcy"
EXP05 = RESULTS / "exp05_inverse_conditioning_ablation"

BACKBONES = ["convnet_small", "resnet", "unet", "unet_no_attention"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def column(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows])


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning
# ─────────────────────────────────────────────────────────────────────────────

def figure_conditioning() -> None:
    """Paired per-sample comparison of the null and concat conditioners."""
    use_report_style()
    rows = read_csv(EXP01 / "figures" / "paired_test_errors.csv")
    null = column(rows, "null_rel_l2")
    concat = column(rows, "concat_rel_l2")

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.35 * HEIGHT_SCALE),
                             gridspec_kw={"width_ratios": [1.05, 1],
                                          "wspace": 0.29})

    # (a) paired scatter
    ax = axes[0]
    limits = [min(null.min(), concat.min()) * 0.7,
              max(null.max(), concat.max()) * 1.4]
    ax.plot(limits, limits, color=GRAY, linewidth=0.7, linestyle=(0, (4, 3)),
            zorder=1)
    ax.scatter(concat, null, s=7, color=ACCENT, alpha=0.55,
               linewidths=0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"concat conditioner, relative $L^2$")
    ax.set_ylabel(r"null conditioner, relative $L^2$")
    grid(ax)
    ax.text(0.05, 0.93,
            f"concat better\non {int((concat < null).sum())}/{len(null)} samples",
            transform=ax.transAxes, fontsize=7, color=ACCENT, va="top")
    panel_label(ax, "(a)", dx=-0.20)

    # (b) distributions
    ax = axes[1]
    for values, colour, label in [(null, WARM, "null"),
                                  (concat, ACCENT, "concat")]:
        xs, ys = ecdf(values)
        ax.step(xs, ys, where="post", color=colour, label=label)
        ax.axvline(np.median(values), color=colour, linewidth=0.6,
                   linestyle=(0, (2, 2)), alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"relative $L^2$")
    ax.set_ylabel("empirical CDF")
    ax.set_ylim(0, 1.02)
    grid(ax)
    ax.legend(loc="center left", title="conditioner", title_fontsize=7)
    ax.text(0.97, 0.06,
            f"median ratio\n{np.median(null / concat):.1f}$\\times$",
            transform=ax.transAxes, fontsize=7, color=GRAY, ha="right")
    panel_label(ax, "(b)", dx=-0.19)

    save(fig, "fig_conditioning")


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────

def figure_backbone() -> None:
    """Accuracy against model size on both forward problems, plus the
    per-sample distribution on Darcy where paired data was saved."""
    use_report_style()

    burgers = {row["variant"]: row for row in read_csv(EXP02 / "summary.csv")}
    darcy = {row["variant"]: row for row in read_csv(EXP03 / "summary.csv")}
    per_sample = read_csv(EXP03 / "figures" / "per_sample_test_errors.csv")

    # The one seed replicate available: exp01/concat and exp03/unet are the
    # same configuration (same data block, same UNet with attention, 500
    # epochs) at seeds 1042 and 2042.
    exp01 = {row["variant"]: row for row in read_csv(EXP01 / "summary.csv")}
    unet_seed_a = float(exp01["concat"]["test_rel_l2"])
    unet_seed_b = float(darcy["unet"]["test_rel_l2"])

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.25 * HEIGHT_SCALE),
                             gridspec_kw={"wspace": 0.38})

    for ax, table, title, letter in [
        (axes[0], burgers, "Burgers 1D", "(a)"),
        (axes[1], darcy, "Darcy 2D", "(b)"),
    ]:
        for name in BACKBONES:
            row = table[name]
            ax.scatter(float(row["parameter_count"]), float(row["test_rel_l2"]),
                       s=26, color=BACKBONE_COLORS[name], zorder=3,
                       linewidths=0, label=BACKBONE_LABELS[name])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_title(title, color=GRAY, pad=4)
        grid(ax)
        panel_label(ax, letter, dx=-0.26)

        # Log axes with a narrow range produce unreadable default ticks
        # ("$9\\times10^{-2}$" next to "$10^{-1}$"); label them as plain numbers.
        counts = [float(table[name]["parameter_count"]) for name in BACKBONES]
        ticks = [3e5, 1e6, 3e6, 6e6]
        ticks = [t for t in ticks if min(counts) / 2 < t < max(counts) * 2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t / 1e6:g}M" for t in ticks])
        ax.set_xticks([], minor=True)

        errors = [float(table[name]["test_rel_l2"]) for name in BACKBONES]
        candidates = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
        y_ticks = [t for t in candidates
                   if min(errors) * 0.85 < t < max(errors) * 1.15]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{t:g}" for t in y_ticks])
        ax.set_yticks([], minor=True)

    axes[0].set_ylabel(r"relative $L^2$")

    # Seed band on the Darcy panel: what a *rerun* of one configuration moves.
    ax = axes[1]
    low, high = sorted((unet_seed_a, unet_seed_b))
    ax.axhspan(low, high, color=PALE, alpha=0.55, zorder=1, linewidth=0)
    ax.annotate(
        "same UNet,\ntwo seeds",
        xy=(float(darcy["unet"]["parameter_count"]), high),
        xytext=(0.30, 0.82), textcoords="axes fraction",
        fontsize=6.5, color=ACCENT_LIGHT, ha="center",
        arrowprops=dict(arrowstyle="-", color=ACCENT_LIGHT, lw=0.5),
    )
    axes[0].legend(loc="lower right", handletextpad=0.3, borderpad=0.2)

    # (c) per-sample distribution, Darcy
    ax = axes[2]
    for name in BACKBONES:
        values = column(per_sample, f"{name}_rel_l2")
        xs, ys = ecdf(values)
        ax.step(xs, ys, where="post", color=BACKBONE_COLORS[name],
                label=BACKBONE_LABELS[name])
    ax.set_xscale("log")
    ax.set_xlabel(r"relative $L^2$ (per test sample)")
    ax.set_ylabel("empirical CDF")
    ax.set_ylim(0, 1.02)
    ax.set_title("Darcy 2D, 200 samples", color=GRAY, pad=4)
    grid(ax)
    panel_label(ax, "(c)", dx=-0.26)

    convnet = column(per_sample, "convnet_small_rel_l2")
    unet = column(per_sample, "unet_rel_l2")
    ax.text(0.03, 0.94,
            f"ConvNet below UNet\non {int((convnet < unet).sum())}/{len(unet)}",
            transform=ax.transAxes, fontsize=6.5, color=ACCENT, va="top")

    save(fig, "fig_backbone")


def figure_backbone_diagnostics() -> None:
    """Where the error sits: physics-targeted diagnostics per backbone."""
    use_report_style()

    darcy = {row["variant"]: row
             for row in read_csv(EXP03 / "figures" / "darcy_diagnostics.csv")}
    burgers = {row["variant"]: row
               for row in read_csv(EXP02 / "figures" / "burgers_diagnostics.csv")}

    darcy_keys = [
        ("rel_l2", r"rel. $L^2$"),
        ("h1", r"rel. $H^1$"),
        ("source_region_rel_l2", "high $|f|$"),
        ("high_kappa_rel_l2", r"high $\kappa$"),
        ("rel_max", r"rel. $L^\infty$"),
    ]
    burgers_keys = [
        ("rel_l2", r"rel. $L^2$"),
        ("gradient_rel_l2", r"$\partial_x u$"),
        ("shock_rel_l2", "shock region"),
        ("spectral_low", "low $|k|$"),
        ("spectral_mid", "mid $|k|$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.15 * HEIGHT_SCALE),
                             gridspec_kw={"wspace": 0.22})

    for ax, table, keys, title, letter in [
        (axes[0], darcy, darcy_keys, "Darcy 2D", "(a)"),
        (axes[1], burgers, burgers_keys, "Burgers 1D", "(b)"),
    ]:
        width = 0.2
        positions = np.arange(len(keys))
        for offset, name in enumerate(BACKBONES):
            values = [float(table[name][key]) for key, _ in keys]
            ax.bar(positions + (offset - 1.5) * width, values, width * 0.92,
                   color=BACKBONE_COLORS[name], label=BACKBONE_LABELS[name],
                   linewidth=0)
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels([label for _, label in keys], fontsize=6.8)
        ax.set_title(title, color=GRAY, pad=4)
        grid(ax, axis="y")
        panel_label(ax, letter, dx=-0.13)

    axes[0].set_ylabel("relative error")
    for ax in axes:
        low, high = ax.get_ylim()
        ax.set_ylim(low, high * 4.0)      # headroom for the legend
        ax.tick_params(axis="x", length=0)
    axes[1].legend(loc="upper left", ncol=2, handletextpad=0.3,
                   columnspacing=0.9, borderpad=0.2)

    save(fig, "fig_backbone_diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def figure_objective() -> None:
    """Flow matching against maximum likelihood on an identical flow."""
    use_report_style()

    flow_matching = json.loads(
        (EXP04 / "flow_matching" / "metrics.json").read_text())
    mle = json.loads((EXP04 / "mle_hutchinson_1" / "metrics.json").read_text())
    runs = [("flow matching", flow_matching, ACCENT),
            ("MLE (Hutchinson)", mle, WARM)]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.15 * HEIGHT_SCALE),
                             gridspec_kw={"wspace": 0.42})

    panels = [
        ("seconds_per_epoch", "seconds per epoch", True),
        ("test_rel_l2", r"test relative $L^2$", False),
        ("test_nll_per_dim", "test NLL per dimension", False),
    ]

    for ax, (key, label, log), letter in zip(axes, panels, ("(a)", "(b)", "(c)")):
        values = [run[key] for _, run, _ in runs]
        colours = [colour for _, _, colour in runs]
        bars = ax.bar([0, 1], values, 0.52, color=colours, linewidth=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["FM", "MLE"], fontsize=7.5)
        ax.set_ylabel(label)
        if log:
            ax.set_yscale("log")
        grid(ax, axis="y")
        panel_label(ax, letter, dx=-0.34)

        for bar, value in zip(bars, values):
            below = value < 0
            ax.annotate(
                f"{value:.4g}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, -9 if below else 2.5), textcoords="offset points",
                ha="center", va="top" if below else "bottom",
                fontsize=6.8, color=GRAY,
            )

    axes[0].annotate(
        f"$\\times${mle['seconds_per_epoch'] / flow_matching['seconds_per_epoch']:.0f}",
        xy=(0.5, 0.55), xycoords="axes fraction", ha="center",
        fontsize=8, color=WARM,
    )
    for ax in (axes[1], axes[2]):
        ax.margins(y=0.22)

    save(fig, "fig_objective")


# ─────────────────────────────────────────────────────────────────────────────
# Inverse problem, point metrics
# ─────────────────────────────────────────────────────────────────────────────

INVERSE_CELLS = [
    ("coefficient_3000", r"$\kappa$", 3000),
    ("coefficient_5000", r"$\kappa$", 5000),
    ("joint_3000", r"$(\kappa, f)$", 3000),
    ("joint_5000", r"$(\kappa, f)$", 5000),
]


def figure_inverse_point() -> None:
    """The inverse ablation read against the prior baseline.

    A null-conditioned model samples the marginal prior, so its score is what
    "no information was used" looks like on this benchmark — which is the only
    thing that makes a relative $L^2$ near 1 interpretable.

    Both panels come from the K = 32 re-evaluation rather than from the original
    run's plotting output, so the single-draw error and the ensemble spread are
    the same draw scored two ways.
    """
    use_report_style()

    summary = {row["label"]: row
               for row in read_csv(Path("results/analysis/uq_summary.csv"))}

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5 * HEIGHT_SCALE),
                             gridspec_kw={"wspace": 0.26, "bottom": 0.245,
                                          "top": 0.90})

    width = 0.36
    positions = np.arange(len(INVERSE_CELLS))
    labels = [f"{target}\n{n} train" for _, target, n in INVERSE_CELLS]

    for ax, (metric, ylabel, letter) in zip(
        axes,
        [("single_draw_rel_l2", r"relative $L^2$ of a single draw", "(a)"),
         ("ensemble_spread", r"ensemble spread of the target", "(b)")],
    ):
        for offset, (variant, colour) in enumerate(
            [("null", WARM), ("concat", ACCENT)]
        ):
            values, missing = [], []
            for index, (tag, _, _) in enumerate(INVERSE_CELLS):
                key = f"inverse_{tag.replace('_3000', '').replace('_5000', '')}"
                key = ("inverse_"
                       + tag.rsplit("_", 1)[0]
                       + f"_{tag.rsplit('_', 1)[1]}_{variant}")
                if key in summary:
                    values.append(float(summary[key][metric]))
                else:
                    values.append(0.0)
                    missing.append(index)
            ax.bar(positions + (offset - 0.5) * width, values, width * 0.92,
                   color=colour, linewidth=0,
                   label="null (prior)" if variant == "null" else "concat")
            for index in missing:
                ax.text(positions[index] + (offset - 0.5) * width, 0.03,
                        "no ckpt.", rotation=90, fontsize=5.6, color=GRAY,
                        ha="center", va="bottom")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_ylabel(ylabel)
        grid(ax, axis="y")
        panel_label(ax, letter, dx=-0.17)
        ax.tick_params(axis="x", length=0)

    axes[0].axhline(1.0, color=GRAY, linewidth=0.9, linestyle=(0, (4, 3)),
                    zorder=4)
    axes[0].set_ylim(0, 1.46)
    axes[0].annotate(
        "prior level", xy=(3.40, 1.0), xytext=(3.40, 1.19),
        fontsize=6.5, color=GRAY, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.5),
    )
    axes[0].legend(loc="upper left", ncol=2, handletextpad=0.3,
                   columnspacing=0.9, borderpad=0.2)
    axes[1].set_ylim(0, 2.35)

    save(fig, "fig_inverse_point")


if __name__ == "__main__":
    figure_conditioning()
    figure_backbone()
    figure_backbone_diagnostics()
    figure_objective()
    figure_inverse_point()
