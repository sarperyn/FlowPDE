"""Figures for accuracy against sampling cost, and for path straightness.

Both read the CSVs written by ``experiments.report_analysis``.  The cost axis is
function evaluations, never steps: one RK4 step costs four network evaluations,
so plotting against steps would credit RK4 with a factor of four it has not
earned.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.report_figures.style import (
    ACCENT, BACKBONE_COLORS, BACKBONE_LABELS, GRAY, RULE, SOLVER_COLORS,
    HEIGHT_SCALE, SOLVER_LABELS, TEXT_WIDTH, WARM, grid, panel_label, save, use_report_style,
)

ANALYSIS = Path("results/analysis")
BACKBONES = ["convnet_small", "resnet", "unet", "unet_no_attention"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def load_nfe() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for name in ("nfe_sweep_burgers.csv", "nfe_sweep_darcy.csv"):
        path = ANALYSIS / name
        if path.exists():
            rows.extend(read_csv(path))
    return rows


def _curve(rows, pde, variant, solver):
    selected = [r for r in rows
                if r["pde"] == pde and r["variant"] == variant
                and r["solver"] == solver]
    selected.sort(key=lambda r: float(r["nfe"]))
    return (np.array([float(r["nfe"]) for r in selected]),
            np.array([float(r["rel_l2_mean"]) for r in selected]))


def figure_nfe() -> None:
    """Error against function evaluations, per solver and per backbone."""
    use_report_style()
    rows = load_nfe()

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.35 * HEIGHT_SCALE),
                             gridspec_kw={"wspace": 0.34})

    for ax, pde, title, letter in [
        (axes[0], "burgers", "Burgers 1D, ConvNet", "(a)"),
        (axes[1], "darcy", "Darcy 2D, ConvNet", "(b)"),
    ]:
        for solver in ("euler", "midpoint", "rk4"):
            nfe, error = _curve(rows, pde, "convnet_small", solver)
            if len(nfe) == 0:
                continue
            ax.plot(nfe, error, "o-", color=SOLVER_COLORS[solver],
                    markersize=2.8, label=SOLVER_LABELS[solver])

        adaptive = [r for r in rows if r["pde"] == pde
                    and r["variant"] == "convnet_small"
                    and r["solver"] == "dopri5"]
        if adaptive:
            nfe = float(adaptive[0]["nfe"])
            error = float(adaptive[0]["rel_l2_mean"])
            ax.axhline(error, color=GRAY, linewidth=0.7, linestyle=(0, (4, 3)))
            ax.plot([nfe], [error], "*", color=GRAY, markersize=7,
                    label=SOLVER_LABELS["dopri5"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("function evaluations")
        ax.set_title(title, color=GRAY, pad=4)
        grid(ax)
        panel_label(ax, letter, dx=-0.28)

    axes[0].set_ylabel(r"relative $L^2$")
    axes[0].legend(loc="upper right", handletextpad=0.4, borderpad=0.2)

    # (c) Euler only, every backbone: the shape of the curve separates
    # discretization error from model error.  Burgers, because that is where
    # the contrast between a steep and a flat curve is unmistakable.
    ax = axes[2]
    for name in BACKBONES:
        nfe, error = _curve(rows, "burgers", name, "euler")
        if len(nfe) == 0:
            continue
        ax.plot(nfe, error, "o-", color=BACKBONE_COLORS[name], markersize=2.8,
                label=BACKBONE_LABELS[name])
        ax.annotate(f"$\\times${error[0] / error.min():.1f}",
                    xy=(nfe[-1], error[-1]), xytext=(3, -1),
                    textcoords="offset points", fontsize=6,
                    color=BACKBONE_COLORS[name], va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("function evaluations (Euler)")
    ax.set_ylabel(r"relative $L^2$")
    ax.set_title("Burgers 1D, all backbones", color=GRAY, pad=4)
    ax.set_xlim(right=ax.get_xlim()[1] * 2.6)
    ax.set_ylim(bottom=ax.get_ylim()[0] * 0.42)   # room for the legend
    grid(ax)
    ax.legend(loc="lower left", handletextpad=0.4, borderpad=0.2,
              labelspacing=0.25)
    panel_label(ax, "(c)", dx=-0.28)

    save(fig, "fig_nfe")


def figure_nfe_fields(index: int = 12, device: str = "cpu") -> None:
    """The same weights and the same noise, sampled at increasing step budgets.

    The curve in ``fig_nfe`` says the error stops falling early; this says what
    that looks like.  One Darcy test condition, one fixed $\\bx_0$, integrated
    with Euler at increasing budgets, against the truth.
    """
    use_report_style()

    from experiments.report_analysis.common import load_run, sample_dataset

    run = load_run("results/experiments/exp03_backbone_ablation_darcy"
                   "/convnet_small", device=device)
    budgets = [1, 2, 4, 8, 32]
    limit = index + 1

    fields, errors = [], []
    for steps in budgets:
        prediction, target = sample_dataset(
            run, n_steps=steps, solver="euler", batch_size=limit, limit=limit)
        fields.append(prediction[index, 0])
        errors.append(
            (prediction[index] - target[index]).norm() / target[index].norm())
    truth = target[index, 0]

    limit_value = float(max(abs(truth.min()), abs(truth.max())))
    fig, axes = plt.subplots(1, len(budgets) + 1,
                             figsize=(TEXT_WIDTH, 1.42 * HEIGHT_SCALE),
                             gridspec_kw={"wspace": 0.07, "left": 0.005,
                                          "right": 0.995, "top": 0.80,
                                          "bottom": 0.10})

    axes[0].imshow(np.asarray(truth), cmap="RdBu_r", origin="lower",
                   vmin=-limit_value, vmax=limit_value)
    axes[0].set_title("truth", pad=3, color=GRAY, fontsize=7)
    for spine in axes[0].spines.values():
        spine.set_color(ACCENT)
        spine.set_linewidth(0.9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for ax, field, steps, error in zip(axes[1:], fields, budgets, errors):
        ax.imshow(np.asarray(field), cmap="RdBu_r", origin="lower",
                  vmin=-limit_value, vmax=limit_value)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(RULE)
            spine.set_linewidth(0.5)
        ax.set_title(f"{steps} NFE", pad=3, color=GRAY, fontsize=7)
        ax.set_xlabel(f"{float(error):.3f}", fontsize=6.5, color=GRAY,
                      labelpad=1.5)

    save(fig, "fig_nfe_fields")


def _straightness_table():
    """Join straightness, few-step penalty and final accuracy per model."""
    import csv as _csv

    straight = read_csv(ANALYSIS / "straightness.csv")
    rows = load_nfe()
    accuracy = {}
    for pde, folder in (
        ("burgers", "exp02_backbone_ablation_burgers"),
        ("darcy", "exp03_backbone_ablation_darcy"),
    ):
        path = Path("results/experiments") / folder / "summary.csv"
        with path.open() as handle:
            for row in _csv.DictReader(handle):
                accuracy[(pde, row["variant"])] = float(row["test_rel_l2"])

    table = []
    for pde in ("burgers", "darcy"):
        for name in BACKBONES:
            modes = {
                r["mode"]: float(r["normalized_straightness"])
                for r in straight if r["pde"] == pde and r["variant"] == name
            }
            nfe, error = _curve(rows, pde, name, "euler")
            if not modes or len(nfe) == 0:
                continue
            penalty = float(error[nfe == 2][0] / error.min())
            table.append({
                "pde": pde, "variant": name,
                "trajectory": modes.get("trajectory"),
                "interpolant": modes.get("interpolant"),
                "penalty": penalty,
                "rel_l2": accuracy[(pde, name)],
            })
    return table


def figure_straightness() -> None:
    """Straightness across backbones, and what it does and does not predict.

    Path geometry is a property of the *objective* -- the interpolation path,
    the coupling, the time schedule -- and all eight models here share one.  The
    figure is therefore mostly a negative result, and is drawn to show that
    rather than to hide it.
    """
    use_report_style()
    table = _straightness_table()

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4 * HEIGHT_SCALE),
                             gridspec_kw={"wspace": 0.40,
                                          "width_ratios": [1.5, 1, 1],
                                          "bottom": 0.24})

    # (a) both modes, every model
    ax = axes[0]
    width = 0.38
    positions = np.arange(len(table))
    for offset, (mode, colour) in enumerate(
        [("trajectory", ACCENT), ("interpolant", WARM)]
    ):
        ax.bar(positions + (offset - 0.5) * width,
               [row[mode] for row in table], width * 0.9,
               color=colour, linewidth=0, label=mode)
    ax.set_xticks(positions)
    ax.set_xticklabels([BACKBONE_LABELS[row["variant"]] for row in table],
                       fontsize=5.6, rotation=32, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("normalized straightness")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)          # room for the legend
    ax.axvline(3.5, color=RULE, linewidth=0.6)
    ax.text(0.22, -0.46, "Burgers 1D", transform=ax.transAxes, fontsize=6.5,
            color=ACCENT, ha="center")
    ax.text(0.78, -0.46, "Darcy 2D", transform=ax.transAxes, fontsize=6.5,
            color=ACCENT, ha="center")
    ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2,
              labelspacing=0.2, ncol=2, columnspacing=0.8)
    grid(ax, axis="y")
    panel_label(ax, "(a)", dx=-0.15)

    # (b, c) what each mode predicts
    for ax, (xkey, ykey, xlabel, ylabel, letter) in zip(
        axes[1:],
        [("trajectory", "penalty", "trajectory straightness",
          "error at 2 NFE / best", "(b)"),
         ("interpolant", "rel_l2", "interpolant straightness",
          r"test relative $L^2$", "(c)")],
    ):
        for row in table:
            ax.scatter(row[xkey], row[ykey],
                       marker="o" if row["pde"] == "burgers" else "s", s=24,
                       color=BACKBONE_COLORS[row["variant"]], linewidths=0)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        if ykey == "rel_l2":
            ax.set_yscale("log")
        grid(ax)
        panel_label(ax, letter, dx=-0.34)

        xs = np.array([row[xkey] for row in table])
        ys = np.array([row[ykey] for row in table])
        ranks = lambda v: np.argsort(np.argsort(v))
        rho = np.corrcoef(ranks(xs), ranks(ys))[0, 1]
        ax.text(0.04, 0.94, rf"$\rho = {rho:+.2f}$", transform=ax.transAxes,
                fontsize=6.8, color=GRAY, va="top")

    axes[1].text(0.5, -0.44, "circles: Burgers      squares: Darcy",
                 transform=axes[1].transAxes, ha="center", fontsize=6.5,
                 color=GRAY)

    save(fig, "fig_straightness")


if __name__ == "__main__":
    figure_nfe()
    figure_straightness()
