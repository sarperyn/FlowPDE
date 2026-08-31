"""Figures describing the benchmarks themselves.

``fig_benchmarks``  — what each problem maps to what, in both directions.
``fig_illposed``    — why inverting an elliptic operator is harder than
                      applying it, shown spectrally rather than asserted.

Both regenerate their data from the seeded generators, so the fields shown are
the actual test-split fields the models were scored on, not decorative
stand-ins.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.report_figures.style import (
    ACCENT, GRAY, RULE, WARM, TEXT_WIDTH,
    HEIGHT_SCALE, field_axes, grid, panel_label, save, use_report_style,
)
from flowpde.datasets.exponax.burgers import BurgersConfig, BurgersGenerator
from flowpde.datasets.exponax.darcy import DarcyConfig, DarcyGenerator
from flowpde.datasets.exponax.poisson import PoissonConfig, PoissonGenerator

CMAP = "RdBu_r"
CMAP_POS = "viridis"


def _show(ax, field, cmap=CMAP, symmetric=True, title=None):
    array = np.asarray(field)
    if symmetric:
        limit = np.abs(array).max()
        kwargs = {"vmin": -limit, "vmax": limit}
    else:
        kwargs = {}
    ax.imshow(array, cmap=cmap, origin="lower", interpolation="nearest", **kwargs)
    field_axes(ax)
    if title:
        ax.set_title(title, pad=3, color=GRAY)


def _line(ax, values, color=ACCENT, title=None, label=None):
    ax.plot(np.asarray(values), color=color, linewidth=1.0, label=label)
    ax.set_xticks([])
    ax.set_yticks([])
    for key, spine in ax.spines.items():
        spine.set_visible(key in ("bottom", "left"))
        spine.set_color(RULE)
    if title:
        ax.set_title(title, pad=3, color=GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# Figure: benchmark gallery
# ─────────────────────────────────────────────────────────────────────────────

def figure_benchmarks(index: int = 3) -> None:
    """One row per benchmark: conditioning channels, then the target."""
    use_report_style()

    poisson = PoissonGenerator(config=PoissonConfig(
        num_spatial_dims=2, num_points=64, num_samples=8, seed=303,
        torch_device="cpu",
    )).generate(num_samples=8, seed=303, problem="forward")

    burgers = BurgersGenerator(config=BurgersConfig(
        num_spatial_dims=1, num_points=64, num_samples=8, seed=303,
        torch_device="cpu", dt=0.001, num_steps=400,
        diffusivity_min=0.01, diffusivity_max=0.01,
        ic_num_terms=3, ic_max_mode=4,
    )).generate(num_samples=8, seed=303, problem="forward")

    darcy_cfg = DarcyConfig(
        num_spatial_dims=2, num_points=64, num_samples=8, seed=303,
        torch_device="cpu", kappa_alpha=2.0, kappa_tau=3.0, kappa_scale=1.0,
        kappa_min=0.1, f_cutoff=8, f_amplitude_min=0.1, f_amplitude_max=5.0,
        cg_steps=500,
    )
    darcy_fwd = DarcyGenerator(config=darcy_cfg).generate(
        num_samples=8, seed=303, problem="forward")

    darcy_inv_cfg = DarcyConfig(**{**darcy_cfg.__dict__,
                                   "obs_noise_std": 2e-4,
                                   "obs_mask_fraction": 0.15})
    darcy_inv = DarcyGenerator(config=darcy_inv_cfg).generate(
        num_samples=8, seed=303, problem="inverse", inverse_mode="coefficient")

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.35 * HEIGHT_SCALE))
    gs = fig.add_gridspec(
        4, 6, width_ratios=[0.92, 1, 1, 1, 0.34, 1],
        hspace=0.46, wspace=0.14, left=0.008, right=0.995, top=0.945, bottom=0.035,
    )

    rows = [
        ("Poisson", "forward", r"$f \mapsto u$"),
        ("Burgers", "forward", r"$u(\cdot,0) \mapsto u(\cdot,T)$"),
        ("Darcy", "forward", r"$(\kappa, f) \mapsto u$"),
        ("Darcy", "inverse", r"$(u_{\mathrm{obs}}, f, m) \mapsto \kappa$"),
    ]

    def row_label(row):
        name, direction, mapping = rows[row]
        ax = fig.add_subplot(gs[row, 0])
        ax.axis("off")
        ax.text(0.0, 0.74, name, transform=ax.transAxes, fontsize=8.5,
                color=ACCENT, fontweight="bold", ha="left", va="center")
        ax.text(0.0, 0.50, direction, transform=ax.transAxes, fontsize=7.5,
                color=WARM if direction == "inverse" else GRAY,
                ha="left", va="center")
        ax.text(0.0, 0.24, mapping, transform=ax.transAxes, fontsize=7,
                color=GRAY, ha="left", va="center")

    def arrow(row, reverse=False):
        ax = fig.add_subplot(gs[row, 4])
        ax.axis("off")
        colour = WARM if reverse else ACCENT
        start, end = ((0.95, 0.05) if reverse else (0.05, 0.95))
        ax.annotate("", xy=(end, 0.46), xytext=(start, 0.46),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.9))
        ax.text(0.5, 0.66, r"$\mathcal{S}^{-1}$" if reverse else r"$\mathcal{S}$",
                transform=ax.transAxes, ha="center", fontsize=8, color=colour)

    def blanks(row, columns):
        for col in columns:
            fig.add_subplot(gs[row, col]).axis("off")

    # Row 0 — Poisson
    sample = poisson[index]
    row_label(0)
    _show(fig.add_subplot(gs[0, 1]), sample["input"][0], title=r"source $f$")
    blanks(0, (2, 3))
    arrow(0)
    _show(fig.add_subplot(gs[0, 5]), sample["target"][0], title=r"solution $u$")

    # Row 1 — Burgers (1-D: draw as curves)
    sample = burgers[index]
    row_label(1)
    _line(fig.add_subplot(gs[1, 1]), sample["input"][0], title=r"initial $u(\cdot,0)$")
    blanks(1, (2, 3))
    arrow(1)
    _line(fig.add_subplot(gs[1, 5]), sample["target"][0], color=WARM,
          title=r"final $u(\cdot,T)$")

    # Row 2 — Darcy forward
    sample = darcy_fwd[index]
    row_label(2)
    _show(fig.add_subplot(gs[2, 1]), sample["input"][0], cmap=CMAP_POS,
          symmetric=False, title=r"coefficient $\kappa$")
    _show(fig.add_subplot(gs[2, 2]), sample["input"][1], title=r"source $f$")
    blanks(2, (3,))
    arrow(2)
    _show(fig.add_subplot(gs[2, 5]), sample["target"][0], title=r"solution $u$")

    # Row 3 — Darcy inverse
    sample = darcy_inv[index]
    row_label(3)
    _show(fig.add_subplot(gs[3, 1]), sample["input"][0],
          title=r"observed $u_{\mathrm{obs}}$")
    _show(fig.add_subplot(gs[3, 2]), sample["input"][1], title=r"source $f$")
    _show(fig.add_subplot(gs[3, 3]), sample["input"][2], cmap="Greys_r",
          symmetric=False, title=r"mask $m$")
    arrow(3, reverse=True)
    _show(fig.add_subplot(gs[3, 5]), sample["target"][0], cmap=CMAP_POS,
          symmetric=False, title=r"coefficient $\kappa$")

    save(fig, "fig_benchmarks")


# ─────────────────────────────────────────────────────────────────────────────
# Figure: why the inverse direction is harder
# ─────────────────────────────────────────────────────────────────────────────

def figure_illposed(index: int = 3, cutoff: int = 5) -> None:
    """The elliptic operator attenuates like $|k|^{-2}$; inversion amplifies.

    Left: a source and its solution, split into low- and high-wavenumber parts.
    Right: the measured spectral gain over a whole dataset, against the
    theoretical $|k|^{-2}$.
    """
    use_report_style()

    generator = PoissonGenerator(config=PoissonConfig(
        num_spatial_dims=2, num_points=64, num_samples=256, seed=303,
        torch_device="cpu", source_num_terms=8, source_max_mode=8,
    ))
    dataset = generator.generate(num_samples=256, seed=303, problem="forward")

    sources = torch.stack([dataset[i]["input"][0] for i in range(256)])
    solutions = torch.stack([dataset[i]["target"][0] for i in range(256)])

    n = sources.shape[-1]
    freq = torch.fft.fftfreq(n) * n
    kx, ky = torch.meshgrid(freq, freq, indexing="ij")
    k_mag = torch.sqrt(kx ** 2 + ky ** 2)

    def band(field, keep_low):
        spectrum = torch.fft.fft2(field)
        mask = (k_mag <= cutoff) if keep_low else (k_mag > cutoff)
        return torch.fft.ifft2(spectrum * mask).real

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.62 * HEIGHT_SCALE))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.30, 2.35],
                          hspace=0.34, wspace=0.10,
                          left=0.062, right=0.985, top=0.845, bottom=0.155)

    source = sources[index]
    solution = solutions[index]

    for row, (field, name, cmap) in enumerate(
        [(source, r"source $f$", CMAP), (solution, r"solution $u$", CMAP)]
    ):
        _show(fig.add_subplot(gs[row, 0]), field, cmap=cmap, title=name)
        low = band(field, True)
        high = band(field, False)
        ax_low = fig.add_subplot(gs[row, 1])
        _show(ax_low, low, cmap=cmap, title=r"$|k| \leq 5$" if row == 0 else r"$|k| \leq 5$")
        ax_high = fig.add_subplot(gs[row, 2])
        _show(ax_high, high, cmap=cmap, title=r"$|k| > 5$")
        energy = (high.pow(2).sum() / field.pow(2).sum()).item()
        ax_high.set_xlabel(f"{100 * energy:.1f}\\% of energy", fontsize=6.5,
                           color=GRAY, labelpad=1.5)
        fig.add_subplot(gs[row, 3]).axis("off")

    for row, name in enumerate([r"source", r"solution"]):
        fig.text(0.004, 0.615 - 0.435 * row, name, fontsize=7.5, color=ACCENT,
                 fontweight="bold", rotation=90, va="center")

    # Spectral gain, measured
    ax = fig.add_subplot(gs[:, 4])
    src_spec = torch.fft.fft2(sources).abs()
    sol_spec = torch.fft.fft2(solutions).abs()
    flat_k = k_mag.flatten()
    # Truncate at the source's spectral support: beyond max_mode there is
    # essentially no energy in f, and the ratio becomes numerical noise.
    bins = torch.arange(1, 14)
    centres, gains = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (flat_k >= lo) & (flat_k < hi)
        if sel.sum() < 4:
            continue
        num = sol_spec.flatten(1)[:, sel].mean()
        den = src_spec.flatten(1)[:, sel].mean()
        if den <= 0:
            continue
        centres.append(((lo + hi) / 2).item())
        gains.append((num / den).item())

    centres_np = np.array(centres)
    gains_np = np.array(gains)
    ax.loglog(centres_np, gains_np, "o", color=ACCENT, markersize=3.2,
              markeredgewidth=0, label="measured gain")
    reference = gains_np[0] * (centres_np / centres_np[0]) ** -2.0
    ax.loglog(centres_np, reference, "--", color=WARM, linewidth=1.0,
              label=r"$|k|^{-2}$")
    grid(ax)
    ax.set_xlabel(r"wavenumber $|k|$")
    ax.set_ylabel(r"gain $\;|\hat{u}| / |\hat{f}|$")
    ax.legend(loc="upper right")
    ax.set_title(r"Measured attenuation of $\nabla^{-2}$", color=GRAY, pad=4)
    panel_label(ax, "(b)", dx=-0.19)
    fig.text(0.052, 0.945, "(a)", fontsize=8.5, fontweight="bold", color=ACCENT)

    save(fig, "fig_illposed")


if __name__ == "__main__":
    figure_benchmarks()
    figure_illposed()
