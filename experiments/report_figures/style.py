"""Shared plotting style for the report's figures.

The palette is the one defined in ``report/main.tex`` so that figures and TikZ
diagrams do not disagree with each other, and the type is Computer Modern so
figure labels match body text.  Every figure is sized against the document's
measured text width (455.24 pt), never scaled by ``includegraphics``, because
rescaling a figure rescales its fonts away from the body size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Palette (report/main.tex) ────────────────────────────────────────────────
ACCENT = "#1F4E79"
ACCENT_LIGHT = "#3A7CA5"
GRAY = "#4A4A4A"
RULE = "#D0D5DA"
WARM = "#C1481E"
PALE = "#9CC3D5"

# Categorical sequence: distinguishable in greyscale as well as in colour,
# ordered so the first two are the two that most often appear alone.
SERIES = [ACCENT, WARM, ACCENT_LIGHT, "#6B8F3A", "#8C6BB1", GRAY]

# Fixed backbone → colour map, so a backbone keeps its colour across figures.
BACKBONE_COLORS = {
    "convnet_small": ACCENT,
    "resnet": WARM,
    "unet": ACCENT_LIGHT,
    "unet_no_attention": "#6B8F3A",
}
BACKBONE_LABELS = {
    "convnet_small": "ConvNet",
    "resnet": "ResNet",
    "unet": "UNet",
    "unet_no_attention": "UNet (no attn.)",
}

SOLVER_COLORS = {
    "euler": ACCENT,
    "midpoint": ACCENT_LIGHT,
    "rk4": WARM,
    "dopri5": GRAY,
}
SOLVER_LABELS = {
    "euler": "Euler",
    "midpoint": "midpoint",
    "rk4": "RK4",
    "dopri5": "dopri5 (adaptive)",
}

# Document text width, in inches (455.24411 pt / 72.27).
TEXT_WIDTH = 6.30
HEIGHT_SCALE = 0.74   # compact layout for the 20-page build
FIG_DIR = Path("report/figures")


def use_report_style() -> None:
    """Install the report's rcParams. Idempotent."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,       # cmr10 has no U+2212
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 9,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": GRAY,
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": RULE,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "lines.markersize": 4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    })


def grid(ax, axis: str = "both") -> None:
    """A grid that sits behind the data and does not compete with it."""
    ax.grid(True, axis=axis, color=RULE, linewidth=0.4, alpha=0.9)
    ax.set_axisbelow(True)


def panel_label(ax, text: str, dx: float = -0.06, dy: float = 1.06) -> None:
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", color=ACCENT, va="top", ha="left")


def field_axes(ax) -> None:
    """Strip an axis down to a bare field image."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RULE)
        spine.set_linewidth(0.5)


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    """Write a figure as PDF into ``report/figures``."""
    out_dir = Path(out_dir) if out_dir is not None else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def ecdf(values: Iterable[float]):
    """Empirical CDF, ready to step-plot."""
    ordered = sorted(values)
    n = len(ordered)
    return ordered, [(i + 1) / n for i in range(n)]
