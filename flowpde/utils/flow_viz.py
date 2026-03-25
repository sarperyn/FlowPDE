import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from typing import Optional


def plot_curve(
    epoch_losses,
    title: str = "Training Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: str = "loss_curve.png",
) -> None:
    """Plot and save a training loss curve.

    Args:
        epoch_losses: Sequence of per-epoch loss values.
        title:        Figure title and legend label.
        xlabel:       X-axis label.
        ylabel:       Y-axis label.
        save_path:    File path for the saved figure (PNG/PDF/SVG).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    epochs = np.arange(1, len(epoch_losses) + 1)
    ax.plot(epochs, epoch_losses, color="#1f77b4", linewidth=2.2, alpha=0.9, label=title)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to {save_path}")


@torch.no_grad()
def visualize_flow_evolution(
    model,
    dataloader,
    fig_path: str,
    n_steps: int = 10,
    n_samples: int = 4,
    device: str = "cpu",
    cmap: str = "viridis",
    show_ground_truth: bool = True,
    integration_method: str = "euler",
    figsize_per_col: float = 2.5,
    title_fontsize: int = 9,
    dpi: int = 200,
) -> None:
    """Visualise the flow matching trajectory from noise (t=0) to solution (t=1).

    Produces a grid figure where each row is one sample and each column is one
    snapshot in time.  The final column (optional) shows the ground-truth
    solution for comparison.

    Supports **1-D** (line plot) and **2-D** (imshow) spatial fields.
    The spatial shape is inferred from the target tensor ``u``; no assumption
    about square grids is made.

    Args:
        model:             Velocity-field model with signature
                           ``model(x, condition, t) → velocity``.
                           ``x`` and ``condition`` keep their original spatial
                           shape (B, C, *spatial).  ``t`` is shape ``(B,)``.
        dataloader:        DataLoader yielding dicts with keys ``'u'`` and
                           ``'f'``, or tuples ``(u, f)``.
        fig_path:          Output path for the saved figure.
        n_steps:           Number of intermediate snapshots (columns between
                           t=0 and t=1).
        n_samples:         Number of samples (rows).  Clamped to batch size.
        device:            Torch device string.
        cmap:              Matplotlib colormap name.
        show_ground_truth: Append a ground-truth column.
        integration_method: ``'euler'`` or ``'midpoint'``.
        figsize_per_col:   Figure width / height per column in inches.
        title_fontsize:    Font size for column titles.
        dpi:               Figure DPI.
    """
    model.eval()

    # ── Fetch one batch ───────────────────────────────────────────────────────
    batch = next(iter(dataloader))
    if isinstance(batch, dict):
        u = batch["u"].float().to(device)   # (B, C_u, *spatial)
        f = batch["f"].float().to(device)   # (B, C_f, *spatial)
    else:
        u, f = batch
        u = u.float().to(device)
        f = f.float().to(device)

    n_samples = min(n_samples, u.shape[0])
    u = u[:n_samples]
    f = f[:n_samples]

    spatial_dims = u.ndim - 2        # number of spatial axes (1 or 2)
    spatial_shape = u.shape[2:]      # e.g. (N,) or (H, W)

    # ── Initial noise — same shape as the target ──────────────────────────────
    x_0 = torch.randn_like(u)        # (n_samples, C_u, *spatial)

    # ── Time grid ─────────────────────────────────────────────────────────────
    t_vals = torch.linspace(0.0, 1.0, n_steps + 2, device=device)  # includes 0 and 1
    n_cols = len(t_vals) + (1 if show_ground_truth else 0)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_samples,
        n_cols,
        figsize=(figsize_per_col * n_cols, figsize_per_col * n_samples),
        squeeze=False,
    )

    def _to_numpy(tensor):
        """(C, *spatial) → 2-D array suitable for imshow / plot."""
        arr = tensor[0].cpu().numpy()   # take first channel
        if spatial_dims == 1:
            return arr                  # shape (N,)
        return arr                      # shape (H, W) for imshow

    def _plot(ax, arr, title, row):
        if spatial_dims == 1:
            ax.plot(arr, linewidth=1.2)
            ax.set_ylim(-3, 3)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.imshow(arr, cmap=cmap, aspect="auto")
            ax.axis("off")
        if row == 0:
            ax.set_title(title, fontsize=title_fontsize)

    # ── Integrate each sample step-by-step ────────────────────────────────────
    for i in range(n_samples):
        x_t = x_0[i : i + 1]           # (1, C_u, *spatial)
        cond = f[i : i + 1]             # (1, C_f, *spatial)

        col = 0
        for j, t in enumerate(t_vals):
            # Render current state before stepping
            _plot(axes[i, col], _to_numpy(x_t.squeeze(0)), _fmt_t(t, j, len(t_vals)), i)
            col += 1

            if j < len(t_vals) - 1:
                dt = t_vals[j + 1] - t_vals[j]
                t_scalar = t_vals[j]
                t_batch = t_scalar.expand(1)         # (1,)

                if integration_method == "midpoint":
                    t_mid_batch = (t_vals[j] + t_vals[j + 1]).unsqueeze(0) / 2
                    v_mid = model(x_t + 0.5 * dt * model(x_t, cond, t_batch), cond, t_mid_batch)
                    x_t = x_t + dt * v_mid
                else:  # euler
                    v_t = model(x_t, cond, t_batch)
                    x_t = x_t + dt * v_t

        # Optionally show ground truth
        if show_ground_truth:
            _plot(axes[i, col], _to_numpy(u[i]), "GT", i)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved flow evolution visualization to {fig_path}")


def _fmt_t(t, j: int, total: int) -> str:
    """Column title for a given time-step index."""
    if j == 0:
        return "t=0\n(noise)"
    if j == total - 1:
        return "t=1\n(pred)"
    return f"t={t.item():.2f}"
