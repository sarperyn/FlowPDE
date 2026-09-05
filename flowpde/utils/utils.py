import os
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    epoch_loss: float,
    save_dir: str,
    filename: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """Save a training checkpoint.

    Args:
        model:       The model whose state to save.
        optimizer:   Optimizer state.
        scheduler:   LR scheduler state.  May be ``None``.
        epoch:       Current epoch number (used in auto-generated filename).
        epoch_loss:  Training loss for this epoch.
        save_dir:    Directory to save the checkpoint in.
        filename:    Optional explicit filename.  Defaults to
                     ``model_{epoch}.pt``.
        extra:       Optional additional entries merged into the checkpoint,
                     e.g. ``{'ema_state': ..., 'normalizer_state': ...}``.
                     A checkpoint is only reusable if everything needed to
                     reproduce inference travels with the weights.

    Returns:
        Absolute path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    fname = filename if filename is not None else f"model_{epoch}.pt"
    ckpt_path = os.path.join(save_dir, fname)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "train_loss": epoch_loss,
        "epoch": epoch,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def print_stats(**kwargs) -> None:
    """Print key-value pairs on one line, auto-formatting numeric values."""
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, float):
            v_str = f"{v:.2e}" if (abs(v) < 1e-3 or abs(v) > 1e3) else f"{v:.6f}".rstrip("0").rstrip(".")
        elif isinstance(v, int):
            v_str = str(v)
        else:
            v_str = str(v)
        parts.append(f"{k}: {v_str}")
    print(" | ".join(parts))


def plot_curve(
    epoch_losses: Sequence[float],
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
