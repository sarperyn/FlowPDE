import torch
import os
import yaml
import importlib
import glob as _glob
from typing import Optional


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    epoch_loss: float,
    save_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Save a training checkpoint.

    Args:
        model:       The model whose state to save.
        optimizer:   Optimizer state.
        scheduler:   LR scheduler state.
        epoch:       Current epoch number (used in auto-generated filename).
        epoch_loss:  Training loss for this epoch.
        save_dir:    Directory to save the checkpoint in.
        filename:    Optional explicit filename.  Defaults to
                     ``model_{epoch}.pt``.

    Returns:
        Absolute path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    fname = filename if filename is not None else f"model_{epoch}.pt"
    ckpt_path = os.path.join(save_dir, fname)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": epoch_loss,
            "epoch": epoch,
        },
        ckpt_path,
    )
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


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_class(class_path: str):
    """Dynamically import and return a class from a dotted path string.

    Example::

        Cls = load_class("flowpde.models.unet.UNet")
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path of the most-recently saved ``model_*.pt`` checkpoint.

    Args:
        checkpoint_dir: Directory containing ``model_*.pt`` files.

    Returns:
        Absolute path to the latest checkpoint, or ``None`` if none found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    pattern = os.path.join(checkpoint_dir, "model_*.pt")
    checkpoints = sorted(_glob.glob(pattern))
    return checkpoints[-1] if checkpoints else None

