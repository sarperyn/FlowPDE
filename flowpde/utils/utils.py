import torch
import os
import yaml
import importlib
import glob


def save_model(**kwargs):
    os.makedirs(kwargs["save_dir"], exist_ok=True)

    # Use custom filename if provided, otherwise use epoch-based naming
    if "filename" in kwargs:
        ckpt_path = os.path.join(kwargs["save_dir"], kwargs["filename"])
    else:
        ckpt_path = os.path.join(kwargs["save_dir"], f"model_{kwargs['epoch']}.pt")

    torch.save({
        "model_state": kwargs["model"].state_dict(),
        "optimizer_state": kwargs["optimizer"].state_dict(),
        "scheduler_state": kwargs["scheduler"].state_dict(),
        "train_loss": kwargs["epoch_loss"],
        "epoch": kwargs["epoch"],
    }, ckpt_path)

def print_stats(**kwargs):
    parts = []
    for k, v in kwargs.items():
        # Auto-format based on value type
        if isinstance(v, float):
            # Use scientific notation if it's small or large
            if abs(v) < 1e-3 or abs(v) > 1e3:
                v_str = f"{v:.2e}"
            else:
                v_str = f"{v:.6f}".rstrip("0").rstrip(".")
        elif isinstance(v, int):
            v_str = f"{v}"
        else:
            v_str = str(v)
        parts.append(f"{k}: {v_str}")

    print(" | ".join(parts))


def load_config(config_path: str) -> dict:
    # Load YAML config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_class(class_path: str):
    # Dynamically import class from string path
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def find_latest_checkpoint(project_dir: str, config: dict) -> str:
    """
    Find the latest checkpoint file for a given configuration.
    
    Args:
        project_dir: Project root directory (absolute path)
        config: Configuration dictionary containing 'name' and 'spatial_dim'
    
    Returns:
        Relative path to latest checkpoint from project_dir, or None if not found
    """
    name = config.get('name', 'exp')
    spatial_dim = config.get('spatial_dim', '')
    
    # Checkpoints are saved in: results/training/{name}_{spatial_dim}/checkpoints/*.pt
    checkpoint_dir = os.path.join(project_dir, "results", "training", f"{name}_{spatial_dim}", "checkpoints")
    
    if os.path.exists(checkpoint_dir):
        # Find all model checkpoint files and sort them
        checkpoint_pattern = os.path.join(checkpoint_dir, "model_*.pt")
        checkpoints = sorted(glob.glob(checkpoint_pattern))
        
        if checkpoints:
            # Return the latest checkpoint (highest epoch number)
            latest = checkpoints[-1]
            # Make path relative to project root
            return os.path.relpath(latest, project_dir)
    
    return None

