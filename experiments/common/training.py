"""Training helpers for FlowPDE experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from flowpde.trainers import Trainer


def build_optimizer(parameters, config: Dict[str, Any]):
    """Build an optimizer from config."""
    name = config.get("optimizer", "adamw").lower()
    lr = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config: Dict[str, Any], epochs: int):
    """Build an optional learning-rate scheduler."""
    name = config.get("scheduler", "none").lower()
    if name in {"none", "null"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=config.get("min_learning_rate", 0.0),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def build_trainer(
    objective,
    train_config: Dict[str, Any],
    device: str,
    validator: Optional[Any],
    checkpoint_extra: Optional[Dict[str, Any]] = None,
) -> Trainer:
    """Create the generic FlowPDE trainer."""
    optimizer = build_optimizer(objective.model.parameters(), train_config)
    scheduler = build_scheduler(optimizer, train_config, train_config["epochs"])
    return Trainer(
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip=train_config.get("gradient_clip"),
        use_amp=train_config.get("use_amp", False),
        ema_decay=train_config.get("ema_decay"),
        validator=validator,
        val_interval=train_config.get("val_interval", 1),
        monitor=train_config.get("monitor", "rel_l2"),
        monitor_mode=train_config.get("monitor_mode", "min"),
        checkpoint_extra=checkpoint_extra,
    )

