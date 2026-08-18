"""
Exponential Moving Average of Model Weights
============================================

Flow-matching and diffusion training produce noisy gradient estimates: the
loss regresses a velocity target that is only conditionally determined by the
network input, so consecutive iterates bounce around the optimum rather than
settling into it.  Averaging weights over training suppresses that jitter and
is standard practice for this model family — the averaged weights are what you
evaluate and ship, while the raw weights keep training.

Usage::

    ema = EMA(model, decay=0.999)

    for batch in loader:
        loss = objective.compute_loss(batch)
        loss.backward()
        optimizer.step()
        ema.update()              # after every optimizer step

    with ema.average_parameters():
        metrics = evaluate(model)  # model temporarily holds EMA weights
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch
from torch import nn


class EMA:
    """
    Maintains an exponential moving average of a model's parameters.

    The shadow weights follow

    .. math::

        \\theta^{\\text{EMA}} \\leftarrow d \\, \\theta^{\\text{EMA}}
                                        + (1 - d) \\, \\theta

    with decay :math:`d`.  Early in training the shadow is dominated by its
    (arbitrary) initial value, so ``warmup`` ramps the effective decay up from
    0 following :math:`\\min(d, (1 + n) / (10 + n))` at step :math:`n`.  This
    is the schedule used by most diffusion implementations and it removes the
    need to tune a separate "start EMA at step N" threshold.

    Buffers (e.g. BatchNorm running statistics) are copied rather than
    averaged, matching the reference implementations.

    Args:
        model: Model whose parameters are tracked.
        decay: Target decay rate.  Typical values are 0.999 (short runs) to
            0.9999 (long runs).  Higher means smoother and slower to adapt.
        warmup: Ramp the effective decay early in training (default: True).
        device: Optional device for the shadow copy.  Defaults to keeping
            each shadow tensor on the same device as its source parameter.

    Example:
        >>> model = nn.Linear(2, 2)
        >>> ema = EMA(model, decay=0.9, warmup=False)
        >>> ema.update()
        >>> with ema.average_parameters():
        ...     _ = model(torch.zeros(1, 2))   # uses averaged weights
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup: bool = True,
        device: Optional[torch.device] = None,
    ):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")

        self.model = model
        self.decay = decay
        self.warmup = warmup
        self.device = device
        self.num_updates = 0

        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow = param.detach().clone()
                self.shadow[name] = shadow.to(device) if device else shadow

        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @property
    def current_decay(self) -> float:
        """Effective decay at the current step, accounting for warmup."""
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None) -> None:
        """
        Update the shadow weights.  Call once after every optimizer step.

        Args:
            model: Optionally update from a different model instance;
                defaults to the model this EMA was constructed with.
        """
        model = model if model is not None else self.model
        decay = self.current_decay
        self.num_updates += 1

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                # A parameter that became trainable after construction.
                self.shadow[name] = param.detach().clone()
                continue
            shadow = self.shadow[name]
            shadow.mul_(decay).add_(param.detach().to(shadow.device), alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: Optional[nn.Module] = None) -> None:
        """Write the averaged weights into ``model``, permanently."""
        model = model if model is not None else self.model
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name].to(param.device))

    @torch.no_grad()
    def store(self, model: Optional[nn.Module] = None) -> None:
        """Stash the model's live weights so they can be restored later."""
        model = model if model is not None else self.model
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def restore(self, model: Optional[nn.Module] = None) -> None:
        """Put back the weights saved by :meth:`store`."""
        if self._backup is None:
            raise RuntimeError("restore() called without a matching store()")
        model = model if model is not None else self.model
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = None

    @contextmanager
    def average_parameters(
        self, model: Optional[nn.Module] = None
    ) -> Iterator[nn.Module]:
        """
        Temporarily swap the averaged weights into the model.

        The live training weights are restored on exit, including when the
        body raises, so a failed validation pass cannot corrupt training.
        """
        model = model if model is not None else self.model
        self.store(model)
        self.copy_to(model)
        try:
            yield model
        finally:
            self.restore(model)

    def to(self, device: torch.device) -> "EMA":
        """Move the shadow weights to ``device``."""
        self.device = device
        for name, tensor in self.shadow.items():
            self.shadow[name] = tensor.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Serializable state for checkpointing."""
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "EMA":
        """Restore state saved by :meth:`state_dict`."""
        self.decay = state["decay"]
        self.warmup = state.get("warmup", True)
        self.num_updates = state["num_updates"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        if self.device is not None:
            self.to(self.device)
        return self

    def __repr__(self) -> str:
        return (
            f"EMA(decay={self.decay}, warmup={self.warmup}, "
            f"num_updates={self.num_updates}, tracked={len(self.shadow)})"
        )
