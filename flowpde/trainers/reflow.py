"""
Reflow — Iterative Path Straightening
======================================

Rectified Flow (Liu et al., 2023) has two parts, and they are easy to
conflate:

1. **The objective.** Training with a linear path and independent coupling.
   This is what ``create_flow_matching(flow, variant='rectified')`` gives you,
   and mathematically it is the same as standard flow matching with a linear
   path.  It does *not*, on its own, straighten anything.

2. **Reflow.** The procedure in this module.  Take the trained model, generate
   pairs :math:`(z, \\mathrm{ODE}(z))` by integrating from noise, then retrain
   on those pairs.  Repeat.  This is what actually straightens trajectories
   and makes few-step Euler sampling accurate.

The correctness requirement that makes or breaks reflow: training **must** use
the same :math:`z` that produced each generated target.  Reflow works because
the pairing is deterministic and induced by the model; if training resamples
noise independently, the pairs decouple and the procedure degrades into
training on the model's own samples, which straightens nothing.

This module therefore emits an explicit ``x_0`` for every pair and requires
the objective to consume it via
:class:`~flowpde.flows.components.sources.BatchSource`.

Usage::

    from flowpde.flows import BatchSource
    from flowpde.trainers import generate_reflow_pairs, reflow

    # The objective must read x_0 from the batch.
    objective.source = BatchSource()

    pairs = generate_reflow_pairs(objective, train_loader, n_steps=100)
    loader = DataLoader(pairs, batch_size=32, shuffle=True)
    trainer.train(loader, epochs=50, ...)

or, for the whole loop::

    reflow(objective, train_loader, optimizer_factory=make_optimizer,
           num_iterations=2, epochs_per_iteration=50)
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from flowpde.flows.components.sources import BatchSource


class ReflowDataset(Dataset):
    """
    Precomputed ``(x_0, x_1, condition)`` triples produced by a trained flow.

    Each item is a batch dict carrying the source point alongside the usual
    target and condition, so a ``BatchSource``-configured objective trains on
    the exact pairing the model generated.

    Args:
        x_0: Source points, shape ``(N, D)``.
        x_1: Generated targets, shape ``(N, D)``.
        condition: Conditioning tensors, shape ``(N, C)``.
        target_key: Key for the target in emitted samples.
        condition_key: Key for the condition in emitted samples.
        source_key: Key for ``x_0`` in emitted samples; must match the
            ``BatchSource`` key on the objective.
    """

    def __init__(
        self,
        x_0: Tensor,
        x_1: Tensor,
        condition: Tensor,
        target_key: str = "target",
        condition_key: str = "input",
        source_key: str = "x_0",
    ):
        if not (len(x_0) == len(x_1) == len(condition)):
            raise ValueError(
                f"Mismatched lengths: x_0={len(x_0)}, x_1={len(x_1)}, "
                f"condition={len(condition)}"
            )
        self.x_0 = x_0
        self.x_1 = x_1
        self.condition = condition
        self.target_key = target_key
        self.condition_key = condition_key
        self.source_key = source_key

    def __len__(self) -> int:
        return len(self.x_0)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            self.source_key: self.x_0[idx],
            self.target_key: self.x_1[idx],
            self.condition_key: self.condition[idx],
        }

    def __repr__(self) -> str:
        return (
            f"ReflowDataset(n={len(self)}, dim={self.x_0.shape[1]}, "
            f"source_key='{self.source_key}')"
        )


def _objective_device(objective: Any) -> torch.device:
    """Device the objective's parameters currently live on."""
    if hasattr(objective, "model_device"):
        return objective.model_device
    return next(objective.model.parameters()).device


@torch.no_grad()
def generate_reflow_pairs(
    objective: Any,
    data_loader: Iterable,
    n_steps: int = 100,
    solver: str = "euler",
    condition_key: Optional[str] = None,
    target_key: Optional[str] = None,
    source_key: str = "x_0",
    seed: Optional[int] = None,
    max_batches: Optional[int] = None,
) -> ReflowDataset:
    """
    Generate ``(z, ODE(z))`` pairs from the current model.

    For every condition in ``data_loader``, draw :math:`z \\sim
    \\mathcal{N}(0, I)` and integrate the learned ODE to obtain
    :math:`x_1' = \\mathrm{ODE}(z \\mid f)`.  The ground-truth targets in the
    loader are ignored — only the conditions are reused.  Reflow trains on the
    model's own transport map, not on the data.

    Use a high ``n_steps`` here: pair quality bounds what reflow can achieve,
    and errors introduced now are baked into the next iteration's targets.

    Args:
        objective: Flow objective exposing ``sample(...)``.
        data_loader: Loader supplying conditions.
        n_steps: ODE steps used to generate targets.
        solver: ODE solver name.
        condition_key: Batch key for conditions.  Defaults to the objective's.
        target_key: Batch key for targets, used only to infer the target
            shape.  Defaults to the objective's.
        source_key: Key under which ``x_0`` is stored in the result.
        seed: Seed for the generated noise, for reproducible pairs.
        max_batches: Optionally cap the number of batches consumed.

    Returns:
        A :class:`ReflowDataset` of the generated pairs.
    """
    condition_key = condition_key or getattr(objective, "condition_key", "input")
    target_key = target_key or getattr(objective, "target_key", "target")

    device = _objective_device(objective)

    was_training = objective.training
    objective.eval()

    sources: List[Tensor] = []
    targets: List[Tensor] = []
    conditions: List[Tensor] = []

    try:
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            condition = batch[condition_key].to(device)
            batch_size = condition.shape[0]
            flat_dim = int(batch[target_key][0].numel())

            if seed is not None:
                generator = torch.Generator().manual_seed(seed + batch_idx)
                z = torch.randn(batch_size, flat_dim, generator=generator).to(device)
            else:
                z = torch.randn(batch_size, flat_dim, device=device)

            x_1 = objective.sample(
                condition=condition,
                n_steps=n_steps,
                solver=solver,
                x_init=z,
            )

            sources.append(z.flatten(start_dim=1).cpu())
            targets.append(x_1.flatten(start_dim=1).cpu())
            conditions.append(condition.flatten(start_dim=1).cpu())
    finally:
        if was_training:
            objective.train()

    if not sources:
        raise RuntimeError("Data loader produced no batches; no pairs generated.")

    return ReflowDataset(
        x_0=torch.cat(sources),
        x_1=torch.cat(targets),
        condition=torch.cat(conditions),
        target_key=target_key,
        condition_key=condition_key,
        source_key=source_key,
    )


def reflow(
    objective: Any,
    data_loader: Iterable,
    optimizer_factory: Callable[[Iterable], torch.optim.Optimizer],
    num_iterations: int = 1,
    epochs_per_iteration: int = 50,
    n_steps: int = 100,
    solver: str = "euler",
    batch_size: int = 32,
    save_dir: Optional[str] = None,
    print_stats_interval: int = 10,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full reflow loop: generate pairs, retrain, repeat.

    Each iteration regenerates pairs with the *current* model, so trajectories
    straighten progressively.  Optimizer state is rebuilt every iteration via
    ``optimizer_factory`` because the target distribution changes between
    iterations and stale moment estimates work against the new objective.

    The objective's source is switched to
    :class:`~flowpde.flows.components.sources.BatchSource` for the duration and
    restored afterwards, so sampling behaviour is unchanged on return.

    Args:
        objective: Flow objective to straighten, already trained.
        data_loader: Loader supplying conditions (targets are ignored).
        optimizer_factory: Callable taking model parameters and returning a
            fresh optimizer, e.g. ``lambda p: torch.optim.Adam(p, lr=1e-4)``.
        num_iterations: Number of reflow iterations.
        epochs_per_iteration: Training epochs per iteration.
        n_steps: ODE steps used when generating pairs.
        solver: ODE solver used when generating pairs.
        batch_size: Batch size for training on generated pairs.
        save_dir: Optional directory; each iteration writes to a
            ``reflow_{i}`` subdirectory.
        print_stats_interval: Passed through to the trainer.
        trainer_kwargs: Extra keyword arguments for ``Trainer`` (``ema_decay``,
            ``gradient_clip``, ``validator``, ...).
        seed: Seed for pair generation.

    Returns:
        One record per iteration with the pair count and final loss.
    """
    from flowpde.trainers.trainer import Trainer

    trainer_kwargs = dict(trainer_kwargs or {})
    # Follow the objective's existing device rather than Trainer's 'cuda'
    # default, so reflow does not relocate an already-placed model.
    trainer_kwargs.setdefault("device", str(_objective_device(objective)))
    original_source = objective.source
    history: List[Dict[str, Any]] = []

    try:
        for iteration in range(1, num_iterations + 1):
            # Pairs come from the model as it stands at the start of the
            # iteration, so generation must use the ordinary noise source.
            objective.source = original_source
            pairs = generate_reflow_pairs(
                objective,
                data_loader,
                n_steps=n_steps,
                solver=solver,
                seed=None if seed is None else seed + 1000 * iteration,
            )

            # Training must consume those exact pairs.
            objective.source = BatchSource(key=pairs.source_key, strict=True)

            pair_loader = DataLoader(pairs, batch_size=batch_size, shuffle=True)
            iteration_dir = (
                os.path.join(save_dir, f"reflow_{iteration}") if save_dir else None
            )
            if iteration_dir:
                os.makedirs(iteration_dir, exist_ok=True)

            trainer = Trainer(
                objective,
                optimizer_factory(objective.model.parameters()),
                **trainer_kwargs,
            )
            trainer.train(
                pair_loader,
                epochs=epochs_per_iteration,
                print_stats_interval=print_stats_interval,
                save_dir=iteration_dir or save_dir or ".",
                save_interval=epochs_per_iteration,
            )

            history.append(
                {
                    "iteration": iteration,
                    "num_pairs": len(pairs),
                    "best_loss": trainer.best_loss,
                }
            )
    finally:
        objective.source = original_source

    return history
