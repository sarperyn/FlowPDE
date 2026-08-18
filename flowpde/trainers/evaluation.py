"""
Sampling-Based Validation for Flow Models
==========================================

The flow-matching training loss is a poor model-selection signal.  It regresses
the velocity ``x_1 - x_0`` from ``x_t`` alone, but many ``(x_0, x_1)`` pairs
produce the same ``x_t``, so the loss has a large irreducible floor set by that
ambiguity.  Two checkpoints can differ substantially in sample quality while
their losses differ in the fourth decimal.

The quantity you actually care about is the error of solutions produced by
integrating the learned ODE.  :class:`FlowEvaluator` measures exactly that:
it samples through the solver and scores against ground truth in physical
units.

Usage::

    evaluator = FlowEvaluator(
        objective, val_loader,
        normalizer=normalizer, target_fields=val_ds.target_fields,
    )
    trainer = Trainer(objective, optimizer, validator=evaluator, monitor='rel_l2')

Sampling noise is drawn from a fixed seed, so the same ``x_0`` is used at every
validation call.  Without that, epoch-to-epoch differences would be dominated
by which noise happened to be drawn rather than by model improvement.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor

from flowpde.utils.metrics import EvalMetrics


class FlowEvaluator:
    """
    Evaluate a flow objective by ODE sampling against ground-truth solutions.

    Args:
        objective: Object exposing ``sample(condition=..., ...)`` (e.g.
            ``FlowMatchingObjective``), or one exposing ``.flow`` that does.
        data_loader: Validation/test loader yielding batches with the
            configured target and condition keys.
        target_key: Batch key holding ground truth.  Defaults to the
            objective's configured key.
        condition_key: Batch key holding conditioning data.  Defaults to the
            objective's configured key.
        n_steps: ODE integration steps.
        solver: Solver name passed through to the sampler.
        max_batches: Cap the number of validation batches, for cheap
            in-training monitoring.  ``None`` uses the whole loader.
        normalizer: ``FieldNormalizer`` used on the data.  When given,
            predictions and targets are mapped back to physical units before
            scoring, so reported errors are comparable across normalization
            choices.
        target_fields: Raw field names composing the target, in channel order
            (``dataset.target_fields``).  Required for denormalization.
        metrics: Metric names for :class:`~flowpde.utils.metrics.EvalMetrics`.
            Defaults to ``['rel_l2']``, which is cheap enough to run often.
        seed: Seed for the fixed evaluation noise.
        ensemble_size: Samples drawn per condition.  With ``1`` (default) the
            metrics score a single draw.  With more, the ensemble *mean* is
            scored and the ensemble spread is reported alongside it — the
            uncertainty-quantification view.
        solver_kwargs: Extra keyword arguments forwarded to the sampler.

    Returns from ``__call__``:
        Dict of metric name → float.  With ``ensemble_size > 1`` the keys
        ``mean_rel_l2`` and ``sample_spread`` are added.
    """

    def __init__(
        self,
        objective: Any,
        data_loader: Iterable,
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
        n_steps: int = 50,
        solver: str = "euler",
        max_batches: Optional[int] = None,
        normalizer: Optional[Any] = None,
        target_fields: Optional[Sequence[str]] = None,
        metrics: Optional[List[str]] = None,
        seed: int = 0,
        ensemble_size: int = 1,
        **solver_kwargs: Any,
    ):
        self.objective = objective
        self.data_loader = data_loader
        self.n_steps = n_steps
        self.solver = solver
        self.max_batches = max_batches
        self.normalizer = normalizer
        self.target_fields = list(target_fields) if target_fields else None
        self.seed = seed
        self.ensemble_size = max(1, int(ensemble_size))
        self.solver_kwargs = solver_kwargs
        self.metrics = EvalMetrics(metrics or ["rel_l2"])

        self.target_key = target_key or getattr(objective, "target_key", "target")
        self.condition_key = condition_key or getattr(
            objective, "condition_key", "input"
        )

        if self.normalizer is not None and self.target_fields is None:
            raise ValueError(
                "target_fields is required when a normalizer is given, so "
                "predictions can be mapped back to physical units. Pass "
                "dataset.target_fields."
            )

    # Internals

    def _sampler(self):
        """Resolve the callable that maps a condition to generated samples."""
        if hasattr(self.objective, "sample"):
            return self.objective.sample
        flow = getattr(self.objective, "flow", None)
        if flow is not None and hasattr(flow, "sample"):
            return flow.sample
        raise AttributeError(
            f"{type(self.objective).__name__} exposes no sample(); cannot run "
            "sampling-based validation."
        )

    def _device(self) -> torch.device:
        for attr in ("model_device",):
            if hasattr(self.objective, attr):
                return getattr(self.objective, attr)
        return next(self.objective.model.parameters()).device

    def _denormalize(self, tensor: Tensor) -> Tensor:
        if self.normalizer is None:
            return tensor
        return self.normalizer.denormalize_channels(self.target_fields, tensor)

    # Evaluation

    @torch.no_grad()
    def __call__(self) -> Dict[str, float]:
        """Run one validation pass and return averaged metrics."""
        sampler = self._sampler()
        device = self._device()

        was_training = self.objective.training
        self.objective.eval()

        totals: Dict[str, float] = {}
        num_batches = 0

        try:
            for batch_idx, batch in enumerate(self.data_loader):
                if self.max_batches is not None and batch_idx >= self.max_batches:
                    break

                target = batch[self.target_key].to(device)
                condition = batch[self.condition_key].to(device)
                batch_size = target.shape[0]
                target_shape = target.shape[1:]
                flat_dim = int(target[0].numel())

                # Fixed noise per (batch, member) so successive validation
                # calls differ only by the model, never by the draw.
                members = []
                for member in range(self.ensemble_size):
                    generator = torch.Generator().manual_seed(
                        self.seed + 1009 * batch_idx + 31 * member
                    )
                    x_init = torch.randn(
                        batch_size, flat_dim, generator=generator
                    ).to(device)

                    samples = sampler(
                        condition=condition,
                        n_steps=self.n_steps,
                        solver=self.solver,
                        x_init=x_init,
                        **self.solver_kwargs,
                    )
                    members.append(samples.reshape(batch_size, *target_shape))

                stacked = torch.stack(members, dim=0)      # (E, B, C, *spatial)
                prediction = stacked.mean(dim=0)

                prediction = self._denormalize(prediction)
                target_physical = self._denormalize(target)

                batch_metrics = self.metrics(prediction, target_physical)

                if self.ensemble_size > 1:
                    physical = self._denormalize(
                        stacked.reshape(-1, *target_shape)
                    ).reshape(self.ensemble_size, batch_size, *target_shape)
                    spread = physical.std(dim=0)
                    scale = target_physical.abs().mean().clamp(min=1e-8)
                    batch_metrics["mean_rel_l2"] = batch_metrics["rel_l2"]
                    batch_metrics["sample_spread"] = (
                        spread.mean() / scale
                    ).item()

                for name, value in batch_metrics.items():
                    totals[name] = totals.get(name, 0.0) + value
                num_batches += 1
        finally:
            if was_training:
                self.objective.train()

        if num_batches == 0:
            raise RuntimeError("Validation loader produced no batches.")

        return {name: value / num_batches for name, value in totals.items()}

    def __repr__(self) -> str:
        return (
            f"FlowEvaluator(solver={self.solver}, n_steps={self.n_steps}, "
            f"ensemble_size={self.ensemble_size}, metrics={self.metrics.metrics})"
        )
