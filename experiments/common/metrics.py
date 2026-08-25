"""Evaluation helpers for experiment scripts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from flowpde.trainers.evaluation import FlowEvaluator


def build_flow_evaluator(
    objective: Any,
    data_loader: Iterable,
    eval_config: Dict[str, Any],
    normalizer: Optional[Any] = None,
    target_fields: Optional[Sequence[str]] = None,
    max_batches: Optional[int] = None,
) -> FlowEvaluator:
    """Build a sampling-based evaluator for a flow objective."""
    return FlowEvaluator(
        objective=objective,
        data_loader=data_loader,
        target_key="target",
        condition_key="input",
        n_steps=eval_config.get("n_steps", 50),
        solver=eval_config.get("solver", "euler"),
        max_batches=max_batches,
        normalizer=normalizer,
        target_fields=target_fields,
        metrics=eval_config.get("metrics", ["rel_l2"]),
        seed=eval_config.get("seed", 0),
        ensemble_size=eval_config.get("ensemble_size", 1),
    )

