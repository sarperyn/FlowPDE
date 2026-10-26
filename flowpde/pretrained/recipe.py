"""Recipes: plain-dict descriptions of a FlowPDE model that can be rebuilt.

A recipe names the backbone and its constructor arguments, the conditioner,
the flow settings and the objective settings.  It holds only strings, numbers,
booleans, lists and dicts, so it can live inside a checkpoint that loads with
``torch.load(..., weights_only=True)`` and be written out as JSON.

Example:
    >>> recipe = {
    ...     "model": {"name": "ConvNet", "kwargs": {
    ...         "spatial_dim": 1, "spatial_size": 64, "hidden_channels": 64,
    ...         "num_blocks": 6, "kernel_size": 5,
    ...         "solution_channels": 1, "condition_channels": 1,
    ...     }},
    ...     "conditioner": "concat",
    ...     "flow": {"target_key": "target", "condition_key": "input"},
    ...     "objective": {"name": "flow_matching", "path": "linear",
    ...                   "time_sampler": "uniform", "coupling": "independent"},
    ... }
    >>> objective = build_objective(recipe)
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from flowpde.core.base_conditioner import ConcatConditioner, NullConditioner
from flowpde.flows import NeuralODEFlow
from flowpde.models import MLP, ConvNet, ResNet, UNet
from flowpde.objectives import FlowMatchingObjective

MODELS = {
    "MLP": MLP,
    "ConvNet": ConvNet,
    "ResNet": ResNet,
    "UNet": UNet,
}

CONDITIONERS = {
    "concat": lambda: ConcatConditioner(dim=1),
    "null": NullConditioner,
}

OBJECTIVES = {"flow_matching"}

# Objective options a recipe may set; anything else is a typo, not a feature.
_OBJECTIVE_OPTIONS = {"path", "time_sampler", "coupling", "source", "sigma"}


def validate_recipe(recipe: Dict[str, Any]) -> None:
    """Raise ``ValueError`` naming the first problem with ``recipe``."""
    for key in ("model", "objective"):
        if key not in recipe:
            raise ValueError(f"Recipe is missing the '{key}' section.")

    model_name = recipe["model"].get("name")
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}' in recipe; expected one of {sorted(MODELS)}."
        )

    conditioner = recipe.get("conditioner")
    if conditioner is not None and conditioner not in CONDITIONERS:
        raise ValueError(
            f"Unknown conditioner '{conditioner}' in recipe; expected one of "
            f"{sorted(CONDITIONERS)} or null."
        )

    objective = recipe["objective"]
    name = objective.get("name", "flow_matching")
    if name not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{name}' in recipe; expected one of {sorted(OBJECTIVES)}."
        )
    unknown = set(objective) - _OBJECTIVE_OPTIONS - {"name"}
    if unknown:
        raise ValueError(
            f"Unknown objective option(s) {sorted(unknown)} in recipe; "
            f"supported: {sorted(_OBJECTIVE_OPTIONS)}."
        )


def build_objective(recipe: Dict[str, Any]) -> FlowMatchingObjective:
    """Build backbone, flow and objective from a recipe, with fresh weights."""
    validate_recipe(recipe)
    recipe = copy.deepcopy(recipe)

    model_kwargs = dict(recipe["model"].get("kwargs", {}))
    conditioner = recipe.get("conditioner")
    if conditioner is not None:
        model_kwargs["conditioner"] = CONDITIONERS[conditioner]()
    model = MODELS[recipe["model"]["name"]](**model_kwargs)

    flow_kwargs = {"target_key": "target", "condition_key": "input"}
    flow_kwargs.update(recipe.get("flow", {}))
    flow = NeuralODEFlow(model, **flow_kwargs)

    objective_kwargs = {
        k: v for k, v in recipe["objective"].items() if k in _OBJECTIVE_OPTIONS
    }
    return FlowMatchingObjective(flow, **objective_kwargs)
