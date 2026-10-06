"""Benchmark task definitions.

Each task fixes the data (generator settings, problem direction, split sizes
and seeds), the backbone, the objective, the optimisation budget and the
evaluation protocol.  The ``full`` settings follow the course-project
experiments reported in ``report.pdf`` (E2, E3, E6 and E10); every task is
evaluated on a held-out test split in physical units.

Three scales share one definition:

- ``full``  — report settings; sized for a GPU.
- ``quick`` — smaller grids, splits and epoch counts; minutes to an hour or two
  on a laptop CPU/MPS.  Numbers are indicative, not comparable to ``full``.
- ``smoke`` — tiny, for CI: checks that the pipeline runs end to end.

To add a task, add an entry to ``TASKS`` with the same keys.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

SCALES = ("full", "quick", "smoke")

# Shared defaults: the optimisation and evaluation protocol used by all of
# the report's experiments.  Tasks override only what differs.
_SPLITS = {
    "train": 3000, "val": 200, "test": 200,
    "train_seed": 101, "val_seed": 202, "test_seed": 303,
}
_TRAINING = {
    "epochs": 500,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "min_learning_rate": 1e-6,
    "weight_decay": 1e-6,
    "gradient_clip": 1.0,
    "ema_decay": 0.999,
    "val_interval": 5,
    "val_max_batches": 4,
}
_EVALUATION = {
    "n_steps": 50,
    "solver": "euler",
    "metrics": ["rel_l2", "h1", "mae", "rel_max"],
    "seed": 7,
}
_OBJECTIVE = {
    "name": "flow_matching",
    "path": "linear",
    "time_sampler": "uniform",
    "coupling": "independent",
    "source": "gaussian",
    "sigma": 0.0,
}
_CONVNET = {"name": "ConvNet", "kwargs": {"hidden_channels": 48, "num_blocks": 6, "kernel_size": 5}}

_DARCY = {
    "num_spatial_dims": 2, "num_points": 64, "domain_extent": 1.0,
    "kappa_alpha": 2.0, "kappa_tau": 3.0, "kappa_scale": 1.0, "kappa_min": 0.1,
    "f_cutoff": 8, "f_amplitude_min": 0.1, "f_amplitude_max": 5.0,
    "cg_steps": 2000,
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _task(**overrides: Any) -> Dict[str, Any]:
    base = {
        "splits": _SPLITS,
        "model": _CONVNET,
        "conditioner": "concat",
        "objective": _OBJECTIVE,
        "training": _TRAINING,
        "evaluation": _EVALUATION,
    }
    return deep_merge(base, overrides)


TASKS: Dict[str, Dict[str, Any]] = {
    "burgers_forward": _task(
        description="1D viscous Burgers: initial state u(x,0) to final state u(x,T)",
        pde="Burgers", problem="forward", report="E2",
        data={
            "generator": "BurgersGenerator",
            "config": {
                "num_spatial_dims": 1, "num_points": 64, "domain_extent": 1.0,
                "dt": 0.001, "num_steps": 400,
                "diffusivity_min": 0.01, "diffusivity_max": 0.01,
                "convection_scale": 1.0, "ic_num_terms": 3, "ic_max_mode": 4,
            },
        },
        model={"kwargs": {"hidden_channels": 64}},
        training={"batch_size": 32},
        scales={
            "quick": {"splits": {"train": 1000, "val": 64, "test": 200},
                      "training": {"epochs": 100}},
        },
    ),
    "poisson_forward": _task(
        description="2D Poisson, eight-mode sources: source f to solution u",
        pde="Poisson", problem="forward", report="E10",
        data={
            "generator": "PoissonGenerator",
            "config": {
                "num_spatial_dims": 2, "num_points": 64, "domain_extent": 10.0,
                "order": 2, "source_num_terms": 8, "source_max_mode": 8,
            },
        },
        training={"epochs": 150},
        scales={
            "quick": {"data": {"config": {"num_points": 32}},
                      "splits": {"train": 1000, "val": 64, "test": 200},
                      "training": {"epochs": 60}},
        },
    ),
    "darcy_forward": _task(
        description="2D Darcy flow: coefficient and source (kappa, f) to pressure u",
        pde="Darcy", problem="forward", report="E3",
        data={"generator": "DarcyGenerator", "config": _DARCY},
        scales={
            "quick": {"data": {"config": {"num_points": 32, "cg_steps": 500}},
                      "splits": {"train": 1000, "val": 64, "test": 200},
                      "training": {"epochs": 60}},
        },
    ),
    "darcy_inverse": _task(
        description="2D Darcy inverse: 15% observed, noisy u and known f to coefficient kappa",
        pde="Darcy", problem="inverse", inverse_mode="coefficient", report="E6",
        data={
            "generator": "DarcyGenerator",
            "config": {**_DARCY, "obs_noise_std": 0.0002, "obs_mask_fraction": 0.15},
        },
        scales={
            "quick": {"data": {"config": {"num_points": 32, "cg_steps": 500}},
                      "splits": {"train": 1000, "val": 64, "test": 200},
                      "training": {"epochs": 60}},
        },
    ),
}

# Applied to every task at the smoke scale, after the task's own settings.
_SMOKE = {
    "data": {"config": {"num_points": 16, "num_steps": 20, "cg_steps": 500}},
    "splits": {"train": 16, "val": 8, "test": 8},
    "model": {"kwargs": {"hidden_channels": 8, "num_blocks": 2}},
    "training": {"epochs": 1, "batch_size": 4, "val_interval": 1, "val_max_batches": 1},
    "evaluation": {"n_steps": 4},
}


def resolve_task(name: str, scale: str = "full") -> Dict[str, Any]:
    """The settings for ``name`` at ``scale``, with scale overrides applied."""
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {', '.join(sorted(TASKS))}")
    if scale not in SCALES:
        raise ValueError(f"Unknown scale '{scale}'. Choose from {SCALES}.")

    task = copy.deepcopy(TASKS[name])
    scales = task.pop("scales", {})
    if scale == "quick":
        task = deep_merge(task, scales.get("quick", {}))
    elif scale == "smoke":
        task = deep_merge(task, _SMOKE)
        # Settings that only one generator understands must not leak into others.
        config = task["data"]["config"]
        if task["data"]["generator"] != "BurgersGenerator":
            config.pop("num_steps", None)
        if task["data"]["generator"] != "DarcyGenerator":
            config.pop("cg_steps", None)
    task["name"] = name
    task["scale"] = scale
    return task
