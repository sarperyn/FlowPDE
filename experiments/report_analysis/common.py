"""Shared loading and scoring helpers for the report's analysis passes.

Every analysis in this package re-evaluates checkpoints that were trained
earlier, on other hardware.  Nothing here trains.  The reason it works at all
is that ``Trainer(checkpoint_extra=...)`` stored the normalizer's state in the
checkpoint, so only the 200-sample test split has to be regenerated (~3 s)
rather than the full training corpus.

Metrics are computed **per test sample** rather than per batch.  ``FlowEvaluator``
accumulates batch means and divides by the batch count, which makes its headline
number depend on batch size whenever the last batch is short; for the report we
want a paired per-sample distribution anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.common.models import build_objective, count_parameters
from flowpde.datasets.exponax.burgers import BurgersConfig, BurgersGenerator
from flowpde.datasets.exponax.darcy import DarcyConfig, DarcyGenerator
from flowpde.datasets.normalization import FieldNormalizer


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample metrics
# ─────────────────────────────────────────────────────────────────────────────

def _flat(x: Tensor) -> Tensor:
    return x.flatten(start_dim=1)


def rel_l2_per_sample(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative L2 error, one value per sample."""
    num = _flat(pred - target).norm(dim=1)
    den = _flat(target).norm(dim=1).clamp(min=eps)
    return num / den


def h1_per_sample(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative H1 error per sample; falls back to relative L2 in 1-D.

    Matches ``flowpde.utils.metrics.h1_error`` term for term — same first-order
    finite differences, same normalization — but returns the per-sample vector
    instead of its mean.
    """
    if pred.ndim < 4:
        return rel_l2_per_sample(pred, target, eps=eps)

    def norm_sq(x: Tensor) -> Tensor:
        out = x.pow(2).flatten(start_dim=1).sum(dim=1)
        out = out + (x[..., 1:, :] - x[..., :-1, :]).pow(2).flatten(start_dim=1).sum(dim=1)
        out = out + (x[..., :, 1:] - x[..., :, :-1]).pow(2).flatten(start_dim=1).sum(dim=1)
        return out

    return norm_sq(pred - target).sqrt() / norm_sq(target).sqrt().clamp(min=eps)


def mae_per_sample(pred: Tensor, target: Tensor) -> Tensor:
    return _flat(pred - target).abs().mean(dim=1)


def rel_max_per_sample(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Max pointwise error normalised by the target's own standard deviation.

    This is why the metric legitimately exceeds 1: the denominator is a spread,
    not a range.
    """
    max_err = _flat(pred - target).abs().max(dim=1).values
    return max_err / _flat(target).std(dim=1).clamp(min=eps)


PER_SAMPLE_METRICS = {
    "rel_l2": rel_l2_per_sample,
    "h1": h1_per_sample,
    "mae": mae_per_sample,
    "rel_max": rel_max_per_sample,
}


def score_per_sample(pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
    return {name: fn(pred, target) for name, fn in PER_SAMPLE_METRICS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Run loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedRun:
    """A trained run, restored far enough to sample from."""

    name: str
    run_dir: Path
    config: Dict[str, Any]
    objective: Any
    normalizer: FieldNormalizer
    dataset: Any
    device: torch.device
    parameter_count: int
    backbone: str
    conditioner: str
    pde: str
    problem: str

    def loader(self, batch_size: int = 16, limit: Optional[int] = None) -> DataLoader:
        dataset = self.dataset
        if limit is not None and limit < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(limit))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def denormalize(self, tensor: Tensor) -> Tensor:
        return self.normalizer.denormalize_channels(
            self.dataset_target_fields, tensor
        )

    @property
    def dataset_target_fields(self):
        base = self.dataset
        return base.target_fields


def _detect_pde(generator_cfg: Dict[str, Any]) -> str:
    if "kappa_alpha" in generator_cfg:
        return "darcy"
    if "dt" in generator_cfg or "num_steps" in generator_cfg:
        return "burgers"
    return "poisson"


def _build_test_dataset(
    config: Dict[str, Any],
    normalizer: FieldNormalizer,
    problem: str,
) -> Tuple[Any, str]:
    """Regenerate only the test split and attach the training normalizer."""
    generator_cfg = dict(config["data"]["generator"])
    splits = config["data"]["splits"]
    pde = _detect_pde(generator_cfg)

    if pde == "darcy":
        generator = DarcyGenerator(config=DarcyConfig(**generator_cfg))
        kwargs = {}
        if problem == "inverse":
            kwargs["inverse_mode"] = config.get("inverse_mode", "coefficient")
        dataset = generator.generate(
            num_samples=splits["test"], seed=splits["test_seed"],
            problem=problem, **kwargs,
        )
    elif pde == "burgers":
        generator = BurgersGenerator(config=BurgersConfig(**generator_cfg))
        dataset = generator.generate(
            num_samples=splits["test"], seed=splits["test_seed"], problem=problem,
        )
    else:
        raise ValueError(f"No loader wired for PDE '{pde}'.")

    dataset.set_normalizer(normalizer)
    return dataset, pde


def load_run(
    run_dir: str | Path,
    device: str = "mps",
    checkpoint: str = "best_model.pt",
) -> LoadedRun:
    """Restore a trained run from its output directory.

    Handles both experiment layouts: exp01/exp05 keep the backbone settings in
    ``model``, while exp02/exp03/exp04 merge variant overrides into
    ``active_model_config``.
    """
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    ckpt = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)

    normalizer = FieldNormalizer()
    normalizer.load_state_dict(ckpt["normalizer_state"])

    variant_cfg = config.get("active_variant_config", {}) or {}
    model_cfg = config.get("active_model_config") or config["model"]
    backbone = ckpt.get("backbone") or variant_cfg.get("backbone") or "unet"
    conditioner = (
        ckpt.get("conditioner") or variant_cfg.get("conditioner") or "concat"
    )
    problem = "inverse" if "inverse_mode" in config else "forward"

    dataset, pde = _build_test_dataset(config, normalizer, problem)

    objective = build_objective(
        model_config=model_cfg,
        objective_config=config["objective"],
        conditioner_name=conditioner,
        backbone=backbone,
    )
    objective.model.load_state_dict(ckpt["model_state"])
    torch_device = torch.device(device)
    objective.model.to(torch_device)
    objective.eval()

    return LoadedRun(
        name=config.get("active_variant", run_dir.name),
        run_dir=run_dir,
        config=config,
        objective=objective,
        normalizer=normalizer,
        dataset=dataset,
        device=torch_device,
        parameter_count=count_parameters(objective.model),
        backbone=backbone,
        conditioner=conditioner,
        pde=pde,
        problem=problem,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_dataset(
    run: LoadedRun,
    n_steps: int,
    solver: str = "euler",
    batch_size: int = 16,
    limit: Optional[int] = None,
    seed: int = 7,
    member: int = 0,
    **solver_kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    """Sample the whole (possibly truncated) test set once.

    Returns ``(prediction, target)`` in **physical units**.  The initial noise is
    drawn from a fixed generator seeded by ``(seed, batch index, member)``, so
    two calls that differ only in ``n_steps`` integrate the *same* trajectory
    from the *same* starting point — which is what makes the NFE sweep a
    statement about the solver rather than about the draw.
    """
    preds, targets = [], []
    loader = run.loader(batch_size=batch_size, limit=limit)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            target = batch["target"].to(run.device)
            condition = batch["input"].to(run.device)
            shape = target.shape[1:]
            flat_dim = int(target[0].numel())

            generator = torch.Generator().manual_seed(
                seed + 1009 * batch_idx + 31 * member
            )
            x_init = torch.randn(
                target.shape[0], flat_dim, generator=generator
            ).to(run.device)

            out = run.objective.sample(
                condition=condition, n_steps=n_steps, solver=solver,
                x_init=x_init, **solver_kwargs,
            )
            preds.append(out.reshape(target.shape[0], *shape).cpu())
            targets.append(target.cpu())

    prediction = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    fields = run.dataset_target_fields
    return (
        run.normalizer.denormalize_channels(fields, prediction),
        run.normalizer.denormalize_channels(fields, target),
    )
