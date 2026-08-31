"""Accuracy against sampling cost, on fixed weights.

Sampling is an ODE solve, so the number of function evaluations is a dial that
can be turned *after* training.  This pass turns it, on checkpoints that are
never modified, and records error against cost for each solver.

Function evaluations rather than steps are the cost unit: one RK4 step costs
four network evaluations, so comparing solvers by step count would flatter RK4
by exactly a factor of four.  NFE is counted by a forward hook rather than
assumed, which also lets the adaptive solver report the count it actually used.

Usage::

    uv run python -m experiments.report_analysis.nfe_sweep
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from experiments.report_analysis.common import (
    LoadedRun,
    load_run,
    sample_dataset,
    score_per_sample,
)

EULER_STEPS = [1, 2, 4, 8, 16, 32, 64, 128]
RK4_STEPS = [1, 2, 4, 8, 16, 32]
MIDPOINT_STEPS = [1, 2, 4, 8, 16, 32, 64]

RUNS = {
    "darcy": [
        ("exp03", "convnet_small"),
        ("exp03", "resnet"),
        ("exp03", "unet"),
        ("exp03", "unet_no_attention"),
    ],
    "burgers": [
        ("exp02", "convnet_small"),
        ("exp02", "resnet"),
        ("exp02", "unet"),
        ("exp02", "unet_no_attention"),
    ],
}

ROOTS = {
    "exp02": "results/experiments/exp02_backbone_ablation_burgers",
    "exp03": "results/experiments/exp03_backbone_ablation_darcy",
}


class EvalCounter:
    """Count network forward passes, so NFE is measured and not inferred."""

    def __init__(self, model: torch.nn.Module):
        self.count = 0
        self._handle = model.register_forward_pre_hook(self._hook)

    def _hook(self, *_: Any) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0

    def close(self) -> None:
        self._handle.remove()


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def sweep_run(
    run: LoadedRun,
    experiment: str,
    limit: int | None,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """One run, every solver setting."""
    rows: List[Dict[str, Any]] = []
    counter = EvalCounter(run.objective.model)
    n_samples = min(limit or len(run.dataset), len(run.dataset))

    settings = (
        [("euler", n) for n in EULER_STEPS]
        + [("midpoint", n) for n in MIDPOINT_STEPS]
        + [("rk4", n) for n in RK4_STEPS]
        + [("dopri5", None)]
    )

    for solver, n_steps in settings:
        counter.reset()
        kwargs: Dict[str, Any] = {}
        if solver == "dopri5":
            kwargs = {"rtol": 1e-5, "atol": 1e-7}
            steps = 1                      # ignored by the adaptive solver
        else:
            steps = n_steps

        _sync(run.device)
        start = time.time()
        pred, target = sample_dataset(
            run, n_steps=steps, solver=solver, batch_size=batch_size,
            limit=limit, **kwargs,
        )
        _sync(run.device)
        elapsed = time.time() - start

        scores = score_per_sample(pred, target)
        n_batches = max(1, (n_samples + batch_size - 1) // batch_size)
        rows.append({
            "experiment": experiment,
            "pde": run.pde,
            "variant": run.name,
            "backbone": run.backbone,
            "parameter_count": run.parameter_count,
            "solver": solver,
            "n_steps": "" if solver == "dopri5" else steps,
            "nfe": counter.count / n_batches,
            "n_test_samples": n_samples,
            "seconds_total": round(elapsed, 3),
            "seconds_per_sample": round(elapsed / n_samples, 5),
            **{
                f"{name}_mean": round(value.mean().item(), 6)
                for name, value in scores.items()
            },
            **{
                f"{name}_median": round(value.median().item(), 6)
                for name, value in scores.items()
            },
        })
        print(
            f"  {run.name:18s} {solver:8s} steps={str(steps):>4s} "
            f"nfe={rows[-1]['nfe']:6.1f} rel_l2={rows[-1]['rel_l2_mean']:.4f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    counter.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pdes", nargs="+", default=["burgers", "darcy"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on test samples; default is the whole split.")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--out", default="results/analysis/nfe_sweep.csv")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for pde in args.pdes:
        for experiment, variant in RUNS[pde]:
            run_dir = Path(ROOTS[experiment]) / variant
            print(f"[{pde}] {run_dir}", flush=True)
            run = load_run(run_dir, device=args.device)
            rows.extend(
                sweep_run(run, experiment, args.limit, args.batch_size)
            )
            del run
            if args.device == "mps":
                torch.mps.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
