"""Straightness of the learned transport, measured on trained checkpoints.

Straightness follows Liu et al. (2023): the deviation between the velocity along
a trajectory and the chord joining that trajectory's endpoints.  It is *chord
deviation*, not the spread of velocity norms — a field that turns at constant
speed is curved, and ``tests/test_metrics.py`` pins that distinction.

Both modes are recorded, because they answer different questions:

``trajectory``
    deviation along the model's **own** ODE paths.  This is the quantity that
    predicts few-step sampling quality, so it is the one to read against the
    NFE sweep.
``interpolant``
    deviation along the training interpolant between sampled pairs.  Cheaper,
    and it reports how far the learned marginal velocity sits from the
    conditional target — a training diagnostic, not a sampling one.

Usage::

    uv run python -m experiments.report_analysis.straightness
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch

from experiments.report_analysis.common import load_run
from experiments.report_analysis.nfe_sweep import ROOTS, RUNS


def measure(
    run,
    experiment: str,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    loader = run.loader(batch_size=batch_size)

    for mode in ("trajectory", "interpolant"):
        totals: Dict[str, float] = {}
        seen = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = {k: v.to(run.device) for k, v in batch.items()}
            # The base draw inside estimate_straightness is unseeded; fix it
            # here so the number is reproducible.
            torch.manual_seed(seed + batch_idx)
            result = run.objective.estimate_straightness(
                batch, n_time_points=10, mode=mode, n_steps=50, solver="euler",
            )
            for key, value in result.items():
                totals[key] = totals.get(key, 0.0) + value
            seen += 1

        rows.append({
            "experiment": experiment,
            "pde": run.pde,
            "variant": run.name,
            "backbone": run.backbone,
            "parameter_count": run.parameter_count,
            "mode": mode,
            "n_batches": seen,
            "batch_size": batch_size,
            **{k: round(v / seen, 6) for k, v in totals.items()},
        })
        print(
            f"  {run.name:18s} {mode:12s} "
            f"normalized={rows[-1]['normalized_straightness']:.4f}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pdes", nargs="+", default=["burgers", "darcy"])
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results/analysis/straightness.csv")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for pde in args.pdes:
        for experiment, variant in RUNS[pde]:
            run_dir = Path(ROOTS[experiment]) / variant
            print(f"[{pde}] {run_dir}", flush=True)
            run = load_run(run_dir, device=args.device)
            rows.extend(
                measure(run, experiment, args.batch_size, args.max_batches, args.seed)
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
