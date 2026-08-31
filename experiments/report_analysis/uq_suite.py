"""Distributional scoring of trained conditional flows.

The report argues at length that relative L2 of the ensemble mean measures
accuracy and says nothing about calibration, and that a generative model has to
be scored as one.  Every experiment run so far used ``ensemble_size=1``, so
nothing in ``results/`` actually tests that.  This pass does.

For each model it draws ``K`` samples per test condition from a fixed seed and
computes, in physical units:

* CRPS (fair estimator) and the energy score — the pointwise proper scoring rule
  and its multivariate generalization.  The second is the one that catches a
  model reproducing every marginal correctly while generating spatially
  incoherent fields, which is the failure mode that matters for PDE solutions.
* credible-interval coverage at 50/90/95 % and the full reliability curve
* the rank histogram — flat means calibrated, U-shaped under-dispersed,
  dome-shaped over-dispersed
* the spread-skill ratio, with the finite-ensemble correction sqrt((K+1)/K)
  applied before it is read
* the error-spread rank correlation: does predicted spread actually predict
  error?

Both directions are scored.  On a well-posed *forward* problem the conditional
law is a point mass, so the ensemble spread there is model error rather than
physical uncertainty — running the same suite on a forward model turns that
remark into a measurement instead of an assertion.

Usage::

    uv run python -m experiments.report_analysis.uq_suite
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from experiments.report_analysis.common import (
    LoadedRun,
    load_run,
    score_per_sample,
)
from flowpde.utils.uq_metrics import (
    crps_ensemble,
    credible_interval_coverage,
    energy_score,
    error_spread_correlation,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
)

EXP05 = "results/experiments/exp05_inverse_conditioning_ablation"
EXP03 = "results/experiments/exp03_backbone_ablation_darcy"

# All eight E5 cells are scored here rather than only the headline ones, so
# that every inverse number in the report comes from one documented draw
# (K members, fixed seed) instead of from the uncommitted plotting script that
# produced the original per-sample CSVs.  The two forward models are included
# to test Remark 4 empirically: on a Dirac posterior the spread should be model
# error, and the ensemble should look badly calibrated.
TARGETS = [
    ("inverse_coefficient_5000_concat", f"{EXP05}/coefficient_5000/concat"),
    ("inverse_coefficient_5000_null", f"{EXP05}/coefficient_5000/null"),
    ("inverse_joint_5000_concat", f"{EXP05}/joint_5000/concat"),
    ("inverse_joint_5000_null", f"{EXP05}/joint_5000/null"),
    ("forward_darcy_convnet", f"{EXP03}/convnet_small"),
    ("inverse_coefficient_3000_concat", f"{EXP05}/coefficient/concat"),
    ("inverse_coefficient_3000_null", f"{EXP05}/coefficient/null"),
    ("inverse_joint_3000_concat", f"{EXP05}/joint/concat"),
    ("inverse_joint_3000_null", f"{EXP05}/joint/null"),
    ("forward_darcy_unet", f"{EXP03}/unet"),
]


def draw_ensemble(
    run: LoadedRun,
    ensemble_size: int,
    n_steps: int,
    solver: str,
    batch_size: int,
    limit: Optional[int],
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Draw ``K`` samples per condition.

    Returns ``(samples, target, condition)`` with samples of shape
    ``(K, B, C, *spatial)``, in physical units.  Noise is seeded per
    (batch, member) exactly as ``FlowEvaluator`` does, so the draw is
    reproducible and independent of batch ordering.
    """
    members: List[List[Tensor]] = [[] for _ in range(ensemble_size)]
    targets, conditions = [], []
    loader = run.loader(batch_size=batch_size, limit=limit)
    fields = run.dataset_target_fields

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            target = batch["target"].to(run.device)
            condition = batch["input"].to(run.device)
            shape = target.shape[1:]
            flat_dim = int(target[0].numel())
            targets.append(target.cpu())
            conditions.append(condition.cpu())

            for member in range(ensemble_size):
                generator = torch.Generator().manual_seed(
                    seed + 1009 * batch_idx + 31 * member
                )
                x_init = torch.randn(
                    target.shape[0], flat_dim, generator=generator
                ).to(run.device)
                out = run.objective.sample(
                    condition=condition, n_steps=n_steps, solver=solver,
                    x_init=x_init,
                )
                members[member].append(out.reshape(target.shape[0], *shape).cpu())
            print(f"    batch {batch_idx}", end="\r", flush=True)

    samples = torch.stack([torch.cat(m, dim=0) for m in members], dim=0)
    target = torch.cat(targets, dim=0)
    condition = torch.cat(conditions, dim=0)

    k, b = samples.shape[0], samples.shape[1]
    samples = run.normalizer.denormalize_channels(
        fields, samples.reshape(k * b, *samples.shape[2:])
    ).reshape(k, b, *samples.shape[2:])
    target = run.normalizer.denormalize_channels(fields, target)
    return samples, target, condition


def score(
    samples: Tensor,
    target: Tensor,
    label: str,
    run: LoadedRun,
    ensemble_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the full suite on one drawn ensemble."""
    mean_pred = samples.mean(dim=0)
    point = score_per_sample(mean_pred, target)
    single = score_per_sample(samples[0], target)

    spread_skill = spread_skill_ratio(samples, target)
    correlation = error_spread_correlation(samples, target)

    row: Dict[str, Any] = {
        "label": label,
        "pde": run.pde,
        "problem": run.problem,
        "variant": run.name,
        "conditioner": run.conditioner,
        "backbone": run.backbone,
        "ensemble_size": ensemble_size,
        "n_test_samples": target.shape[0],
        "crps": round(crps_ensemble(samples, target).item(), 6),
        "energy_score": round(energy_score(samples, target).item(), 6),
        "spread_skill_ratio": round(spread_skill["ratio"], 6),
        "spread_skill_corrected": round(spread_skill["adjusted_ratio"], 6),
        "ensemble_spread": round(spread_skill["spread"], 6),
        "ensemble_mean_error": round(spread_skill["skill"], 6),
        "spearman": round(correlation["spearman"], 6),
        "top_decile_error_ratio": round(correlation["top_decile_error_ratio"], 6),
        "mean_rel_l2": round(point["rel_l2"].mean().item(), 6),
        "mean_h1": round(point["h1"].mean().item(), 6),
        "single_draw_rel_l2": round(single["rel_l2"].mean().item(), 6),
    }
    for level in (0.5, 0.9, 0.95):
        key = f"coverage_{int(round(level * 100))}"
        row[key] = round(
            credible_interval_coverage(samples, target, level).item(), 6
        )

    hist = rank_histogram(samples, target, normalize=True)
    hist_rows = [
        {"label": label, "bin_index": i, "frequency": round(v, 8),
         "n_bins": len(hist), "uniform": round(1.0 / len(hist), 8)}
        for i, v in enumerate(hist.tolist())
    ]

    levels = [0.05 * i for i in range(1, 20)]
    curve = reliability_curve(samples, target, levels)
    rel_rows = [
        {"label": label, "nominal": round(n, 4), "empirical": round(e, 6),
         "calibration_error": round(curve["calibration_error"], 6)}
        for n, e in zip(curve["nominal"], curve["empirical"])
    ]
    return row, hist_rows, rel_rows


def _write(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--ensemble-size", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--solver", default="euler")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out-dir", default="results/analysis")
    parser.add_argument("--cache-dir", default="results/analysis/ensembles")
    parser.add_argument("--refresh", action="store_true",
                        help="Redraw ensembles even when a cache exists.")
    args = parser.parse_args()

    selected = [
        (label, path) for label, path in TARGETS
        if args.labels is None or label in args.labels
    ]

    summary: List[Dict[str, Any]] = []
    hist_rows: List[Dict[str, Any]] = []
    rel_rows: List[Dict[str, Any]] = []

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    skipped: List[str] = []
    for label, path in selected:
        checkpoint = Path(path) / "best_model.pt"
        if not checkpoint.exists():
            # Some E5 cells have metrics and per-sample CSVs but no saved
            # weights, so they cannot be re-scored.  Skip loudly rather than
            # aborting the whole pass.
            print(f"[{label}] SKIPPED — no checkpoint at {checkpoint}",
                  flush=True)
            skipped.append(label)
            continue

        print(f"[{label}] {path}", flush=True)
        cache_path = cache_dir / f"{label}.pt"
        run = load_run(path, device=args.device)

        cached = None
        if cache_path.exists() and not args.refresh:
            candidate = torch.load(cache_path, map_location="cpu",
                                   weights_only=False)
            if (candidate.get("ensemble_size") == args.ensemble_size
                    and candidate["target"].shape[0] == min(
                        args.limit or len(run.dataset), len(run.dataset))):
                cached = candidate
                print("  reusing cached ensemble", flush=True)

        if cached is None:
            samples, target, condition = draw_ensemble(
                run, args.ensemble_size, args.n_steps, args.solver,
                args.batch_size, args.limit, args.seed,
            )
            # Cached so the identifiability map and the sample figures do not
            # have to redraw a 32-member ensemble, and so an interrupted pass
            # can be resumed.
            torch.save(
                {"samples": samples, "target": target, "condition": condition,
                 "target_fields": run.dataset_target_fields,
                 "ensemble_size": args.ensemble_size},
                cache_path,
            )
        else:
            samples = cached["samples"]
            target = cached["target"]
            condition = cached["condition"]

        row, hist, rel = score(samples, target, label, run, args.ensemble_size)
        summary.append(row)
        hist_rows.extend(hist)
        rel_rows.extend(rel)
        print(
            f"  crps={row['crps']:.4g} energy={row['energy_score']:.4g} "
            f"cov90={row['coverage_90']:.3f} "
            f"spread/skill={row['spread_skill_corrected']:.3f} "
            f"mean_rel_l2={row['mean_rel_l2']:.4f}",
            flush=True,
        )
        del run, samples, target, condition
        if args.device == "mps":
            torch.mps.empty_cache()

    out_dir = Path(args.out_dir)
    _write(out_dir / "uq_summary.csv", summary)
    _write(out_dir / "uq_rank_histogram.csv", hist_rows)
    _write(out_dir / "uq_reliability.csv", rel_rows)
    settings = dict(vars(args))
    settings["skipped_no_checkpoint"] = skipped
    (out_dir / "uq_settings.json").write_text(json.dumps(settings, indent=2))
    print(f"\nWrote {len(summary)} models to {out_dir}")
    if skipped:
        print(f"Skipped for want of a checkpoint: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
