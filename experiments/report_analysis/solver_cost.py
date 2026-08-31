"""Cost of a learned sample against the cost of the reference solver.

A surrogate is only worth building if querying it is cheaper than solving the
equation, so the comparison has to be made -- and made fairly.  Two rules are
observed here.  Both sides run on the **same device** (CPU), because a
GPU-versus-CPU comparison measures the hardware.  And both are batched over the
whole test split, because the reference solver is a ``jax.vmap`` over a
fixed-iteration conjugate gradient and would be penalised absurdly if called one
sample at a time.

The reference cost includes drawing $\\kappa$ and $f$, which slightly overstates
it; the surrogate cost excludes loading weights, which slightly understates it.
Neither correction changes the order of magnitude.

Usage::

    uv run python -m experiments.report_analysis.solver_cost
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from experiments.report_analysis.common import load_run, sample_dataset
from flowpde.datasets.exponax.darcy import DarcyConfig, DarcyGenerator

RUN = "results/experiments/exp03_backbone_ablation_darcy/convnet_small"


def time_reference(config: Dict[str, Any], n_samples: int, repeats: int) -> float:
    """Seconds per solution for the conjugate-gradient reference solver."""
    generator = DarcyGenerator(config=DarcyConfig(**config["data"]["generator"]))
    generator.generate(num_samples=8, seed=1, problem="forward")   # warm up JIT

    best = float("inf")
    for repeat in range(repeats):
        start = time.time()
        generator.generate(num_samples=n_samples, seed=1000 + repeat,
                           problem="forward")
        best = min(best, time.time() - start)
    return best / n_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16, 50])
    parser.add_argument("--out", default="results/analysis/solver_cost.json")
    args = parser.parse_args()

    config = yaml.safe_load((Path(args.run) / "resolved_config.yaml").read_text())

    torch.set_num_threads(torch.get_num_threads())
    print("Reference solver (JAX, CPU)...", flush=True)
    reference = time_reference(config, args.n_samples, args.repeats)
    print(f"  {reference * 1000:.2f} ms per solution", flush=True)

    print("Surrogate (torch, CPU)...", flush=True)
    run = load_run(args.run, device="cpu")
    surrogate = {}
    for steps in args.steps:
        start = time.time()
        sample_dataset(run, n_steps=steps, solver="euler",
                       batch_size=args.batch_size, limit=args.n_samples)
        elapsed = (time.time() - start) / args.n_samples
        surrogate[steps] = elapsed
        print(f"  {steps:3d} Euler steps: {elapsed * 1000:.2f} ms per sample "
              f"({elapsed / reference:.1f}x the solver)", flush=True)

    payload = {
        "device": "cpu",
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "reference_seconds_per_solution": reference,
        "surrogate_seconds_per_sample": surrogate,
        "threads": torch.get_num_threads(),
        "run": args.run,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
