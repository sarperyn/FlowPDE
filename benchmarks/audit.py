"""Reproduce CPU timing and retained-output checks for the SciML audit fixes.

Run: python -m benchmarks.audit
Numbers exclude compilation/warmup and are hardware dependent. Both paths use
identical inputs; this is a microbenchmark, not a PDE training benchmark.
"""

import json
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from flowpde import FlowMatchingObjective, NeuralODEFlow
from flowpde.solvers import ODEFlowSolver
from flowpde.trainers import FlowEvaluator


class Relaxation(nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(-0.2))

    def forward(self, x, condition, t):
        return self.rate * x + condition


def median_time(function, repeats=5):
    warmup = function()
    if hasattr(warmup, "block_until_ready"):
        warmup.block_until_ready()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def peak_cpu_bytes(function):
    """Incremental peak tensor memory from PyTorch's CPU allocation timeline."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                profile_memory=True, record_shapes=True, with_stack=True) as profile:
        function()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        profile.export_memory_timeline(str(path), device="cpu")
        _, sizes = json.loads(path.read_text())
    totals = [sum(row) for row in sizes]
    return max(totals) - totals[0]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = Relaxation()
    solver = ODEFlowSolver(model, method="euler")
    initial = torch.randn(32, 1024)
    condition = torch.zeros_like(initial)

    def sample(trajectory):
        return solver.sample(condition, x_init=initial, n_steps=200,
                             return_trajectory=trajectory)

    final = sample(False)
    dense, _ = sample(True)
    torch.testing.assert_close(final, dense, atol=1e-6, rtol=1e-5)
    results = {
        "endpoint_only": {"seconds": median_time(lambda: sample(False)),
                          "retained_output_bytes": final.untyped_storage().nbytes(),
                          "peak_incremental_cpu_bytes": peak_cpu_bytes(lambda: sample(False))},
        "dense_output": {"seconds": median_time(lambda: sample(True)),
                         "retained_output_bytes": dense.untyped_storage().nbytes(),
                         "peak_incremental_cpu_bytes": peak_cpu_bytes(lambda: sample(True))},
    }
    objective = FlowMatchingObjective(NeuralODEFlow(model, target_key="target", condition_key="input"))
    data = [{"input": torch.zeros(256), "target": torch.ones(256)} for _ in range(32)]
    loader = DataLoader(data, batch_size=32)
    serial = FlowEvaluator(objective, loader, n_steps=40, ensemble_size=8, ensemble_chunk_size=1)
    batched = FlowEvaluator(objective, loader, n_steps=40, ensemble_size=8, ensemble_chunk_size=8)
    batched_metrics = batched()
    for name, value in serial().items():
        assert abs(value - batched_metrics[name]) < 1e-6
    results["serial_ensemble_seconds"] = median_time(serial)
    results["batched_ensemble_seconds"] = median_time(batched)

    import jax
    import jax.numpy as jnp

    from flowpde.datasets.exponax.darcy import _cg_scan, _matvec_2d, enable_x64

    with enable_x64(True):
        n = 32
        grid = jnp.linspace(0, 1, n)
        kappa = jnp.exp(jnp.sin(grid[:, None] * 4) * jnp.cos(grid[None, :] * 3))
        h = 1 / (n - 1)
        rhs = jnp.sin(jnp.pi * grid[1:-1, None]) * jnp.sin(jnp.pi * grid[None, 1:-1])
        b = rhs.ravel()
        ki = kappa[1:-1, 1:-1]
        diagonal = (2 * ki + 0.5 * (kappa[2:, 1:-1] + kappa[:-2, 1:-1]
                    + kappa[1:-1, 2:] + kappa[1:-1, :-2])).ravel() / h ** 2

        def matvec(x):
            return _matvec_2d(x, kappa, h, n - 2)

        def fixed_cg():
            def step(carry, _):
                x, r, p = carry
                ap = matvec(p)
                rr = jnp.dot(r, r)
                alpha = rr / (jnp.dot(p, ap) + 1e-30)
                new_r = r - alpha * ap
                new_p = new_r + jnp.dot(new_r, new_r) / (rr + 1e-30) * p
                return (x + alpha * p, new_r, new_p), None
            return jax.lax.scan(step, (jnp.zeros_like(b), b, b), None, length=2000)[0][0]

        fixed = jax.jit(fixed_cg)
        pcg = jax.jit(lambda: _cg_scan(matvec, b, 2000, diagonal)[0])
        old, new = fixed(), pcg()
        old.block_until_ready()
        new.block_until_ready()
        results["fixed_cg_seconds"] = median_time(fixed)
        results["preconditioned_cg_seconds"] = median_time(pcg)
        results["cg_max_solution_difference"] = float(jnp.max(jnp.abs(old - new)))
        results["pcg_true_relative_residual"] = float(jnp.linalg.norm(b - matvec(new)) / jnp.linalg.norm(b))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
