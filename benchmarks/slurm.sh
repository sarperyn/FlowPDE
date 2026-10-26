#!/bin/bash
# Run the full-scale FlowPDE benchmarks on a SLURM cluster, one task per array job.
#
#   sbatch benchmarks/slurm.sh                 # all four tasks
#   sbatch --array=0 benchmarks/slurm.sh       # just the first task
#
# Adjust the #SBATCH lines and the environment setup to your cluster, then run
# from the repository root. Afterwards, on any machine with the results:
#
#   python -m benchmarks.report benchmarks/runs/full --write docs/benchmarks.md
#
#SBATCH --job-name=flowpde-bench
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=benchmarks/logs/%x-%A_%a.out

set -euo pipefail

TASKS=(burgers_forward poisson_forward darcy_forward darcy_inverse)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

# Environment: either uv (installs from uv.lock) ...
uv sync
# ... or an existing virtualenv / conda env with `pip install -e ".[data]"`:
# source /path/to/venv/bin/activate

# JAX generates the data; keep it on the CPU so the GPU memory is left to PyTorch.
export JAX_PLATFORMS=cpu

uv run python -m benchmarks.run "$TASK" --scale full --device cuda --out benchmarks/runs
