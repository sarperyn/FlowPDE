# Installation

## Requirements

- Python 3.11+
- PyTorch 2.0+
- JAX 0.4.12+ (for Exponax PDE data generation)

## Install with uv (recommended)

```bash
git clone https://github.com/sarperyn/FlowPDE.git
cd FlowPDE

uv python install 3.11 && uv python pin 3.11
uv venv && uv sync
source .venv/bin/activate
```

## Install with pip

```bash
git clone https://github.com/sarperyn/FlowPDE.git
cd FlowPDE
pip install -e .
```

### Optional extras

```bash
pip install -e ".[dev]"    # pytest
pip install -e ".[docs]"   # mkdocs-material, mkdocstrings, mkdocs-jupyter
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `torchdiffeq` | ODE integration for inference |
| `exponax` | Spectral PDE solvers for data generation ([docs](https://fkoehler.site/exponax/)) |
| `jax` | Required by Exponax |
| `scipy` | Mini-batch OT coupling |
| `matplotlib` | Visualization |
| `PyYAML` | Configuration files |
| `numpy` | Numerical operations |

## JAX Installation

Exponax requires JAX. Install the build matching your hardware:

=== "CPU"

    ```bash
    pip install jax
    ```

=== "CUDA 12"

    ```bash
    pip install -U "jax[cuda12]"
    ```

See the [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html)
for more options.

## Verify Installation

```python
import torch
from flowpde import NeuralODEFlow, FlowMatchingObjective, UNet

model = UNet(spatial_dim=2, spatial_size=32)
flow = NeuralODEFlow(model, target_key="target", condition_key="input")
objective = FlowMatchingObjective(flow)
print("FlowPDE installed successfully!")
```

## Running the Tests

```bash
uv run -m pytest                 # full suite (~8s)
uv run -m pytest -m "not slow"   # skip the Exponax integration tests
```

## Building the Docs

```bash
pip install -e ".[docs]"
mkdocs serve    # local preview at http://127.0.0.1:8000
mkdocs build    # static site into site/
```
