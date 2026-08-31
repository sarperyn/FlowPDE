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
uv sync
source .venv/bin/activate
```

`uv sync` creates the virtual environment when needed and installs the versions
recorded in `uv.lock`.

## Install with pip

This path requires Python 3.11 to be installed already and does not require `uv`:

```bash
git clone https://github.com/sarperyn/FlowPDE.git
cd FlowPDE
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Optional extras

```bash
python -m pip install -e ".[dev]"    # pytest
python -m pip install -e ".[docs]"   # documentation tools
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

With uv:

```bash
uv run -m pytest                 # full suite (~8s)
uv run -m pytest -m "not slow"   # skip the Exponax integration tests
```

With pip, first install the development extra and then run pytest directly:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Building the Docs

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve    # local preview at http://127.0.0.1:8000
python -m mkdocs build    # static site into site/
```
