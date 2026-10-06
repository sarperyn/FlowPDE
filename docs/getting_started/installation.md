# Installation

## Requirements

- Python 3.11+
- PyTorch 2.0+
- JAX 0.4.12+ (only for the Exponax data generators)

## Install from PyPI

```bash
pip install "flowpde[data]"   # library + Exponax/JAX PDE data generators
pip install flowpde           # library only
```

The core install covers flows, objectives, models, solvers, the trainer, metrics and
`FieldNormalizer`. The `data` extra adds JAX and Exponax, which the `PoissonGenerator`,
`BurgersGenerator` and `DarcyGenerator` in `flowpde.datasets` need; importing one of
them without the extra raises an `ImportError` that names the missing install.

| Extra | Adds |
|-------|------|
| `data` | `jax`, `exponax` — PDE dataset generation |
| `docs` | MkDocs toolchain for building this site |

## Install from source with uv (recommended for development)

```bash
git clone https://github.com/sarperyn/FlowPDE.git
cd FlowPDE

uv python install 3.11 && uv python pin 3.11
uv sync
source .venv/bin/activate
```

`uv sync` creates the virtual environment when needed and installs the versions
recorded in `uv.lock`, including the `data` extra and the `dev` group (pytest, ruff).

## Install from source with pip

This path requires Python 3.11+ to be installed already and does not require `uv`:

```bash
git clone https://github.com/sarperyn/FlowPDE.git
cd FlowPDE
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `torch` | Deep learning framework | core |
| `torchdiffeq` | ODE integration for inference | core |
| `numpy` | Numerical operations | core |
| `scipy` | Mini-batch OT coupling | core |
| `matplotlib` | Training-curve plots | core |
| `exponax` | Spectral PDE solvers for data generation ([docs](https://fkoehler.site/exponax/)) | `data` |
| `jax` | Required by Exponax | `data` |

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
uv run -m pytest                 # full suite (~20s)
uv run -m pytest -m "not slow"   # skip the Exponax integration tests
```

With pip, install the package with the `data` extra plus pytest, then run pytest directly:

```bash
python -m pip install -e ".[data]" pytest
python -m pytest
```
