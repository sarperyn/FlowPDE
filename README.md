## FlowPDE

Backend library for normalizing flows solving inverse/forward problems in PDEs.

## Setup (uv)

```bash
uv python install 3.11
uv python pin 3.11
uv venv
uv sync
```

Activate the environment if you want a shell session:

```bash
source .venv/bin/activate
```

## Common tasks

```bash
# Run tests
uv run -m pytest

# Run a training script
uv run python scripts/train_poisson_2d_flowmatching.py
```
