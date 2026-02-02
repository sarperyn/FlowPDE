# APEBench Integration Module for FlowPDE

This module provides a unified, object-oriented interface for generating PDE datasets using **APEBench's** JAX-based spectral solvers, converted to **PyTorch** format for seamless integration with FlowPDE's normalizing flow training pipeline.

## Features

- **Procedural Data Generation**: No need to download large datasets - generate on-the-fly
- **JAX → PyTorch Conversion**: Automatic conversion with proper tensor format
- **Opt-in Disk Caching**: Cache generated data to `.cache/apebench/` for reuse
- **Modular Design**: Easily extensible to new PDEs via scenario classes
- **Rich Metadata**: Physics parameters, normalization stats, and generation config

## Supported PDEs

| PDE | Dimensions | Problem Types | Key Parameters |
|-----|------------|---------------|----------------|
| **Burgers** | 1D | Forward, Inverse | `viscosity`, `resolution`, `num_time_steps` |
| **Poisson** | 2D | Source-to-Solution | `resolution`, `source_scale` |

## Installation

The APEBench integration requires additional dependencies:

```bash
pip install apebench exponax jax
```

For GPU support with JAX:
```bash
pip install -U "jax[cuda12]"  # For CUDA 12
```

## Quick Start

### Basic Usage

```python
from flowpde.datasets.apebench import APEBenchProvider

# Create a Burgers 1D dataset (forward problem: IC → final state)
dataset = APEBenchProvider.create(
    pde='burgers_1d',
    problem='forward',
    resolution=160,
    viscosity=0.0003,
    n_train_samples=500,
    cache=True,  # Recommended: cache to disk
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    inputs, targets = batch
    # inputs: initial conditions (B, 1, 160)
    # targets: final states (B, 1, 160)
```

### Poisson 2D Dataset

```python
# Create Poisson 2D dataset (source → solution mapping)
poisson_dataset = APEBenchProvider.create(
    pde='poisson_2d',
    resolution=64,
    n_train_samples=1000,
    cache=True,
)

# Access raw data
raw_data = poisson_dataset.get_raw_data()
# raw_data['source']: (N, 1, 64, 64) source terms
# raw_data['solution']: (N, 1, 64, 64) solutions
```

### Access Trajectory Data (Burgers)

```python
# Get full time evolution trajectories
dataset = APEBenchProvider.create(
    pde='burgers_1d',
    problem='forward',
    resolution=160,
    n_train_samples=100,
    num_time_steps=50,
)

raw_data = dataset.get_raw_data()
# raw_data['initial']: (N, 1, 160) - initial conditions
# raw_data['final']: (N, 1, 160) - final states  
# raw_data['trajectory']: (N, T, 1, 160) - full time evolution
# raw_data['time']: (T,) - time points
# raw_data['viscosity']: scalar - physics parameter
```

## Architecture

```
flowpde/datasets/apebench/
├── __init__.py          # Public API exports
├── provider.py          # APEBenchProvider factory class
├── base.py              # BaseAPEBenchScenario ABC + ScenarioConfig
├── converters.py        # JAX ↔ PyTorch conversion utilities
├── cache.py             # CacheManager for disk persistence
└── scenarios/
    ├── __init__.py
    ├── burgers.py       # BurgersScenario implementation
    └── poisson.py       # PoissonScenario implementation
```

### Key Classes

| Class | Description |
|-------|-------------|
| `APEBenchProvider` | High-level factory - main entry point |
| `BaseAPEBenchScenario` | Abstract base for PDE scenarios |
| `BurgersScenario` | 1D Burgers equation (viscous) |
| `PoissonScenario` | 2D Poisson equation (elliptic) |
| `CacheManager` | Handles disk caching with human-readable keys |

## Caching

Data is cached to `.cache/apebench/` with human-readable directory names:

```
.cache/apebench/
├── burgers_1d/
│   └── res160_nu0.0003_n500_t50_s0/
│       ├── train.pt       # PyTorch tensors
│       └── metadata.json  # Generation config
└── poisson_2d/
    └── res64_n1000_t1_s0_L10.0/
        ├── train.pt
        └── metadata.json
```

Enable caching by setting `cache=True`:

```python
dataset = APEBenchProvider.create(
    pde='burgers_1d',
    cache=True,  # Enable caching
    cache_dir='.cache/apebench',  # Optional: custom cache directory
)
```

## Adding New PDEs

To add a new PDE scenario, subclass `BaseAPEBenchScenario`:

```python
from flowpde.datasets.apebench import BaseAPEBenchScenario
from flowpde.datasets.apebench.base import ScenarioConfig

class MyPDEScenario(BaseAPEBenchScenario):
    """Custom PDE scenario."""
    
    @classmethod
    def get_default_config(cls) -> ScenarioConfig:
        return ScenarioConfig(
            pde_name='my_pde',
            spatial_dims=2,
            default_resolution=64,
            # ... other defaults
        )
    
    def generate(self, config: dict) -> dict:
        # Implement JAX-based generation using exponax
        # Return dict with 'input', 'target', and metadata
        pass
```

Then register with the provider:

```python
APEBenchProvider.register('my_pde', MyPDEScenario)
```

## JAX Device Configuration

By default, JAX runs on CPU to avoid GPU memory conflicts with PyTorch. To use GPU:

```python
import jax
jax.config.update('jax_platform_name', 'gpu')

# Then create dataset
dataset = APEBenchProvider.create(...)
```

## Data Format

All data follows PyTorch conventions:

- **Tensor format**: `(N, C, *spatial_dims)` - batch, channels, then spatial
- **dtype**: `torch.float32`
- **Domain**: Periodic boundaries on $[0, L]^d$

### Burgers 1D Output

| Key | Shape | Description |
|-----|-------|-------------|
| `initial` | `(N, 1, resolution)` | Initial conditions $u(x, 0)$ |
| `final` | `(N, 1, resolution)` | Final states $u(x, T)$ |
| `trajectory` | `(N, T, 1, resolution)` | Full evolution (optional) |
| `time` | `(T,)` | Time points |
| `viscosity` | scalar | Viscosity parameter $\nu$ |

### Poisson 2D Output

| Key | Shape | Description |
|-----|-------|-------------|
| `source` | `(N, 1, H, W)` | Source terms $f(x, y)$ |
| `solution` | `(N, 1, H, W)` | Solutions $u(x, y)$ where $-\nabla^2 u = f$ |

## Examples

See the examples directory for complete usage:

- [examples/apebench_dataset_examples.py](../../../examples/apebench_dataset_examples.py) - Comprehensive API examples
- [notebooks/apebench_data_visualization.ipynb](../../../notebooks/apebench_data_visualization.ipynb) - Data visualization

## Citation

This integration uses **APEBench** and **Exponax** for PDE data generation. If you use this module in your research, please cite:

### APEBench (NeurIPS 2024)

```bibtex
@article{koehler2024apebench,
  title={{APEBench}: A Benchmark for Autoregressive Neural Emulators of {PDE}s},
  author={Felix Koehler and Simon Niedermayr and R{\"u}diger Westermann and Nils Thuerey},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={38},
  year={2024}
}
```

### Exponax

Exponax is the underlying spectral solver library used by APEBench:

- **Paper**: [arxiv.org/abs/2411.00180](https://arxiv.org/abs/2411.00180)
- **GitHub**: [github.com/Ceyron/exponax](https://github.com/Ceyron/exponax)
- **Documentation**: [fkoehler.site/exponax](https://fkoehler.site/exponax/)

### Related References

The spectral methods in Exponax are based on:

1. Cox, S. M., & Matthews, P. C. (2002). *Exponential time differencing for stiff systems*. Journal of Computational Physics, 176(2), 430-455.

2. Kassam, A. K., & Trefethen, L. N. (2005). *Fourth-order time-stepping for stiff PDEs*. SIAM Journal on Scientific Computing, 26(4), 1214-1233.

3. Montanelli, H., & Bootland, N. (2020). *Solving periodic semilinear stiff PDEs in 1D, 2D and 3D with exponential integrators*. Mathematics and Computers in Simulation, 178, 307-327.

## License

This module is part of FlowPDE and is released under the same license as the main project.

The underlying APEBench and Exponax libraries are released under the MIT License by Felix Köhler and the TUM Physics-based Simulation Group.

## Acknowledgments

- **Felix Köhler** ([@Ceyron](https://github.com/Ceyron)) - Creator of APEBench and Exponax
- **TUM Physics-based Simulation Group** ([ge.in.tum.de](https://ge.in.tum.de/)) - Research group
- **Munich Center for Machine Learning** ([mcml.ai](https://mcml.ai/)) - Funding
