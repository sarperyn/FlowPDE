r"""Poisson datasets from Exponax: solve $-\nabla^2 u = f$ periodically."""

from dataclasses import dataclass
from typing import Literal, Optional

import exponax as ex
import jax

from .base import GenerationConfig, PDEDataset
from .generator import ExponaxDatasetGenerator


@dataclass
class PoissonConfig(GenerationConfig):
    """Options passed directly to Exponax's Poisson and Fourier field generator.

    Sources are mean-zero, as required by the periodic Poisson equation.
    ``order=2`` solves the negative Laplacian; ``order=4`` uses Exponax's
    higher-order operator with its native sign convention.
    """

    order: int = 2
    source_cutoff: int = 5
    source_max_one: bool = False


class PoissonGenerator(ExponaxDatasetGenerator):
    """Wrap native Exponax source generation and Poisson solves for PyTorch.

    Uses ``ex.ic.RandomTruncatedFourierSeries``, ``ex.build_ic_set``, and
    ``ex.poisson.Poisson`` directly. ``problem='inverse'`` swaps the clean
    source/solution pair without adding noise or masking.
    """

    config_cls = PoissonConfig

    def generate(
        self,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        problem: Literal['forward', 'inverse'] = 'forward',
    ) -> PDEDataset:
        """Generate source/solution pairs on a periodic grid."""
        cfg, n, s = self.resolve_run(num_samples, seed)
        self.validate_problem(problem)
        solver = ex.poisson.Poisson(
            num_spatial_dims=cfg.num_spatial_dims,
            domain_extent=cfg.domain_extent,
            num_points=cfg.num_points,
            order=cfg.order,
        )
        sources = ex.build_ic_set(
            ex.ic.RandomTruncatedFourierSeries(
                cfg.num_spatial_dims, cutoff=cfg.source_cutoff,
                max_one=cfg.source_max_one,
            ),
            num_points=cfg.num_points, num_samples=n, key=jax.random.PRNGKey(s),
        )
        data = self.to_torch_data({
            'source': sources, 'solution': jax.vmap(solver)(sources),
        })
        self.print_summary("Poisson", n)
        return self.wrap_dataset(data, problem=problem)
