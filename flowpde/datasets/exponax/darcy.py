r"""
Darcy Flow / Variable-Coefficient Poisson Data Generation
==========================================================

Generates $(\kappa, f) \to u$ datasets for the variable-coefficient
elliptic PDE:

$$
-\nabla \cdot (\kappa(x)\, \nabla u(x)) = f(x)
\quad \text{on } \Omega = [0, L]^d,
\qquad u = 0 \text{ on } \partial\Omega.
$$

Unlike the constant-coefficient Poisson equation supported by Exponax,
$\kappa$ is a random *spatial field*, making this a significantly more
realistic and challenging operator-learning problem.  This is the standard
Darcy-flow benchmark used in FNO, DeepONet, and related literature.

**Solver**
    Second-order finite differences on an endpoint-inclusive grid with zero
    Dirichlet boundaries. Jacobi-preconditioned CG stops on convergence and
    recomputes the true final residual. Solves and stored solution fields use
    float64 so the reported tolerance is not invalidated by a float32 cast.
    Training batches are converted to the model dtype by the flow.

**κ generation**
    A discrete periodic spectral field is standardized per realization, scaled,
    exponentiated, and clipped below by ``kappa_min``. Per-realization
    standardization means this is not an exact Gaussian covariance prior or an
    exact reproduction of the original FNO permeability distribution. Field
    arrays are used as nodal values on the finite-difference grid; their prior
    is specified on indices, not by a shared continuum coordinate generator.

**f generation**
    Smooth fixed-cutoff Fourier-series source fields.

**Example**

```python
gen = DarcyGenerator(num_points=64, kappa_alpha=2.0, kappa_tau=3.0)
dataset = gen.generate(num_samples=1000, seed=42)

sample = dataset[0]
# sample['input']  → (2, 64, 64)  cat([κ, f], channel dim)
# sample['target'] → (1, 64, 64)  solution u
```
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Literal, Optional

import jax
import jax.numpy as jnp

try:
    from jax import enable_x64
except ImportError:  # JAX < 0.8
    from jax.experimental import enable_x64
import torch
from torch.utils.data import Dataset

from ..normalization import FieldNormalizer
from .base import GenerationConfig
from .generator import (
    ExponaxDatasetGenerator,
    FourierFieldConfig,
    sample_fourier_fields,
)
from .utilities import jax_to_torch

logger = logging.getLogger(__name__)


def _grf_1d(key, N: int, alpha: float, tau: float):
    r"""Zero-mean 1-D Gaussian random field via spectral filtering.

    Power spectrum:

    $$
    S(k) = (\tau^2 + k^2)^{-\alpha}.
    $$

    Args:
        key:   JAX PRNG key.
        N:     Number of grid points.
        alpha: Spectral decay exponent.  Higher → smoother field.
        tau:   Inverse correlation length.  Higher → more oscillatory field.

    Returns:
        JAX array of shape `(N,)`.
    """
    freqs  = jnp.fft.fftfreq(N) * N                    # integer wavenumbers
    sqrt_S = (tau ** 2 + freqs ** 2) ** (-alpha / 2)
    sqrt_S = sqrt_S.at[0].set(0.0)                     # zero DC → zero mean
    noise  = jax.random.normal(key, (N,))
    return jnp.fft.ifft(jnp.fft.fft(noise) * sqrt_S).real   # (N,)


def _grf_2d(key, N: int, alpha: float, tau: float):
    r"""Zero-mean 2-D Gaussian random field via spectral filtering.

    Power spectrum:

    $$
    S(\lvert k \rvert) = (\tau^2 + \lvert k \rvert^2)^{-\alpha}.
    $$

    Args:
        key:   JAX PRNG key.
        N:     Number of grid points per spatial dimension.
        alpha: Spectral decay exponent.  Higher → smoother field.
        tau:   Inverse correlation length.  Higher → more oscillatory field.

    Returns:
        JAX array of shape `(N, N)`.
    """
    freqs        = jnp.fft.fftfreq(N) * N
    KX, KY       = jnp.meshgrid(freqs, freqs, indexing='ij')
    sqrt_S       = (tau ** 2 + KX ** 2 + KY ** 2) ** (-alpha / 2)
    sqrt_S       = sqrt_S.at[0, 0].set(0.0)            # zero DC → zero mean
    noise        = jax.random.normal(key, (N, N))
    return jnp.fft.ifft2(jnp.fft.fft2(noise) * sqrt_S).real   # (N, N)


def _matvec_1d(u_int, kappa, h: float):
    r"""Apply $-\partial_x(\kappa\, \partial_x \cdot)$ via 1-D finite differences on interior nodes.

    Interface permeabilities use arithmetic averaging:

    $$
    \kappa_{i \pm 1/2} = \frac{\kappa_i + \kappa_{i \pm 1}}{2}.
    $$

    Args:
        u_int: Interior solution values, shape `(N-2,)`.
        kappa: Permeability on the full grid, shape `(N,)`.
        h:     Uniform grid spacing.

    Returns:
        `Au_int`, shape `(N-2,)`.
    """
    u_full = jnp.concatenate([jnp.zeros(1), u_int, jnp.zeros(1)])
    ki    = kappa[1:-1]                             # κ at interior nodes
    kx_r  = 0.5 * (ki + kappa[2:  ])               # κ_{i+1/2}
    kx_l  = 0.5 * (ki + kappa[ :-2])               # κ_{i-1/2}
    return (
        (kx_r + kx_l) * u_full[1:-1]
        - kx_r * u_full[2:]
        - kx_l * u_full[:-2]
    ) / h ** 2


def _matvec_2d(u_flat, kappa, h: float, N_int: int):
    r"""Apply $-\nabla \cdot (\kappa\, \nabla \cdot)$ via 2-D FD on interior nodes (five-point stencil).

    Interface permeabilities use arithmetic averaging:

    $$
    \kappa_{i+1/2,\,j} = \frac{\kappa_{i,j} + \kappa_{i+1,j}}{2},
    \quad
    \kappa_{i,\,j+1/2} = \frac{\kappa_{i,j} + \kappa_{i,j+1}}{2},
    \quad \text{etc.}
    $$

    The resulting operator is symmetric positive definite for
    $\kappa > 0$ with homogeneous Dirichlet boundary conditions.

    Args:
        u_flat: Interior solution values flattened, shape `(N_int^2,)`.
        kappa:  Permeability on the full grid, shape `(N, N)`.
        h:      Uniform grid spacing.
        N_int:  Interior grid size per dimension (= N - 2).

    Returns:
        `Au_flat`, shape `(N_int^2,)`.
    """
    u_int  = u_flat.reshape(N_int, N_int)
    u_full = jnp.pad(u_int, 1)                     # zero Dirichlet BC padding
    ki    = kappa[1:-1, 1:-1]                       # κ at interior nodes
    kx_r  = 0.5 * (ki + kappa[2:,   1:-1])         # κ_{i+1/2, j}
    kx_l  = 0.5 * (ki + kappa[:-2,  1:-1])         # κ_{i-1/2, j}
    ky_u  = 0.5 * (ki + kappa[1:-1, 2:  ])         # κ_{i, j+1/2}
    ky_d  = 0.5 * (ki + kappa[1:-1, :-2 ])         # κ_{i, j-1/2}
    Au = (
        (kx_r + kx_l + ky_u + ky_d) * u_full[1:-1, 1:-1]
        - kx_r * u_full[2:,   1:-1]
        - kx_l * u_full[:-2,  1:-1]
        - ky_u * u_full[1:-1, 2:  ]
        - ky_d * u_full[1:-1, :-2 ]
    ) / h ** 2
    return Au.ravel()


def _cg_scan(matvec, b, steps: int, diagonal=None):
    """Jacobi-preconditioned CG with convergence stopping and true residual.

    The diagonal is supplied by the finite-difference stencil. JAX's CG is
    vmap-compatible and stops at convergence rather than consuming the full
    iteration budget. Solves run in float64 to certify the default tolerance.
    """
    from jax.scipy.sparse.linalg import cg

    x, _ = cg(matvec, b, tol=1e-9, atol=0.0, maxiter=steps,
              M=None if diagonal is None else lambda residual: residual / diagonal)
    residual = b - matvec(x)
    return x, jnp.vdot(residual, residual).real


def _relative_residual(residual_sq, b):
    r"""$\lVert r \rVert / \lVert b \rVert$ after the final CG iteration.

    CG starts from $x = 0$, so this is the factor by which the initial
    residual was reduced -- the only honest measure of whether the generated
    "ground truth" actually solves the PDE.
    """
    return jnp.sqrt(residual_sq) / jnp.maximum(jnp.linalg.norm(b), 1e-30)


def _solve_one_1d(kappa, f, h: float, N: int, cg_steps: int):
    """Solve one 1-D Darcy sample.

    Args:
        kappa: shape `(1, N)`
        f:     shape `(1, N)`

    Returns:
        `(u, relative_residual)`: solution of shape `(1, N)` with u=0 at the
        endpoints, and the CG relative residual for that sample.
    """
    f_int = f[0, 1:-1]                             # interior RHS, (N - 2,)

    def matvec(u, diagonal_only=False):
        if diagonal_only:
            ki = kappa[0, 1:-1]
            return (ki + 0.5 * (kappa[0, 2:] + kappa[0, :-2])) / h ** 2
        return _matvec_1d(u, kappa[0], h)

    u_int, residual_sq = _cg_scan(matvec, f_int, cg_steps, matvec(f_int, diagonal_only=True))
    u_full   = jnp.concatenate([jnp.zeros(1), u_int, jnp.zeros(1)])
    return u_full[jnp.newaxis], _relative_residual(residual_sq, f_int)


def _solve_one_2d(kappa, f, h: float, N: int, cg_steps: int):
    """Solve one 2-D Darcy sample.

    Args:
        kappa: shape `(1, N, N)`
        f:     shape `(1, N, N)`

    Returns:
        `(u, relative_residual)`: solution of shape `(1, N, N)` with u=0 on all
        edges, and the CG relative residual for that sample.
    """
    N_int      = N - 2
    f_int      = f[0, 1:-1, 1:-1].ravel()          # interior RHS, (N_int²,)

    def matvec(u_flat, diagonal_only=False):
        if diagonal_only:
            k = kappa[0]
            diagonal = (2 * k[1:-1, 1:-1] + 0.5 * (
                k[2:, 1:-1] + k[:-2, 1:-1] + k[1:-1, 2:] + k[1:-1, :-2])) / h ** 2
            return diagonal.ravel()
        return _matvec_2d(u_flat, kappa[0], h, N_int)

    u_int_flat, residual_sq = _cg_scan(matvec, f_int, cg_steps, matvec(f_int, diagonal_only=True))
    u_full        = jnp.pad(u_int_flat.reshape(N_int, N_int), 1)
    return u_full[jnp.newaxis], _relative_residual(residual_sq, f_int)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DarcyConfig(GenerationConfig):
    r"""
    Configuration for the Darcy-flow / variable-coefficient Poisson generator.

    Inherits all base fields from `GenerationConfig` (`num_points`,
    `num_samples`, `seed`, `torch_device`, `obs_noise_std`,
    `obs_mask_fraction`).

    Attributes:
        num_spatial_dims: Spatial dimension (1 or 2).  Default 2.
        domain_extent: Side length of the square/interval domain.  Default
            1.0 ([0,1]^d), which is the standard Darcy benchmark domain.
        kappa_alpha: Spectral exponent $\alpha$ for the $\kappa$
            GRF power spectrum
            $S(\lvert k\rvert) \propto (\tau^2 + \lvert k\rvert^2)^{-\alpha}$.
            Higher → smoother permeability.
            $\alpha = 2.0$ matches the original FNO Darcy benchmark.
        kappa_tau: GRF inverse correlation length $\tau$.  Higher →
            more oscillatory $\kappa$ with smaller features.
        kappa_scale: Standard deviation of $\log\kappa$ before
            exponentiation.  Scale 1.0 gives
            $\kappa \in [e^{-2}, e^{2}] \approx [0.14,\, 7.4]$
            roughly.  Increase for higher contrast between high- and
            low-permeability regions.
        kappa_min: Hard lower bound on $\kappa$
            (positivity / ellipticity floor).
        f_cutoff: Fourier cutoff for the random source term.
        f_amplitude_min: Minimum per-sample amplitude scaling for f.
        f_amplitude_max: Maximum per-sample amplitude scaling for f.
        cg_steps: Maximum Jacobi-preconditioned CG iterations. Solves stop
            at convergence and return float64 solution fields. The final
            diagnostic is recomputed from b - A u, not the recursive residual.
        cg_tolerance: Largest acceptable CG relative residual, checked over
            every generated sample.  `generate()` raises if it is exceeded,
            so an under-converged dataset fails loudly instead of quietly
            becoming wrong ground truth.  Set to `None` to skip the check.
    """
    # Domain — override base defaults for the standard Darcy setting
    num_spatial_dims: int   = 2
    domain_extent:    float = 1.0

    # κ field
    kappa_alpha: float = 2.0
    kappa_tau:   float = 3.0
    kappa_scale: float = 1.0
    kappa_min:   float = 0.1

    # f field
    f_cutoff:        int   = 8
    f_amplitude_min: float = 0.1
    f_amplitude_max: float = 5.0

    # Solver
    cg_steps: int = 2000
    cg_tolerance: Optional[float] = 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DarcyDataset(Dataset):
    """
    PyTorch Dataset for Darcy-flow / variable-coefficient Poisson data.

    Returns samples in the flow-trainer-compatible format:

    **Forward** (`problem='forward'`):

        sample['input']  → (2, *spatial)  cat([κ, f], dim=0)
        sample['target'] → (1, *spatial)  solution u

    **Inverse** (`problem='inverse'`):

        inverse_mode='both':
            sample['input']  → (1, *spatial)  observed u
            sample['target'] → (2, *spatial)  cat([κ, f], dim=0)

        inverse_mode='coefficient':
            sample['input']  → (2, *spatial)  cat([u, f], dim=0)
            sample['target'] → (1, *spatial)  permeability κ

        inverse_mode='source':
            sample['input']  → (2, *spatial)  cat([u, κ], dim=0)
            sample['target'] → (1, *spatial)  source f

    When `obs_mask_fraction < 1.0`, the observation mask is appended to
    `sample['input']` as an extra channel and `sample['obs_mask']` (shape
    `(1, *spatial)`, float, 1 = observed, 0 = hidden) is also returned.

    All raw tensors (κ, f, u) are accessible via `get_raw_data()`.
    """

    def __init__(
        self,
        data: dict,
        problem: Literal['forward', 'inverse'] = 'forward',
        inverse_mode: Literal['both', 'coefficient', 'source'] = 'both',
        metadata: Optional[dict] = None,
        normalizer: Optional[FieldNormalizer] = None,
    ):
        self.data     = data
        self.problem  = problem
        self.inverse_mode = inverse_mode
        self.metadata = metadata or {}
        # Which raw field carries the (possibly masked) observation, if any.
        self.observation_key = self.metadata.get('observation_key')
        self.normalizer = normalizer

        if self.problem not in {'forward', 'inverse'}:
            raise ValueError("problem must be 'forward' or 'inverse'")
        if self.inverse_mode not in {'both', 'coefficient', 'source'}:
            raise ValueError(
                "inverse_mode must be 'both', 'coefficient', or 'source'"
            )

    # Normalization
    def set_normalizer(self, normalizer: Optional[FieldNormalizer]) -> 'DarcyDataset':
        """
        Attach (or clear) a field normalizer applied on every access.

        Pass the *training* split's normalizer to validation and test splits
        so all data is standardized with the same statistics.

        Returns:
            `self`, for chaining.
        """
        self.normalizer = normalizer
        return self

    def _field(self, name: str, idx: int) -> torch.Tensor:
        """Fetch one raw field, standardized when a normalizer is attached."""
        value = self.data[name][idx]
        if self.normalizer is not None:
            value = self.normalizer.normalize(name, value)
            # Re-apply the mask after standardizing: normalization subtracts
            # the field mean, which would turn the "not observed" zeros into a
            # nonzero constant and contradict the obs_mask channel.
            if name == self.observation_key:
                obs_mask = self.data.get('obs_mask')
                if obs_mask is not None:
                    value = value * obs_mask[idx]
        return value

    @property
    def input_fields(self) -> List[str]:
        """Raw field names composing `sample['input']`, in channel order.

        The observation mask is appended to the input as an extra channel, so
        it is listed here too; it carries no statistics and so passes through
        denormalization unchanged.
        """
        if self.problem == 'forward':
            fields = ['kappa', 'source']
        elif self.inverse_mode == 'both':
            fields = ['solution']
        elif self.inverse_mode == 'coefficient':
            fields = ['solution', 'source']
        else:
            fields = ['solution', 'kappa']

        if self.data.get('obs_mask') is not None:
            fields.append('obs_mask')
        return fields

    @property
    def target_fields(self) -> List[str]:
        """Raw field names composing `sample['target']`, in channel order."""
        if self.problem == 'forward':
            return ['solution']
        if self.inverse_mode == 'both':
            return ['kappa', 'source']
        if self.inverse_mode == 'coefficient':
            return ['kappa']
        return ['source']

    def __len__(self) -> int:
        return len(self.data['solution'])

    def __getitem__(self, idx: int) -> dict:
        kappa = self._field('kappa', idx)       # (1, *spatial)
        f     = self._field('source', idx)      # (1, *spatial)
        u     = self._field('solution', idx)    # (1, *spatial)

        if self.problem == 'forward':
            inp    = torch.cat([kappa, f], dim=0)   # (2, *spatial)
            target = u
        elif self.inverse_mode == 'both':
            # Recover both PDE inputs from the observed solution.
            inp    = u
            target = torch.cat([kappa, f], dim=0)   # (2, *spatial)
        elif self.inverse_mode == 'coefficient':
            # Recover κ with source f treated as known conditioning data.
            inp    = torch.cat([u, f], dim=0)       # (2, *spatial)
            target = kappa
        elif self.inverse_mode == 'source':
            # Recover f with coefficient κ treated as known conditioning data.
            inp    = torch.cat([u, kappa], dim=0)   # (2, *spatial)
            target = f
        else:
            raise ValueError(
                f"Unknown inverse_mode: {self.inverse_mode}. "
                "Expected 'both', 'coefficient', or 'source'."
            )

        obs_mask = self.data.get('obs_mask')
        if obs_mask is not None:
            inp = torch.cat([inp, obs_mask[idx]], dim=0)

        sample = {'input': inp, 'target': target}
        if obs_mask is not None:
            sample['obs_mask'] = obs_mask[idx]
        return sample

    def get_stats(self) -> dict:
        """Normalization statistics for each field."""
        return self.metadata.get('stats', {})

    def get_config(self) -> dict:
        """Generation configuration dict."""
        return self.metadata.get('config', {})

    def get_raw_data(self) -> dict:
        """Raw tensor dict with keys `'kappa'`, `'source'`, `'solution'`."""
        return self.data



class DarcyGenerator(ExponaxDatasetGenerator):
    r"""
    Generate Darcy-flow / variable-coefficient Poisson datasets.

    Workflow:

        1. Draw per-sample log-normal $\kappa$ fields from a GRF.
        2. Draw per-sample smooth Fourier source fields.
        3. Solve $-\nabla\cdot(\kappa\,\nabla u)=f$ via FD + fixed-step CG for each sample.
        4. Optionally apply additive noise and/or spatial masking to the
           solution field (for inverse-problem datasets).
        5. Convert to PyTorch tensors and return a `DarcyDataset`.

    Args:
        config: A `DarcyConfig` instance.  Keyword arguments are forwarded
                to `DarcyConfig` if *config* is `None`.

    **Example**

    ```python
    gen = DarcyGenerator(num_points=64)
    train = gen.generate(num_samples=1000, seed=0)
    test  = gen.generate(num_samples=200,  seed=1)
    ```
    """

    config_cls = DarcyConfig
    dataset_cls = DarcyDataset

    def generate(
        self,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        problem: Literal['forward', 'inverse'] = 'forward',
        inverse_mode: Literal['both', 'coefficient', 'source'] = 'both',
    ) -> DarcyDataset:
        """
        Generate a Darcy-flow dataset.

        Args:
            num_samples: Override `config.num_samples`.
            seed:        Override `config.seed`.
            problem:     `'forward'` (κ,f → u) or `'inverse'`.
            inverse_mode: Inverse mapping to use:
                `'both'` for u → (κ,f), `'coefficient'` for (u,f) → κ,
                or `'source'` for (u,κ) → f.

        Returns:
            A `DarcyDataset`.
        """
        cfg, n, s = self.resolve_run(num_samples, seed)
        self.validate_problem(problem)
        N   = cfg.num_points
        d   = cfg.num_spatial_dims
        L   = cfg.domain_extent

        if d not in (1, 2):
            raise ValueError(
                f"DarcyGenerator supports num_spatial_dims ∈ {{1, 2}}; got {d}."
            )

        # Vertex-centred grid: N points on [0, L], spacing h = L/(N-1)
        if N < 3 or cfg.cg_steps < 1:
            raise ValueError("Darcy requires at least 3 grid points and positive cg_steps")
        if not math.isfinite(cfg.kappa_scale) or not math.isfinite(cfg.kappa_min) or cfg.kappa_min <= 0:
            raise ValueError("kappa_scale must be finite and kappa_min positive and finite")
        h = L / (N - 1)

        # Split all keys upfront for full reproducibility
        key = jax.random.PRNGKey(s)
        key, kappa_key, f_key = jax.random.split(key, 3)

        # ── 1. κ fields: per-sample log-normal GRF ────────────────────────
        kappa_sample_keys = jax.random.split(kappa_key, n)

        def make_kappa(k):
            grf = (
                _grf_1d(k, N, cfg.kappa_alpha, cfg.kappa_tau) if d == 1
                else _grf_2d(k, N, cfg.kappa_alpha, cfg.kappa_tau)
            )
            # Standardise then scale → kappa_scale controls log-contrast
            grf   = (grf - grf.mean()) / (grf.std() + 1e-6) * cfg.kappa_scale
            kappa = jnp.maximum(jnp.exp(grf), cfg.kappa_min)
            return kappa[jnp.newaxis]               # (1, *spatial)

        kappas = jax.vmap(make_kappa)(kappa_sample_keys)   # (n, 1, *spatial)
        if not bool(jnp.isfinite(kappas).all()):
            raise RuntimeError("Darcy coefficient generation produced nonfinite values")

        # ── 2. f fields: smooth fixed-cutoff Fourier series ───────────────
        sources = sample_fourier_fields(
            f_key,
            n=n,
            cfg=cfg,
            field=FourierFieldConfig(
                cutoff=cfg.f_cutoff,
                amplitude_min=cfg.f_amplitude_min,
                amplitude_max=cfg.f_amplitude_max,
                normalize=True,
            ),
        )

        # ── 3. Solve for each sample ───────────────────────
        if d == 1:
            def solve_one(kappa, f):
                return _solve_one_1d(kappa, f, h, N, cfg.cg_steps)
        else:
            def solve_one(kappa, f):
                return _solve_one_2d(kappa, f, h, N, cfg.cg_steps)

        # Keep generated ground truth in float64: casting it back to float32
        # can invalidate a strict residual tolerance on fine elliptic grids.
        with enable_x64(True):
            solutions, cg_residuals = jax.vmap(solve_one)(
                kappas.astype(jnp.float64), sources.astype(jnp.float64)
            )
            worst_residual = float(jnp.max(cg_residuals))
            mean_residual = float(jnp.mean(cg_residuals))
        # solutions: (n, 1, *spatial);  cg_residuals: (n,)

        if not math.isfinite(worst_residual):
            raise RuntimeError("Darcy CG produced nonfinite coefficients, solutions, or residuals")
        if cfg.cg_tolerance is not None and worst_residual > cfg.cg_tolerance:
            raise RuntimeError(
                f"Darcy CG did not converge: worst relative residual "
                f"{worst_residual:.3e} over {n} samples exceeds "
                f"cg_tolerance={cfg.cg_tolerance:.1e} after "
                f"{cfg.cg_steps} iterations. These solutions do not solve the "
                f"PDE and must not be used as ground truth. Raise cg_steps "
                f"(preconditioned CG may need more of them as the grid or the "
                f"contrast in kappa grows), or relax cg_tolerance if you "
                f"genuinely accept this error level."
            )

        # ── 4. Convert, optionally augment inverse observations, and wrap ──
        data = self.to_torch_data({
            'kappa': kappas,
            'source': sources,
        })
        data['solution'] = jax_to_torch(solutions, device=cfg.torch_device, dtype=torch.float64)
        augmentation = self.apply_observation_augmentation(
            data,
            observation_key='solution',
            problem=problem,
            n=n,
            seed=s,
        )

        spatial_str = 'x'.join([str(N)] * d)
        logger.info(
            "Generated Darcy %dD dataset: %d samples, %s grid, "
            "κ ~ LogNormal(α=%s, τ=%s, scale=%s), CG steps=%s, "
            "worst CG residual=%.2e",
            d, n, spatial_str, cfg.kappa_alpha, cfg.kappa_tau,
            cfg.kappa_scale, cfg.cg_steps, worst_residual,
        )

        return self.wrap_dataset(
            data,
            problem=problem,
            inverse_mode=inverse_mode,
            extra_stats=augmentation.pop("clean_stats", None),
            extra_metadata={
                **augmentation,
                "grid": {"spacing": h, "periodic": False, "endpoint_included": True},
                "cg_residual_max": worst_residual,
                "cg_residual_mean": mean_residual,
                "cg_steps": cfg.cg_steps,
            },
        )
