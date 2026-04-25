"""
Darcy Flow / Variable-Coefficient Poisson Data Generation
==========================================================

Generates :math:`(\kappa, f) \to u` datasets for the variable-coefficient
elliptic PDE:

.. math::

    -\nabla \cdot (\kappa(x)\, \nabla u(x)) = f(x)
    \quad \text{on } \Omega = [0, L]^d,
    \qquad u = 0 \text{ on } \partial\Omega.

Unlike the constant-coefficient Poisson equation supported by Exponax,
:math:`\kappa` is a random *spatial field*, making this a significantly more
realistic and challenging operator-learning problem.  This is the standard
Darcy-flow benchmark used in FNO, DeepONet, and related literature.

**Solver**
    Second-order finite differences + fixed-step conjugate gradient
    implemented via ``jax.lax.scan``, making the whole pipeline
    ``jax.vmap``-compatible with no Python-level loops over samples.

**κ generation**
    Log-normal Gaussian random fields,

    .. math::

        \kappa = \exp(c \cdot g), \quad
        g \sim \mathcal{N}\!\left(0,\,(\tau^2 I - \Delta)^{-\alpha}\right),

    ensuring :math:`\kappa > 0` everywhere (strict ellipticity) and matching
    the GRF formulation used in the original FNO Darcy benchmark.

**f generation**
    Same Fourier-series + Gaussian-bump mixture used by ``PoissonGenerator``,
    giving a wide distribution of smooth, localised, and oscillatory source
    terms in every batch.

Example::

    gen = DarcyGenerator(num_points=64, kappa_alpha=2.0, kappa_tau=3.0)
    dataset = gen.generate(num_samples=1000, seed=42)

    sample = dataset[0]
    # sample['input']  → (2, 64, 64)  cat([κ, f], channel dim)
    # sample['target'] → (1, 64, 64)  solution u
"""

from dataclasses import dataclass
from typing import Optional, Literal

import torch
from torch.utils.data import Dataset

import jax
import jax.numpy as jnp
import exponax as ex

from .base import GenerationConfig
from .converters import (
    jax_to_torch,
    compute_normalization_stats,
    apply_spectral_cutoff,
    gaussian_bump_ic_batch,
)


def _grf_1d(key, N: int, alpha: float, tau: float):
    """Zero-mean 1-D Gaussian random field via spectral filtering.

    Power spectrum:

    .. math::

        S(k) = (\tau^2 + k^2)^{-\alpha}.

    Args:
        key:   JAX PRNG key.
        N:     Number of grid points.
        alpha: Spectral decay exponent.  Higher → smoother field.
        tau:   Inverse correlation length.  Higher → more oscillatory field.

    Returns:
        JAX array of shape ``(N,)``.
    """
    freqs  = jnp.fft.fftfreq(N) * N                    # integer wavenumbers
    sqrt_S = (tau ** 2 + freqs ** 2) ** (-alpha / 2)
    sqrt_S = sqrt_S.at[0].set(0.0)                     # zero DC → zero mean
    noise  = jax.random.normal(key, (N,))
    return jnp.fft.ifft(jnp.fft.fft(noise) * sqrt_S).real   # (N,)


def _grf_2d(key, N: int, alpha: float, tau: float):
    """Zero-mean 2-D Gaussian random field via spectral filtering.

    Power spectrum:

    .. math::

        S(\lvert k \rvert) = (\tau^2 + \lvert k \rvert^2)^{-\alpha}.

    Args:
        key:   JAX PRNG key.
        N:     Number of grid points per spatial dimension.
        alpha: Spectral decay exponent.  Higher → smoother field.
        tau:   Inverse correlation length.  Higher → more oscillatory field.

    Returns:
        JAX array of shape ``(N, N)``.
    """
    freqs        = jnp.fft.fftfreq(N) * N
    KX, KY       = jnp.meshgrid(freqs, freqs, indexing='ij')
    sqrt_S       = (tau ** 2 + KX ** 2 + KY ** 2) ** (-alpha / 2)
    sqrt_S       = sqrt_S.at[0, 0].set(0.0)            # zero DC → zero mean
    noise        = jax.random.normal(key, (N, N))
    return jnp.fft.ifft2(jnp.fft.fft2(noise) * sqrt_S).real   # (N, N)


def _matvec_1d(u_int, kappa, h: float):
    """Apply :math:`-\\partial_x(\\kappa\\, \\partial_x \\cdot)` via 1-D finite differences on interior nodes.

    Interface permeabilities use arithmetic averaging:

    .. math::

        \kappa_{i \pm 1/2} = \\frac{\kappa_i + \kappa_{i \pm 1}}{2}.

    Args:
        u_int: Interior solution values, shape ``(N-2,)``.
        kappa: Permeability on the full grid, shape ``(N,)``.
        h:     Uniform grid spacing.

    Returns:
        ``Au_int``, shape ``(N-2,)``.
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
    """Apply :math:`-\\nabla \\cdot (\\kappa\\, \\nabla \\cdot)` via 2-D FD on interior nodes (five-point stencil).

    Interface permeabilities use arithmetic averaging:

    .. math::

        \kappa_{i+1/2,\,j} = \\frac{\kappa_{i,j} + \kappa_{i+1,j}}{2},
        \quad
        \kappa_{i,\,j+1/2} = \\frac{\kappa_{i,j} + \kappa_{i,j+1}}{2},
        \quad \text{etc.}

    The resulting operator is symmetric positive definite for
    :math:`\kappa > 0` with homogeneous Dirichlet boundary conditions.

    Args:
        u_flat: Interior solution values flattened, shape ``(N_int^2,)``.
        kappa:  Permeability on the full grid, shape ``(N, N)``.
        h:      Uniform grid spacing.
        N_int:  Interior grid size per dimension (= N - 2).

    Returns:
        ``Au_flat``, shape ``(N_int^2,)``.
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


def _cg_scan(matvec, b, steps: int):
    """Fixed-iteration conjugate gradient via ``jax.lax.scan``.

    Runs exactly *steps* CG iterations with no convergence check, keeping the
    computation graph fully static so that the function is compatible with
    ``jax.vmap``.  The operator *matvec* may close over ``jax.vmap``-traced
    values (e.g., a per-sample :math:`\kappa` field).

    Args:
        matvec: SPD linear operator :math:`A : \\mathbb{R}^n \\to \\mathbb{R}^n`.
        b:      Right-hand side vector, shape ``(n,)``.
        steps:  Number of CG iterations.

    Returns:
        ``(x, final_residual_sq)``: Approximate solution and
        :math:`\\lVert r \\rVert^2` after the last iteration (scalar,
        useful for diagnostics).
    """
    x = jnp.zeros_like(b)
    r = b
    p = b

    def body(carry, _):
        x, r, p = carry
        Ap    = matvec(p)
        rr    = jnp.dot(r, r)
        alpha = rr / (jnp.dot(p, Ap) + 1e-30)
        x     = x + alpha * p
        r_new = r - alpha * Ap
        beta  = jnp.dot(r_new, r_new) / (rr + 1e-30)
        p_new = r_new + beta * p
        return (x, r_new, p_new), jnp.dot(r_new, r_new)

    (x, _, _), residuals = jax.lax.scan(body, (x, r, p), None, length=steps)
    return x, residuals[-1]


def _solve_one_1d(kappa, f, h: float, N: int, cg_steps: int):
    """Solve one 1-D Darcy sample.

    Args:
        kappa: shape ``(1, N)``
        f:     shape ``(1, N)``

    Returns:
        Solution u, shape ``(1, N)``, with u=0 at endpoints.
    """
    N_int = N - 2
    f_int = f[0, 1:-1]                             # interior RHS, (N_int,)

    def matvec(u):
        return _matvec_1d(u, kappa[0], h)

    u_int, _ = _cg_scan(matvec, f_int, cg_steps)
    u_full   = jnp.concatenate([jnp.zeros(1), u_int, jnp.zeros(1)])
    return u_full[jnp.newaxis]                      # (1, N)


def _solve_one_2d(kappa, f, h: float, N: int, cg_steps: int):
    """Solve one 2-D Darcy sample.

    Args:
        kappa: shape ``(1, N, N)``
        f:     shape ``(1, N, N)``

    Returns:
        Solution u, shape ``(1, N, N)``, with u=0 on all edges.
    """
    N_int      = N - 2
    f_int      = f[0, 1:-1, 1:-1].ravel()          # interior RHS, (N_int²,)

    def matvec(u_flat):
        return _matvec_2d(u_flat, kappa[0], h, N_int)

    u_int_flat, _ = _cg_scan(matvec, f_int, cg_steps)
    u_full        = jnp.pad(u_int_flat.reshape(N_int, N_int), 1)
    return u_full[jnp.newaxis]                      # (1, N, N)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DarcyConfig(GenerationConfig):
    """
    Configuration for the Darcy-flow / variable-coefficient Poisson generator.

    Inherits all base fields from ``GenerationConfig`` (``num_points``,
    ``num_samples``, ``seed``, ``torch_device``, ``obs_noise_std``,
    ``obs_mask_fraction``).

    Attributes:
        num_spatial_dims: Spatial dimension (1 or 2).  Default 2.
        domain_extent: Side length of the square/interval domain.  Default
            1.0 ([0,1]^d), which is the standard Darcy benchmark domain.
        kappa_alpha: Spectral exponent :math:`\\alpha` for the :math:`\kappa`
            GRF power spectrum
            :math:`S(\\lvert k\\rvert) \\propto (\\tau^2 + \\lvert k\\rvert^2)^{-\\alpha}`.
            Higher → smoother permeability.
            :math:`\\alpha = 2.0` matches the original FNO Darcy benchmark.
        kappa_tau: GRF inverse correlation length :math:`\\tau`.  Higher →
            more oscillatory :math:`\kappa` with smaller features.
        kappa_scale: Standard deviation of :math:`\\log\kappa` before
            exponentiation.  Scale 1.0 gives
            :math:`\kappa \\in [e^{-2}, e^{2}] \\approx [0.14,\, 7.4]`
            roughly.  Increase for higher contrast between high- and
            low-permeability regions.
        kappa_min: Hard lower bound on :math:`\kappa`
            (positivity / ellipticity floor).
        f_cutoff_min: Minimum spectral cutoff for the random source term.
        f_cutoff_max: Maximum spectral cutoff for the random source term.
        f_amplitude_min: Minimum per-sample amplitude scaling for f.
        f_amplitude_max: Maximum per-sample amplitude scaling for f.
        f_gaussian_bump_prob: Probability of drawing f from the
            Gaussian-bump generator rather than Fourier series (0.5 = equal
            mixture).
        f_gaussian_bump_max_bumps: Maximum number of bumps per source sample.
        cg_steps: Fixed number of conjugate-gradient iterations used to solve
            the linear system.  500 is conservative for a 64×64 grid; reduce
            to ~100 for fast prototyping.
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
    f_cutoff_min:             int   = 3
    f_cutoff_max:             int   = 20
    f_amplitude_min:          float = 0.1
    f_amplitude_max:          float = 5.0
    f_gaussian_bump_prob:     float = 0.5
    f_gaussian_bump_max_bumps: int  = 5

    # Solver
    cg_steps: int = 500


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DarcyDataset(Dataset):
    """
    PyTorch Dataset for Darcy-flow / variable-coefficient Poisson data.

    Returns samples in the flow-trainer-compatible format:

    **Forward** (``problem='forward'``)::

        sample['input']  → (2, *spatial)  cat([κ, f], dim=0)
        sample['target'] → (1, *spatial)  solution u

    **Inverse** (``problem='inverse'``)::

        inverse_mode='both':
            sample['input']  → (1, *spatial)  observed u
            sample['target'] → (2, *spatial)  cat([κ, f], dim=0)

        inverse_mode='coefficient':
            sample['input']  → (2, *spatial)  cat([u, f], dim=0)
            sample['target'] → (1, *spatial)  permeability κ

        inverse_mode='source':
            sample['input']  → (2, *spatial)  cat([u, κ], dim=0)
            sample['target'] → (1, *spatial)  source f

    When ``obs_mask_fraction < 1.0``, the observation mask is appended to
    ``sample['input']`` as an extra channel and ``sample['obs_mask']`` (shape
    ``(1, *spatial)``, float, 1 = observed, 0 = hidden) is also returned.

    All raw tensors (κ, f, u) are accessible via ``get_raw_data()``.
    """

    def __init__(
        self,
        data: dict,
        problem: Literal['forward', 'inverse'] = 'forward',
        inverse_mode: Literal['both', 'coefficient', 'source'] = 'coefficient',
        metadata: Optional[dict] = None,
    ):
        self.data     = data
        self.problem  = problem
        self.inverse_mode = inverse_mode
        self.metadata = metadata or {}

        if self.problem != 'inverse' and self.inverse_mode != 'both':
            raise ValueError("inverse_mode is only used when problem='inverse'")

    def __len__(self) -> int:
        return len(self.data['solution'])

    def __getitem__(self, idx: int) -> dict:
        kappa = self.data['kappa'][idx]       # (1, *spatial)
        f     = self.data['source'][idx]      # (1, *spatial)
        u     = self.data['solution'][idx]    # (1, *spatial)

        if self.problem == 'forward':
            inp    = torch.cat([kappa, f], dim=0)   # (2, *spatial)
            target = u
        elif self.inverse_mode == 'both':
            # Recover both PDE inputs from the observed solution. #HARDEST
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
        """Raw tensor dict with keys ``'kappa'``, ``'source'``, ``'solution'``."""
        return self.data



class DarcyGenerator:
    """
    Generate Darcy-flow / variable-coefficient Poisson datasets.

    Workflow:

        1. Draw per-sample log-normal :math:`\kappa` fields from a GRF.
        2. Draw per-sample :math:`f` fields (Fourier series + Gaussian-bump mixture).
        3. Solve :math:`-\\nabla\\cdot(\kappa\\,\\nabla u)=f` via FD + fixed-step CG for each sample.
        4. Optionally apply additive noise and/or spatial masking to the
           solution field (for inverse-problem datasets).
        5. Convert to PyTorch tensors and return a ``DarcyDataset``.

    Args:
        config: A ``DarcyConfig`` instance.  Keyword arguments are forwarded
                to ``DarcyConfig`` if *config* is ``None``.

    Example::

        gen = DarcyGenerator(num_points=64)
        train = gen.generate(num_samples=1000, seed=0)
        test  = gen.generate(num_samples=200,  seed=1)
    """

    def __init__(self, config: Optional[DarcyConfig] = None, **kwargs):
        if config is None:
            config = DarcyConfig(**kwargs)
        self.config = config

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
            num_samples: Override ``config.num_samples``.
            seed:        Override ``config.seed``.
            problem:     ``'forward'`` (κ,f → u) or ``'inverse'``.
            inverse_mode: Inverse mapping to use:
                ``'both'`` for u → (κ,f), ``'coefficient'`` for (u,f) → κ,
                or ``'source'`` for (u,κ) → f.

        Returns:
            A ``DarcyDataset``.
        """
        cfg = self.config
        n   = num_samples or cfg.num_samples
        s   = seed if seed is not None else cfg.seed
        N   = cfg.num_points
        d   = cfg.num_spatial_dims
        L   = cfg.domain_extent

        if d not in (1, 2):
            raise ValueError(
                f"DarcyGenerator supports num_spatial_dims ∈ {{1, 2}}; got {d}."
            )

        # Vertex-centred grid: N points on [0, L], spacing h = L/(N-1)
        h = L / (N - 1)

        # Split all keys upfront for full reproducibility
        key = jax.random.PRNGKey(s)
        key, kappa_key, f_key, f_cut_key, f_amp_key, f_bump_key, f_mix_key = (
            jax.random.split(key, 7)
        )

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

        # ── 2. f fields: Fourier series + Gaussian-bump mixture ───────────
        ic_gen = ex.ic.RandomTruncatedFourierSeries(
            num_spatial_dims=d,
            cutoff=cfg.f_cutoff_max,
            max_one=False,
        )
        sources = ex.build_ic_set(ic_gen, num_points=N, num_samples=n, key=f_key)
        # (n, 1, *spatial)

        # Per-sample random spectral cutoff
        f_cutoffs = jax.random.randint(
            f_cut_key, (n,), minval=cfg.f_cutoff_min, maxval=cfg.f_cutoff_max + 1
        )
        sources = jax.vmap(
            lambda f, c: apply_spectral_cutoff(f, c, d)
        )(sources, f_cutoffs)

        # Gaussian bump mixing (single-channel only)
        if sources.shape[1] == 1:
            sources_bumps = gaussian_bump_ic_batch(
                f_bump_key,
                n=n,
                num_points=N,
                domain_extent=L,
                num_spatial_dims=d,
                max_bumps=cfg.f_gaussian_bump_max_bumps,
            )  # (n, 1, *spatial)
            mix_mask = jax.random.uniform(f_mix_key, (n,)) < cfg.f_gaussian_bump_prob
            mask     = mix_mask.reshape((n,) + (1,) * (sources.ndim - 1))
            sources  = jnp.where(mask, sources_bumps, sources)

        # Per-sample amplitude scaling
        amp_shape = (n,) + (1,) * (sources.ndim - 1)
        f_amps    = jax.random.uniform(
            f_amp_key, amp_shape,
            minval=cfg.f_amplitude_min,
            maxval=cfg.f_amplitude_max,
        )
        sources = sources * f_amps

        # ── 3. Solve for each sample ───────────────────────
        if d == 1:
            def solve_one(kappa, f):
                return _solve_one_1d(kappa, f, h, N, cfg.cg_steps)
        else:
            def solve_one(kappa, f):
                return _solve_one_2d(kappa, f, h, N, cfg.cg_steps)

        solutions = jax.vmap(solve_one)(kappas, sources)   # (n, 1, *spatial)

        # ── 4. Convert to PyTorch ─────────────────────────────────────────
        kappas_pt  = jax_to_torch(kappas,    device=cfg.torch_device)
        sources_pt = jax_to_torch(sources,   device=cfg.torch_device)
        sols_pt    = jax_to_torch(solutions, device=cfg.torch_device)

        # Inverse-problem augmentations: noise + spatial masking
        if problem == 'inverse':
            if cfg.obs_noise_std > 0.0:
                sols_pt = sols_pt + cfg.obs_noise_std * torch.randn_like(sols_pt)
            if cfg.obs_mask_fraction < 1.0:
                spatial_shape = (N,) * d
                mask_gen  = torch.Generator().manual_seed(s)
                obs_mask  = (
                    torch.rand((n, 1) + spatial_shape, generator=mask_gen)
                    < cfg.obs_mask_fraction
                ).float().to(device=cfg.torch_device)
                sols_pt  = sols_pt * obs_mask
            else:
                obs_mask = None
        else:
            obs_mask = None

        # ── 5. Assemble dataset ───────────────────────────────────────────
        data = {
            'kappa':    kappas_pt,
            'source':   sources_pt,
            'solution': sols_pt,
        }
        if obs_mask is not None:
            data['obs_mask'] = obs_mask

        stats = {
            'kappa':    compute_normalization_stats(kappas_pt),
            'source':   compute_normalization_stats(sources_pt),
            'solution': compute_normalization_stats(sols_pt),
        }
        metadata = {
            'stats':  stats,
            'config': cfg.to_dict(),
        }

        spatial_str = 'x'.join([str(N)] * d)
        print(
            f"Generated Darcy {d}D dataset: {n} samples, {spatial_str} grid, "
            f"κ ~ LogNormal(α={cfg.kappa_alpha}, τ={cfg.kappa_tau}, "
            f"scale={cfg.kappa_scale}), CG steps={cfg.cg_steps}"
        )

        return DarcyDataset(
            data,
            problem=problem,
            inverse_mode=inverse_mode,
            metadata=metadata,
        )
