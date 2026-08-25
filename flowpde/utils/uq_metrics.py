"""
Uncertainty Quantification Metrics
===================================

A conditional flow produces a *distribution* :math:`p(u \\mid f)`, not a point
prediction.  Relative L2 of the ensemble mean says nothing about whether that
distribution is any good: a model can produce diverse, plausible-looking
samples whose spread bears no relation to its actual error.  These metrics ask
the harder question — **is the predicted distribution calibrated?**

Three families are provided.

**Calibration** — does the stated uncertainty match observed error?

- :func:`credible_interval_coverage` — fraction of truth inside the central
  :math:`\\alpha` interval.  A calibrated model covers 90% of the truth with
  its 90% interval.
- :func:`reliability_curve` — coverage across many levels; the diagonal is
  perfect calibration.
- :func:`rank_histogram` — where the truth ranks among ensemble members.
  Flat means calibrated, U-shaped means under-dispersed (over-confident),
  dome-shaped means over-dispersed.  Standard in ensemble weather
  verification and rarely used in ML-for-PDE work.
- :func:`spread_skill_ratio` — ensemble spread divided by the error of the
  ensemble mean.  Should be about 1.

**Proper scoring rules** — single numbers that reward accuracy *and*
calibration together, and cannot be gamed by lying about uncertainty.

- :func:`crps_ensemble` — pointwise; the standard probabilistic analogue of
  MAE.
- :func:`energy_score` — the multivariate generalization.  Use it: pointwise
  CRPS is blind to spatial correlation, and a model that gets marginals right
  while producing spatially incoherent fields scores well on CRPS and badly
  here.

**Decomposition** — where does the uncertainty come from?

- :func:`variance_decomposition` — splits total variance into aleatoric
  (within-model sampling spread) and epistemic (disagreement across
  independently trained models) via the law of total variance.
- :func:`error_spread_correlation` — does predicted spread actually predict
  error?  The practical question for using uncertainty to flag bad
  predictions.

Convention: ``samples`` has shape ``(K, B, C, *spatial)`` with ``K`` ensemble
members, matching :func:`~flowpde.utils.metrics.ensemble_relative_l2`.
A list of ``(B, C, *spatial)`` tensors is also accepted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

SampleInput = Union[Tensor, Sequence[Tensor]]


def _stack(samples: SampleInput) -> Tensor:
    """Normalize input to a ``(K, B, C, *spatial)`` tensor."""
    if isinstance(samples, Tensor):
        if samples.dim() < 2:
            raise ValueError(
                f"samples must have shape (K, B, ...), got {tuple(samples.shape)}"
            )
        return samples
    if len(samples) == 0:
        raise ValueError("samples must be a non-empty sequence of tensors.")
    return torch.stack(list(samples), dim=0)


def _flatten_fields(x: Tensor) -> Tensor:
    """Collapse everything after the batch dimension: ``(B, ...) -> (B, D)``."""
    return x.reshape(x.shape[0], -1)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────


def credible_interval_coverage(
    samples: SampleInput,
    target: Tensor,
    level: float = 0.9,
) -> Tensor:
    """Fraction of ground-truth values inside the central credible interval.

    For a calibrated model this equals ``level``.  Below it means the model is
    over-confident (the usual failure); above it means over-dispersed.

    .. warning::

        **Empirical quantiles from a finite ensemble under-cover**, and the
        effect grows with the level.  For a perfectly calibrated ensemble:

        =======  ========  ========  ========
        ``K``    0.5       0.9       0.99
        =======  ========  ========  ========
        50       0.479     0.871     0.956
        200      0.493     0.893     0.978
        1000     0.498     0.901     0.986
        =======  ========  ========  ========

        So a 99% interval built from 50 samples covers about 96% *even when
        the model is exactly right*.  Reading that as over-confidence is an
        artefact of the estimator, not a property of the model.  Use at least
        ~200 members for levels up to 0.9, and avoid reporting 0.99 coverage
        below ~1000.  When comparing models, hold ``K`` fixed — the bias
        cancels.

    Args:
        samples: ``(K, B, C, *spatial)`` ensemble, or a list of ``K`` tensors.
        target: Ground truth, ``(B, C, *spatial)``.
        level: Nominal coverage in (0, 1), e.g. 0.9.

    Returns:
        Scalar tensor — empirical coverage, comparable directly to ``level``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    stacked = _stack(samples)
    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q

    lower = torch.quantile(stacked, lower_q, dim=0)
    upper = torch.quantile(stacked, upper_q, dim=0)

    inside = (target >= lower) & (target <= upper)
    return inside.float().mean()


def reliability_curve(
    samples: SampleInput,
    target: Tensor,
    levels: Optional[Sequence[float]] = None,
) -> Dict[str, List[float]]:
    """Empirical coverage across a range of nominal levels.

    Plotting ``empirical`` against ``nominal`` gives the reliability diagram;
    the diagonal is perfect calibration, below it is over-confidence.

    Args:
        samples: ``(K, B, C, *spatial)`` ensemble.
        target: Ground truth, ``(B, C, *spatial)``.
        levels: Nominal levels.  Defaults to 0.1 … 0.9 in steps of 0.1.

    Returns:
        ``{'nominal': [...], 'empirical': [...], 'calibration_error': float}``
        where the error is the mean absolute gap from the diagonal.
    """
    levels = list(levels) if levels is not None else [i / 10 for i in range(1, 10)]
    stacked = _stack(samples)

    empirical = [
        credible_interval_coverage(stacked, target, level).item() for level in levels
    ]
    calibration_error = sum(
        abs(e - n) for e, n in zip(empirical, levels)
    ) / len(levels)

    return {
        "nominal": levels,
        "empirical": empirical,
        "calibration_error": calibration_error,
    }


def rank_histogram(
    samples: SampleInput,
    target: Tensor,
    normalize: bool = True,
) -> Tensor:
    """Histogram of the truth's rank among ensemble members.

    At each location, count how many ensemble members fall below the truth,
    giving a rank in ``[0, K]``.  For a calibrated ensemble the truth is
    exchangeable with the members, so ranks are uniform and the histogram is
    flat.  A U shape means the truth often falls outside the ensemble
    (under-dispersed); a dome means the ensemble is too wide.

    Args:
        samples: ``(K, B, C, *spatial)`` ensemble.
        target: Ground truth, ``(B, C, *spatial)``.
        normalize: Return frequencies rather than counts.

    Returns:
        Tensor of length ``K + 1``.  Compare against ``1 / (K + 1)`` per bin.
    """
    stacked = _stack(samples)
    num_members = stacked.shape[0]

    ranks = (stacked < target.unsqueeze(0)).sum(dim=0).flatten()
    histogram = torch.bincount(ranks, minlength=num_members + 1).float()

    if normalize:
        histogram = histogram / histogram.sum().clamp(min=1.0)
    return histogram


def spread_skill_ratio(
    samples: SampleInput,
    target: Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Ensemble spread compared against the error of the ensemble mean.

    A calibrated ensemble has spread comparable to its own error.  With a
    finite ensemble of ``K`` members the expected ratio is
    :math:`\\sqrt{(K+1)/K}` rather than exactly 1, so a bias-adjusted value is
    reported alongside the raw one.

    Returns:
        ``{'spread', 'skill', 'ratio', 'adjusted_ratio'}``.  ``adjusted_ratio``
        below 1 means over-confident, above 1 means over-dispersed.
    """
    stacked = _stack(samples)
    num_members = stacked.shape[0]
    if num_members < 2:
        raise ValueError("spread_skill_ratio needs at least 2 ensemble members.")

    spread = stacked.var(dim=0, unbiased=True).mean().sqrt()
    skill = (stacked.mean(dim=0) - target).pow(2).mean().sqrt()

    ratio = (spread / skill.clamp(min=eps)).item()
    correction = ((num_members + 1) / num_members) ** 0.5

    return {
        "spread": spread.item(),
        "skill": skill.item(),
        "ratio": ratio,
        "adjusted_ratio": ratio / correction,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Proper scoring rules
# ─────────────────────────────────────────────────────────────────────────────


def crps_ensemble(samples: SampleInput, target: Tensor) -> Tensor:
    """Continuous Ranked Probability Score, pointwise, fair estimator.

    .. math::

        \\mathrm{CRPS} = \\mathbb{E}|X - y|
                       - \\tfrac{1}{2} \\mathbb{E}|X - X'|

    A *proper* scoring rule: it is minimized only by the true predictive
    distribution, so a model cannot improve it by misreporting its
    uncertainty.  For a deterministic forecast it reduces exactly to MAE,
    which makes it directly comparable against a point-prediction baseline.

    The fair (unbiased) estimator is used, dividing the second term by
    ``K(K-1)`` rather than ``K^2``; the biased version rewards small
    ensembles for being artificially narrow.

    Computed via the sorted formulation, which is ``O(K log K)`` and avoids
    materializing the ``K x K`` pairwise differences.

    Args:
        samples: ``(K, B, C, *spatial)`` ensemble.
        target: Ground truth, ``(B, C, *spatial)``.

    Returns:
        Scalar tensor — CRPS averaged over the batch and field, lower better.
    """
    stacked = _stack(samples)
    num_members = stacked.shape[0]
    if num_members < 2:
        raise ValueError("crps_ensemble needs at least 2 ensemble members.")

    absolute_error = (stacked - target.unsqueeze(0)).abs().mean(dim=0)

    # sum_ij |x_i - x_j| = 2 * sum_i (2i - K - 1) * x_(i)   for 1-indexed i
    ordered, _ = torch.sort(stacked, dim=0)
    weights = (
        2 * torch.arange(1, num_members + 1, device=stacked.device, dtype=stacked.dtype)
        - num_members
        - 1
    )
    weights = weights.view(-1, *([1] * (ordered.dim() - 1)))
    pairwise_term = (weights * ordered).sum(dim=0) / (num_members * (num_members - 1))

    return (absolute_error - pairwise_term).mean()


def energy_score(samples: SampleInput, target: Tensor) -> Tensor:
    """Energy score — the multivariate generalization of CRPS.

    .. math::

        \\mathrm{ES} = \\mathbb{E}\\lVert X - y \\rVert_2
                     - \\tfrac{1}{2} \\mathbb{E}\\lVert X - X' \\rVert_2

    Norms are taken over the whole field, so unlike pointwise CRPS this is
    sensitive to **spatial structure**.  A model that reproduces every
    pointwise marginal correctly but generates spatially incoherent fields
    scores well on CRPS and poorly here — which is exactly the failure mode
    worth catching in a PDE surrogate, where solutions are smooth and
    correlated by construction.

    Args:
        samples: ``(K, B, C, *spatial)`` ensemble.
        target: Ground truth, ``(B, C, *spatial)``.

    Returns:
        Scalar tensor, lower is better.
    """
    stacked = _stack(samples)
    num_members = stacked.shape[0]
    if num_members < 2:
        raise ValueError("energy_score needs at least 2 ensemble members.")

    flat = stacked.reshape(num_members, stacked.shape[1], -1)   # (K, B, D)
    flat_target = _flatten_fields(target)                        # (B, D)

    error_term = (flat - flat_target.unsqueeze(0)).norm(dim=2).mean(dim=0)  # (B,)

    # Pairwise distances between members: (K, K, B) — cheap, K is small.
    diffs = flat.unsqueeze(1) - flat.unsqueeze(0)
    pairwise = diffs.norm(dim=3).sum(dim=(0, 1)) / (num_members * (num_members - 1))

    return (error_term - 0.5 * pairwise).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition and diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def variance_decomposition(
    model_samples: Sequence[SampleInput],
) -> Dict[str, float]:
    """Split predictive variance into aleatoric and epistemic parts.

    By the law of total variance, for models :math:`m` and samples :math:`x`,

    .. math::

        \\mathrm{Var}[x] = \\underbrace{\\mathbb{E}_m[\\mathrm{Var}[x \\mid m]]}
                                       _{\\text{aleatoric}}
                         + \\underbrace{\\mathrm{Var}_m[\\mathbb{E}[x \\mid m]]}
                                       _{\\text{epistemic}}

    Aleatoric is the spread the flow produces for a *fixed* model — genuine
    posterior uncertainty on an ill-posed inverse problem.  Epistemic is
    disagreement between independently trained models — reducible with more
    data or better training.

    The distinction matters for interpretation: on a **forward** problem the
    map is deterministic, its posterior is a Dirac, and *all* observed spread
    is model error rather than physical uncertainty.  Reporting it as
    "uncertainty" without this split is misleading.

    Args:
        model_samples: One ensemble per independently trained model, each
            ``(K, B, C, *spatial)``.

    Returns:
        ``{'aleatoric', 'epistemic', 'total', 'epistemic_fraction'}`` as
        variances (not standard deviations).
    """
    if len(model_samples) < 2:
        raise ValueError(
            "variance_decomposition needs at least 2 independently trained "
            "models to estimate epistemic uncertainty."
        )

    stacked = torch.stack([_stack(s) for s in model_samples], dim=0)  # (M, K, B, ...)

    within_variance = stacked.var(dim=1, unbiased=True)   # (M, B, ...)
    per_model_mean = stacked.mean(dim=1)                  # (M, B, ...)

    aleatoric = within_variance.mean().item()
    epistemic = per_model_mean.var(dim=0, unbiased=True).mean().item()
    total = aleatoric + epistemic

    return {
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total,
        "epistemic_fraction": epistemic / total if total > 0 else 0.0,
    }


def _spearman(a: Tensor, b: Tensor) -> Tensor:
    """Spearman rank correlation between two 1-D tensors."""

    def ranks(x: Tensor) -> Tensor:
        order = x.argsort()
        result = torch.empty_like(order, dtype=x.dtype)
        result[order] = torch.arange(len(x), device=x.device, dtype=x.dtype)
        return result

    rank_a, rank_b = ranks(a), ranks(b)
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denominator = (rank_a.norm() * rank_b.norm()).clamp(min=1e-12)
    return (rank_a * rank_b).sum() / denominator


def error_spread_correlation(
    samples: SampleInput,
    target: Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Does predicted spread actually predict error?

    The practical test of an uncertainty estimate: if you flag the
    highest-variance predictions, do you catch the worst ones?  A model can be
    globally well-calibrated yet assign uncertainty that is uninformative
    per-sample, in which case it cannot be used to triage predictions.

    Also reports ``top_decile_error_ratio``: the mean error of the 10% of
    samples with the largest spread, divided by the overall mean error.
    Values above 1 mean high-variance predictions really are worse.

    Args:
        samples: ``(K, B, C, *spatial)`` ensemble.
        target: Ground truth, ``(B, C, *spatial)``.

    Returns:
        ``{'spearman', 'top_decile_error_ratio'}``.
    """
    stacked = _stack(samples)
    batch_size = stacked.shape[1]
    if batch_size < 4:
        raise ValueError("error_spread_correlation needs at least 4 samples.")

    spread = _flatten_fields(stacked.std(dim=0)).mean(dim=1)          # (B,)
    mean_prediction = stacked.mean(dim=0)
    error = _flatten_fields((mean_prediction - target).pow(2)).mean(dim=1).sqrt()

    correlation = _spearman(spread, error).item()

    num_top = max(1, batch_size // 10)
    top_indices = spread.argsort(descending=True)[:num_top]
    ratio = (error[top_indices].mean() / error.mean().clamp(min=eps)).item()

    return {"spearman": correlation, "top_decile_error_ratio": ratio}


class UQMetrics:
    """Compute the standard UQ suite in one call.

    Example::

        uq = UQMetrics(levels=[0.5, 0.9])
        results = uq(samples, target)   # samples: (K, B, C, *spatial)
    """

    def __init__(self, levels: Optional[Sequence[float]] = None):
        self.levels = list(levels) if levels is not None else [0.5, 0.9, 0.95]

    def __call__(self, samples: SampleInput, target: Tensor) -> Dict[str, float]:
        stacked = _stack(samples)
        results: Dict[str, float] = {
            "crps": crps_ensemble(stacked, target).item(),
            "energy_score": energy_score(stacked, target).item(),
        }
        for level in self.levels:
            key = f"coverage_{int(round(level * 100))}"
            results[key] = credible_interval_coverage(stacked, target, level).item()

        results.update(spread_skill_ratio(stacked, target))
        if stacked.shape[1] >= 4:
            results.update(error_spread_correlation(stacked, target))
        return results

    def __repr__(self) -> str:
        return f"UQMetrics(levels={self.levels})"
