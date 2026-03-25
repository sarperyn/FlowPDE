"""
PDE Operator Evaluation Metrics
================================

Standard metrics used in neural PDE operator benchmarking literature,
following the conventions of the FNO paper (Kovachki et al., 2021) and
subsequent works (DeepONet, UQNO, etc.).

Primary metric
--------------
**Relative L2 error** (also called "relative ℓ₂ error" or "nRMSE" in some
papers) is the single number reported in virtually every neural operator paper:

    ε = ‖û - u‖₂ / ‖u‖₂

It is computed **per sample** (norms taken over channels + spatial dims) and
then averaged over the batch.  This removes scale-dependence so errors are
comparable across PDEs with very different solution magnitudes.

Secondary metrics
-----------------
- **H1 semi-norm error**: penalises gradient mismatch on top of value mismatch.
  Meaningful for smooth PDEs (Poisson, Darcy) where solutions are in H¹.
  Discretised via first-order finite differences.
- **MAE** (Mean Absolute Error): complementary to L2, less sensitive to
  outliers.
- **Max pointwise error** / **relative max error**: worst-case analysis.

Flow-model-specific metrics
---------------------------
- **Ensemble relative L2**: generate N samples per condition, average the
  predictions, then compute relative L2 of the ensemble mean.  This separates
  the model's mean-prediction quality from its uncertainty.

No external dependencies beyond PyTorch are required.  The ``neuraloperator``
library provides a similar ``LpLoss`` class, but implementing these here keeps
the dependency footprint minimal and makes the definitions transparent.

Usage::

    from flowpde.utils.metrics import relative_l2_error, EvalMetrics

    # Single batch
    err = relative_l2_error(pred, target)   # scalar tensor

    # All metrics at once
    em = EvalMetrics()
    results = em(pred, target)   # dict
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_dims(x: Tensor):
    """Return the tuple of all non-batch, non-channel dimensions."""
    return tuple(range(2, x.ndim))


def _norm(x: Tensor, p: float = 2.0) -> Tensor:
    """Lp norm over (channels, *spatial), returning shape (batch,)."""
    dims = tuple(range(1, x.ndim))
    if p == 2.0:
        return x.pow(2).sum(dim=dims).sqrt()
    return x.abs().pow(p).sum(dim=dims).pow(1.0 / p)


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics (functional API)
# ─────────────────────────────────────────────────────────────────────────────

def relative_l2_error(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative L2 error — the primary neural-operator benchmark metric.

    .. math::

        \\varepsilon = \\frac{1}{N} \\sum_{i=1}^{N}
            \\frac{\\lVert \\hat{u}_i - u_i \\rVert_2}
                  {\\lVert u_i \\rVert_2 + \\epsilon}

    Norms are taken over all non-batch dimensions (channels + spatial).

    Args:
        pred:   Predicted tensor, shape ``(B, C, *spatial)``.
        target: Ground-truth tensor, same shape.
        eps:    Small constant added to the denominator for numerical
                stability (prevents division by zero for near-zero targets).

    Returns:
        Scalar tensor — the mean relative L2 error over the batch.
    """
    diff_norm   = _norm(pred - target, p=2.0)
    target_norm = _norm(target,        p=2.0).clamp(min=eps)
    return (diff_norm / target_norm).mean()


def relative_l2_error_batch(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Per-sample relative L2 errors.

    Like :func:`relative_l2_error` but returns a ``(B,)`` tensor instead of
    averaging, useful for inspecting the error distribution.
    """
    diff_norm   = _norm(pred - target, p=2.0)
    target_norm = _norm(target,        p=2.0).clamp(min=eps)
    return diff_norm / target_norm


def mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error averaged over all dimensions including batch.

    Args:
        pred:   Predicted tensor, shape ``(B, C, *spatial)``.
        target: Ground-truth tensor, same shape.

    Returns:
        Scalar tensor.
    """
    return F.mse_loss(pred, target)


def mae(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Absolute Error averaged over all dimensions including batch.

    Less sensitive to large outliers than MSE.

    Args:
        pred:   Predicted tensor, shape ``(B, C, *spatial)``.
        target: Ground-truth tensor, same shape.

    Returns:
        Scalar tensor.
    """
    return F.l1_loss(pred, target)


def max_pointwise_error(pred: Tensor, target: Tensor) -> Tensor:
    """Maximum absolute pointwise error over the batch.

    Useful as a worst-case measure — reports the single largest error
    across all samples, channels, and spatial locations.

    Args:
        pred:   Predicted tensor, shape ``(B, C, *spatial)``.
        target: Ground-truth tensor, same shape.

    Returns:
        Scalar tensor.
    """
    return (pred - target).abs().max()


def relative_max_error(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative maximum pointwise error (L∞ norm, normalised by target range).

    Normalises by the standard deviation of each sample's target field so
    the result is comparable across PDEs with different solution scales.

    .. math::

        \\varepsilon_{\\infty} = \\frac{1}{N} \\sum_{i=1}^{N}
            \\frac{\\max_{x} | \\hat{u}_i(x) - u_i(x) |}
                  {\\mathrm{std}(u_i) + \\epsilon}

    Args:
        pred:   Predicted tensor, shape ``(B, C, *spatial)``.
        target: Ground-truth tensor, same shape.
        eps:    Stability constant.

    Returns:
        Scalar tensor — mean relative max error over the batch.
    """
    dims = tuple(range(1, pred.ndim))
    max_err    = (pred - target).abs().flatten(start_dim=1).max(dim=1).values
    target_std = target.flatten(start_dim=1).std(dim=1).clamp(min=eps)
    return (max_err / target_std).mean()


def h1_error(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative H¹ semi-norm error for 2-D spatial fields.

    The H¹ semi-norm adds a gradient penalty on top of the L2 value
    mismatch.  It is the standard "H1 loss" used in the ``neuraloperator``
    library and is meaningful for smooth PDEs (Poisson, Darcy) whose
    solutions are in the Sobolev space H¹.

    Gradients are estimated with first-order finite differences along each
    spatial axis.  For **1-D** inputs the function falls back to plain
    relative L2 error (only one spatial axis, same as L2 loss for
    1-D problems where the FD gradient information is less informative).

    .. math::

        \\varepsilon_{H^1} = \\frac{1}{N} \\sum_{i=1}^{N}
            \\frac{\\lVert \\hat{u}_i - u_i \\rVert_{H^1}}
                  {\\lVert u_i \\rVert_{H^1} + \\epsilon}

    where

    .. math::

        \\lVert v \\rVert_{H^1}^2
            = \\lVert v \\rVert_2^2
            + \\sum_{d} \\lVert \\partial_d v \\rVert_2^2.

    Args:
        pred:   Predicted tensor, shape ``(B, C, *spatial)``.  Must be at
                least 2-D in space for the gradient term to be computed.
        target: Ground-truth tensor, same shape.
        eps:    Stability constant.

    Returns:
        Scalar tensor — mean relative H¹ error over the batch.
    """
    if pred.ndim < 4:
        # 1-D spatial: fall back to relative L2
        return relative_l2_error(pred, target, eps=eps)

    def _h1_norm_sq(x: Tensor) -> Tensor:
        """H¹ semi-norm² per sample: shape (B,)."""
        val_sq = x.pow(2).flatten(start_dim=1).sum(dim=1)
        grad_sq = torch.zeros_like(val_sq)
        # Finite differences along height (dim -2) and width (dim -1)
        grad_sq = grad_sq + (x[..., 1:, :] - x[..., :-1, :]).pow(2).flatten(start_dim=1).sum(dim=1)
        grad_sq = grad_sq + (x[..., :, 1:] - x[..., :, :-1]).pow(2).flatten(start_dim=1).sum(dim=1)
        return val_sq + grad_sq

    diff_h1   = _h1_norm_sq(pred - target).sqrt()
    target_h1 = _h1_norm_sq(target).sqrt().clamp(min=eps)
    return (diff_h1 / target_h1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Flow-model-specific: ensemble metrics
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_relative_l2(
    preds: List[Tensor],
    target: Tensor,
    eps: float = 1e-8,
) -> Dict[str, Tensor]:
    """Evaluate a flow model's ensemble of predictions.

    Flow matching models are *generative* — they produce a distribution
    p(u | f) rather than a single prediction.  By sampling multiple times
    from the same condition we can decompose performance into:

    * **Mean error**: relative L2 of the ensemble mean vs. ground truth.
      Reflects the model's average prediction quality.
    * **Sample spread (std)**: average standard deviation of the ensemble,
      normalised by the ground-truth norm.  Reflects how much the model
      explores the solution space.

    Args:
        preds:  List of ``K`` prediction tensors, each shape
                ``(B, C, *spatial)``.  All tensors must share the same shape.
        target: Ground-truth tensor, shape ``(B, C, *spatial)``.
        eps:    Stability constant.

    Returns:
        Dict with keys:

        * ``"mean_rel_l2"``   — relative L2 of the ensemble mean.
        * ``"sample_spread"`` — normalised ensemble standard deviation.
        * ``"best_rel_l2"``   — lowest per-sample relative L2 across ensemble
          members (oracle bound; useful for theoretical analysis).
    """
    if len(preds) == 0:
        raise ValueError("preds must be a non-empty list of tensors.")

    stacked = torch.stack(preds, dim=0)           # (K, B, C, *spatial)
    mean_pred = stacked.mean(dim=0)               # (B, C, *spatial)

    mean_err = relative_l2_error(mean_pred, target, eps=eps)

    # Normalised sample spread: std across ensemble members, averaged per sample
    target_norm = _norm(target, p=2.0).clamp(min=eps)          # (B,)
    spread_per_sample = stacked.std(dim=0)                       # (B, C, *spatial)
    spread_norm = _norm(spread_per_sample, p=2.0) / target_norm  # (B,)
    sample_spread = spread_norm.mean()

    # Best sample (oracle) — lower bound on single-sample performance
    per_sample_errors = torch.stack(
        [relative_l2_error_batch(p, target, eps=eps) for p in preds],
        dim=0,
    )  # (K, B)
    best_rel_l2 = per_sample_errors.min(dim=0).values.mean()

    return {
        "mean_rel_l2":   mean_err,
        "sample_spread": sample_spread,
        "best_rel_l2":   best_rel_l2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience class
# ─────────────────────────────────────────────────────────────────────────────

class EvalMetrics:
    """Compute a standard suite of PDE evaluation metrics in one call.

    By default computes relative L2, H1, MAE, and relative max error.
    You can restrict which metrics are computed by passing a list of names.

    Available metric names: ``"rel_l2"``, ``"h1"``, ``"mse"``, ``"mae"``,
    ``"rel_max"``.

    Example::

        em = EvalMetrics()
        results = em(pred, target)
        print(results)
        # {'rel_l2': tensor(0.0312), 'h1': tensor(0.0421), ...}

        em_fast = EvalMetrics(metrics=["rel_l2"])
        results = em_fast(pred, target)
    """

    _ALL = ("rel_l2", "h1", "mse", "mae", "rel_max")

    def __init__(self, metrics: Optional[List[str]] = None):
        if metrics is None:
            self.metrics = list(self._ALL)
        else:
            unknown = set(metrics) - set(self._ALL)
            if unknown:
                raise ValueError(
                    f"Unknown metrics: {unknown}.  Choose from {self._ALL}."
                )
            self.metrics = list(metrics)

    def __call__(self, pred: Tensor, target: Tensor) -> Dict[str, float]:
        """Compute all configured metrics.

        Args:
            pred:   Predicted tensor, shape ``(B, C, *spatial)``.
            target: Ground-truth tensor, same shape.

        Returns:
            Dict mapping metric name → Python float.
        """
        results: Dict[str, float] = {}
        fns = {
            "rel_l2":  relative_l2_error,
            "h1":      h1_error,
            "mse":     mse,
            "mae":     mae,
            "rel_max": relative_max_error,
        }
        for name in self.metrics:
            results[name] = fns[name](pred, target).item()
        return results

    def __repr__(self) -> str:
        return f"EvalMetrics(metrics={self.metrics})"
