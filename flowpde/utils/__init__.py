"""
Utility functions and classes for FlowPDE.

This module provides:
- Metrics (relative_l2_error, h1_error, mse, mae, EvalMetrics, ensemble_relative_l2)
- Uncertainty quantification metrics (UQMetrics, crps_ensemble, energy_score, ...)
- General utilities (save_model, print_stats, plot_curve)
"""

# Metrics (primary entry point for evaluation)
from .metrics import (
    relative_l2_error,
    relative_l2_error_batch,
    h1_error,
    mse,
    mae,
    relative_max_error,
    ensemble_relative_l2,
    EvalMetrics,
)

# General utilities
from .utils import (
    save_model,
    print_stats,
    plot_curve,
)

# Uncertainty quantification metrics
from .uq_metrics import (
    UQMetrics,
    credible_interval_coverage,
    crps_ensemble,
    energy_score,
    error_spread_correlation,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
    variance_decomposition,
)

__all__ = [
    # Metrics
    "relative_l2_error",
    "relative_l2_error_batch",
    "h1_error",
    "mse",
    "mae",
    "relative_max_error",
    "ensemble_relative_l2",
    "EvalMetrics",

    # General utilities
    "save_model",
    "print_stats",
    "plot_curve",

    # Uncertainty quantification
    "UQMetrics",
    "credible_interval_coverage",
    "crps_ensemble",
    "energy_score",
    "error_spread_correlation",
    "rank_histogram",
    "reliability_curve",
    "spread_skill_ratio",
    "variance_decomposition",
]
