"""
Utility functions and classes for FlowPDE.

This module provides:
- Metrics (relative_l2_error, h1_error, mse, mae, EvalMetrics, ensemble_relative_l2)
- Visualization utilities (plot_curve, visualize_flow_evolution)
- Configuration management (load_config, get_args, override_config)
- Model utilities (save_model, load_class, find_latest_checkpoint)
- Inference config creation (create_inference_config)

Note: ``Swish`` has been removed — use ``torch.nn.SiLU()`` directly.
"""

# Metrics (primary entry point for evaluation)
from .metrics import (
    relative_l2_error,
    relative_l2_error_batch,
    h1_error,
    mse,
    mae,
    max_pointwise_error,
    relative_max_error,
    ensemble_relative_l2,
    EvalMetrics,
)

# Visualization utilities
from .flow_viz import (
    plot_curve,
    visualize_flow_evolution,
)

# Configuration and arguments
from .args_utils import (
    get_args,
    override_config,
)

# General utilities
from .utils import (
    save_model,
    print_stats,
    load_config,
    load_class,
    find_latest_checkpoint,
)

# Inference config creation
from .generate_inference_config import (
    create_inference_config,
    save_yaml,
)

__all__ = [
    # Metrics
    "relative_l2_error",
    "relative_l2_error_batch",
    "h1_error",
    "mse",
    "mae",
    "max_pointwise_error",
    "relative_max_error",
    "ensemble_relative_l2",
    "EvalMetrics",

    # Visualization
    "plot_curve",
    "visualize_flow_evolution",

    # Configuration
    "get_args",
    "override_config",
    "load_config",

    # Model utilities
    "save_model",
    "print_stats",
    "load_class",
    "find_latest_checkpoint",

    # Inference config
    "create_inference_config",
    "save_yaml",
]
