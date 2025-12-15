"""
Utility functions and classes for FlowPDE.

This module provides:
- Activation functions (Swish/SiLU)
- Visualization utilities (plot_curve, visualize_flow_evolution)
- Configuration management (load_config, get_args, override_config)
- Model utilities (save_model, load_class, find_latest_checkpoint)
- Inference config creation (create_inference_config)
"""

# Activation functions
from .activation_functions import Swish

# Visualization utilities
from .flow_viz import (
    plot_curve,
    visualize_flow_evolution
)

# Configuration and arguments
from .args_utils import (
    get_args,
    override_config
)

# General utilities
from .utils import (
    save_model,
    print_stats,
    load_config,
    load_class,
    find_latest_checkpoint
)

# Inference config creation
from .generate_inference_config import (
    create_inference_config,
    save_yaml
)

__all__ = [
    # Activation functions
    'Swish',
    
    # Visualization
    'plot_curve',
    'visualize_flow_evolution',
    
    # Configuration
    'get_args',
    'override_config',
    'load_config',
    
    # Model utilities
    'save_model',
    'print_stats',
    'load_class',
    'find_latest_checkpoint',
    
    # Inference config
    'create_inference_config',
    'save_yaml',
]
