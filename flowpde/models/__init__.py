"""
Neural network architectures for FlowPDE.

All models follow a consistent interface:
    forward(x, f, t) -> velocity_field

Where:
    x: Current state x_t (flattened or spatial)
    f: Condition (PDE parameters, forcing, etc.)
    t: Flow time in [0, 1]
"""

# Shared components (for advanced users building custom architectures)
from .components import (
    FourierTimeEmbedding,
    TimeMLPEmbedding,
    get_activation,
    get_conv_layer,
    get_norm_layer,
    init_weights,
)

# Main architectures
from .convnet import ConvNet
from .mlp import MLP
from .resnet import ResNet, resnet8, resnet14, resnet18, resnet26
from .unet import UNet

__all__ = [
    # Main architectures
    'MLP',
    'UNet',
    'ConvNet',
    'ResNet',
    # ResNet factory functions
    'resnet8',
    'resnet14',
    'resnet18',
    'resnet26',
    # Shared components
    'FourierTimeEmbedding',
    'TimeMLPEmbedding',
    'get_activation',
    'get_norm_layer',
    'get_conv_layer',
    'init_weights',
]
