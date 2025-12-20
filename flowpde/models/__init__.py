"""
Model architectures for FlowPDE
"""

from .mlp import MLP
from .unet import UNet
from .cnn import CNN, CNN1D, CNN2D

__all__ = ['MLP', 'UNet', 'CNN', 'CNN1D', 'CNN2D']
