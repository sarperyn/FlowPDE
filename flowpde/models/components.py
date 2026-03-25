"""
Shared neural network components for FlowPDE models.

This module contains reusable building blocks used across all architectures:
- FourierTimeEmbedding: Sinusoidal time encoding
- Normalization factories
- Activation functions
- Dimension-agnostic convolution utilities
"""

import math
from typing import Union, Tuple, Optional, Type
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F




def get_activation(name: str = "silu") -> nn.Module:
    """
    Factory function for activation functions.
    
    Args:
        name: Activation name ('silu'/'swish', 'relu', 'gelu', 'tanh', 'leaky_relu')
    
    Returns:
        nn.Module activation function
    """
    activations = {
        # Swish is commonly used as a synonym for SiLU.
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leaky_relu": partial(nn.LeakyReLU, negative_slope=0.1),
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[key]()


class FourierTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for flow matching and diffusion models.
    
    Maps scalar time $$t \in [0, 1]$$ to high-dimensional feature vector using
    sinusoids at exponentially spaced frequencies. This provides a smooth,
    continuous representation of time that networks can easily learn from.
    
    Args:
        dim: Embedding dimension (must be even)
        max_period: Maximum period for lowest frequency sinusoid
        learnable: If True, add a learnable linear projection
    """
    def __init__(
        self, 
        dim: int = 128, 
        max_period: float = 10000.0,
        learnable: bool = False
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dim must be even, got {dim}")
        
        self.dim = dim
        self.max_period = max_period
        
        # Precompute frequency bands
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)
        
        # Optional learnable projection
        self.proj = nn.Linear(dim, dim) if learnable else nn.Identity()
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Time tensor of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            Embedding of shape (batch_size, dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # t: (B, 1), freqs: (dim//2,)
        args = t * self.freqs  # (B, dim//2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj(embedding)


class TimeMLPEmbedding(nn.Module):
    """
    MLP-based time embedding that first applies Fourier features.
    
    Combines FourierTimeEmbedding with a small MLP for richer representations.
    
    Args:
        dim: Output embedding dimension
        hidden_mult: Hidden layer multiplier (default: 4)
        activation: Activation function name
    """
    def __init__(
        self,
        dim: int = 128,
        hidden_mult: int = 4,
        activation: str = "silu"
    ):
        super().__init__()
        self.fourier = FourierTimeEmbedding(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            get_activation(activation),
            nn.Linear(dim * hidden_mult, dim)
        )
    
    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.fourier(t))



def get_num_groups(channels: int, preferred: int = 32) -> int:
    """
    Compute valid number of groups for GroupNorm.
    
    Tries preferred number first, then falls back to largest divisor.
    
    Args:
        channels: Number of channels
        preferred: Preferred number of groups
    
    Returns:
        Valid number of groups that divides channels
    """
    for num_groups in [preferred, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0:
            return num_groups
    return 1


def get_norm_layer(
    norm_type: str,
    num_features: int,
    spatial_dim: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function for normalization layers.
    
    Args:
        norm_type: Type of normalization ('group', 'batch', 'instance', 'layer', 'none')
        num_features: Number of features/channels
        spatial_dim: Spatial dimensionality (1 or 2) for batch/instance norm
        **kwargs: Additional arguments passed to norm layer
    
    Returns:
        Normalization layer
    """
    norm_type = norm_type.lower()
    
    if norm_type == "group":
        num_groups = kwargs.pop("num_groups", get_num_groups(num_features))
        return nn.GroupNorm(num_groups, num_features, **kwargs)
    
    elif norm_type == "batch":
        if spatial_dim == 1:
            return nn.BatchNorm1d(num_features, **kwargs)
        else:
            return nn.BatchNorm2d(num_features, **kwargs)
    
    elif norm_type == "instance":
        if spatial_dim == 1:
            return nn.InstanceNorm1d(num_features, **kwargs)
        else:
            return nn.InstanceNorm2d(num_features, **kwargs)
    
    elif norm_type == "layer":
        return nn.LayerNorm(num_features, **kwargs)
    
    elif norm_type == "none":
        return nn.Identity()
    
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# =============================================================================
# Dimension-Agnostic Convolution
# =============================================================================

def get_conv_layer(spatial_dim: int) -> Type[nn.Module]:
    """Get Conv1d or Conv2d class based on spatial dimension."""
    if spatial_dim == 1:
        return nn.Conv1d
    elif spatial_dim == 2:
        return nn.Conv2d
    else:
        raise ValueError(f"spatial_dim must be 1 or 2, got {spatial_dim}")


def get_conv_transpose_layer(spatial_dim: int) -> Type[nn.Module]:
    """Get ConvTranspose1d or ConvTranspose2d class based on spatial dimension."""
    if spatial_dim == 1:
        return nn.ConvTranspose1d
    elif spatial_dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError(f"spatial_dim must be 1 or 2, got {spatial_dim}")


def get_pool_layer(spatial_dim: int, pool_type: str = "max") -> Type[nn.Module]:
    """Get pooling layer class based on spatial dimension."""
    if pool_type == "max":
        return nn.MaxPool1d if spatial_dim == 1 else nn.MaxPool2d
    elif pool_type == "avg":
        return nn.AvgPool1d if spatial_dim == 1 else nn.AvgPool2d
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")


class DimensionalConv(nn.Module):
    """
    Dimension-agnostic convolution layer.
    
    Automatically uses Conv1d or Conv2d based on spatial_dim parameter.
    
    Args:
        spatial_dim: Spatial dimensionality (1 or 2)
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding (default: kernel_size // 2 for 'same' padding)
        groups: Number of groups for grouped convolution
        bias: Whether to include bias
    """
    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        
        if padding is None:
            padding = kernel_size // 2
        
        Conv = get_conv_layer(spatial_dim)
        self.conv = Conv(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


# =============================================================================
# Weight Initialization
# =============================================================================

def init_weights(module: nn.Module, zero_init_last: bool = True):
    """
    Initialize weights using Kaiming initialization with zero-init for final layer.
    
    Args:
        module: Module to initialize
        zero_init_last: If True, zero-initialize layers named 'output_*' or 'final_*'
    """
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            # Check if this is a final/output layer
            is_output = any(tag in name for tag in ['output', 'final', 'out_conv', 'proj_out'])
            
            if zero_init_last and is_output:
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.Linear):
            is_output = any(tag in name for tag in ['output', 'final', 'out_proj', 'proj_out'])
            
            if zero_init_last and is_output:
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# =============================================================================
# Utility Functions
# =============================================================================

def expand_time_embedding(t_emb: Tensor, spatial_dim: int) -> Tensor:
    """
    Expand time embedding for broadcasting with spatial tensors.
    
    Args:
        t_emb: Time embedding of shape (B, C)
        spatial_dim: Number of spatial dimensions (1 or 2)
    
    Returns:
        Expanded tensor ready for broadcasting: (B, C, 1) or (B, C, 1, 1)
    """
    if spatial_dim == 1:
        return t_emb[:, :, None]
    elif spatial_dim == 2:
        return t_emb[:, :, None, None]
    else:
        raise ValueError(f"spatial_dim must be 1 or 2, got {spatial_dim}")


def compute_spatial_size(
    input_dim: int,
    channels: int,
    spatial_dim: int
) -> Union[int, Tuple[int, int]]:
    """
    Compute spatial size from flattened dimension.
    
    Args:
        input_dim: Total flattened dimension
        channels: Number of channels
        spatial_dim: Spatial dimensionality (1 or 2)
    
    Returns:
        Spatial size (int for 1D, tuple for 2D if non-square)
    """
    spatial_elements = input_dim // channels
    
    if spatial_dim == 1:
        return spatial_elements
    else:
        side = int(math.sqrt(spatial_elements))
        if side * side != spatial_elements:
            raise ValueError(
                f"Cannot reshape {spatial_elements} elements into square grid. "
                f"Got input_dim={input_dim}, channels={channels}"
            )
        return side
