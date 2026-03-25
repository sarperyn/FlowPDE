"""
Unified Convolutional Neural Network for FlowPDE.

Dimension-agnostic CNN that handles both 1D and 2D spatial data through
explicit configuration rather than heuristic inference.
"""

from typing import Optional, Union, Tuple

import torch
from torch import nn, Tensor

from .components import (
    FourierTimeEmbedding,
    TimeMLPEmbedding,
    DimensionalConv,
    get_conv_layer,
    get_norm_layer,
    get_activation,
    get_num_groups,
    expand_time_embedding,
    init_weights,
)
from flowpde.core.base_conditioner import BaseConditioner, ConcatConditioner, FiLMConditioner, NullConditioner


def _input_channels_for_conditioner(
    conditioner: BaseConditioner,
    solution_channels: int,
    condition_channels: int,
) -> int:
    """Return the number of channels the first layer receives after conditioning."""
    if isinstance(conditioner, ConcatConditioner):
        return solution_channels + condition_channels
    elif isinstance(conditioner, NullConditioner):
        return solution_channels
    elif isinstance(conditioner, FiLMConditioner):
        raise ValueError(
            "FiLMConditioner modulates existing feature maps and cannot be used "
            "as an input-level conditioner. Apply FiLM inside residual blocks instead."
        )
    else:
        # Unknown conditioner — assume concat so existing models never break.
        return solution_channels + condition_channels


class ResidualBlock(nn.Module):
    """
    Dimension-agnostic residual block with time conditioning.
    
    Pre-activation ResNet block: Norm → Act → Conv → Norm → Act → Conv
    Time embedding is added after the first convolution.
    
    Args:
        spatial_dim: Spatial dimensionality (1 or 2)
        channels: Number of input/output channels
        kernel_size: Convolution kernel size
        time_emb_dim: Dimension of time embedding
        norm_type: Normalization type ('group', 'batch', 'instance', 'none')
        activation: Activation function name
        dropout: Dropout probability (0 = no dropout)
    """
    def __init__(
        self,
        spatial_dim: int,
        channels: int,
        kernel_size: int = 3,
        time_emb_dim: int = 64,
        norm_type: str = "group",
        activation: str = "swish",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        padding = kernel_size // 2
        
        Conv = get_conv_layer(spatial_dim)
        
        # Pre-activation block 1
        self.norm1 = get_norm_layer(norm_type, channels, spatial_dim)
        self.act1 = get_activation(activation)
        self.conv1 = Conv(channels, channels, kernel_size, padding=padding)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, channels)
        
        # Pre-activation block 2
        self.norm2 = get_norm_layer(norm_type, channels, spatial_dim)
        self.act2 = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = Conv(channels, channels, kernel_size, padding=padding)
    
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, C, L) for 1D or (B, C, H, W) for 2D
            t_emb: Time embedding (B, time_emb_dim)
        
        Returns:
            Output with residual connection, same shape as input
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add time embedding (broadcast across spatial dims)
        h = h + expand_time_embedding(self.time_proj(t_emb), self.spatial_dim)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h


class ConvNet(nn.Module):
    """
    Unified Convolutional Neural Network for flow matching on PDE data.
    
    A dimension-agnostic CNN that uses residual blocks with time conditioning.
    Suitable for both 1D problems (e.g., Burgers equation) and 2D problems
    (e.g., Poisson equation).
    
    Architecture:
        Input Conv → [ResidualBlock × num_blocks] → Output Norm → Output Conv
    
    Args:
        spatial_dim: Spatial dimensionality (1 or 2)
        spatial_size: Size of spatial domain (int for 1D, int for square 2D)
        hidden_channels: Number of hidden channels
        num_blocks: Number of residual blocks
        solution_channels: Number of channels in solution (default: 1)
        condition_channels: Number of channels in condition (default: 1)
        kernel_size: Convolution kernel size (default: 3)
        norm_type: Normalization type ('group', 'batch', 'instance', 'none')
        activation: Activation function name ('swish', 'relu', 'gelu')
        dropout: Dropout probability (default: 0.0)
        return_spatial: If True, return spatial tensor; if False, flatten (default: False)
    """
    def __init__(
        self,
        spatial_dim: int,
        spatial_size: int,
        hidden_channels: int,
        num_blocks: int,
        solution_channels: int = 1,
        condition_channels: int = 1,
        kernel_size: int = 3,
        norm_type: str = "group",
        activation: str = "swish",
        dropout: float = 0.0,
        return_spatial: bool = False,
        conditioner: Optional[BaseConditioner] = None,
    ):
        super().__init__()

        if spatial_dim not in [1, 2]:
            raise ValueError(f"spatial_dim must be 1 or 2, got {spatial_dim}")

        self.spatial_dim = spatial_dim
        self.spatial_size = spatial_size
        self.hidden_channels = hidden_channels
        self.solution_channels = solution_channels
        self.condition_channels = condition_channels
        self.return_spatial = return_spatial
        self.conditioner = conditioner if conditioner is not None else ConcatConditioner(dim=1)

        # Time embedding
        time_emb_dim = hidden_channels
        self.time_embed = TimeMLPEmbedding(dim=time_emb_dim, activation=activation)

        # Input projection: channels depend on conditioner type
        in_channels = _input_channels_for_conditioner(
            self.conditioner, solution_channels, condition_channels
        )
        Conv = get_conv_layer(spatial_dim)
        self.input_conv = Conv(in_channels, hidden_channels, 3, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(
                spatial_dim=spatial_dim,
                channels=hidden_channels,
                kernel_size=kernel_size,
                time_emb_dim=time_emb_dim,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_norm = get_norm_layer(norm_type, hidden_channels, spatial_dim)
        self.output_act = get_activation(activation)
        self.output_conv = Conv(hidden_channels, solution_channels, 3, padding=1)
        
        # Initialize weights (zero-init final layer)
        init_weights(self, zero_init_last=True)
    
    def _reshape_input(self, tensor: Tensor, channels: int) -> Tensor:
        """Reshape flattened input to spatial format."""
        if tensor.dim() == 2:
            B = tensor.shape[0]
            if self.spatial_dim == 1:
                return tensor.view(B, channels, self.spatial_size)
            else:
                return tensor.view(B, channels, self.spatial_size, self.spatial_size)
        return tensor
    
    def forward(self, x: Tensor, f: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field for flow matching.
        
        Args:
            x: State x_t, shape (B, C, *spatial) or flattened (B, C*spatial)
            f: Condition, shape (B, C', *spatial) or flattened (B, C'*spatial)
            t: Time t ∈ [0, 1], shape (B,) or (B, 1)
        
        Returns:
            Velocity field v(x, f, t), same shape as x (or flattened if return_spatial=False)
        """
        # Track if input was flattened
        was_flattened = x.dim() == 2
        
        # Reshape inputs to spatial format
        x = self._reshape_input(x, self.solution_channels)
        f = self._reshape_input(f, self.condition_channels)
        
        B = x.shape[0]
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Apply conditioner (concat by default)
        h = self.conditioner(x, f)
        h = self.input_conv(h)
        
        # Apply residual blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output projection
        h = self.output_norm(h)
        h = self.output_act(h)
        out = self.output_conv(h)
        
        # Return format
        if self.return_spatial:
            return out
        else:
            return out.view(B, -1)
    
    def extra_repr(self) -> str:
        return (
            f"spatial_dim={self.spatial_dim}, spatial_size={self.spatial_size}, "
            f"hidden_channels={self.hidden_channels}, "
            f"solution_channels={self.solution_channels}, "
            f"condition_channels={self.condition_channels}"
        )
