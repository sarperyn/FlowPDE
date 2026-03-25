"""
Multi-Layer Perceptron for FlowPDE.

Fully-connected network for low-dimensional PDE problems where
convolutional structure is not beneficial.
"""

from typing import Optional

import torch
from torch import nn, Tensor

from .components import (
    FourierTimeEmbedding,
    get_activation,
    init_weights,
)
from flowpde.core.base_conditioner import BaseConditioner, ConcatConditioner, NullConditioner


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for flow matching.
    
    A fully-connected neural network that predicts velocity fields for
    continuous normalizing flows. Suitable for low-dimensional problems
    or when spatial structure is not important.
    
    Architecture:
        Input Proj → [Residual Block × num_layers] → Output Proj
    
    Features:
        - Fourier time embeddings (sinusoidal features)
        - Residual connections in middle layers
        - Flexible hidden dimension and depth
    
    Args:
        input_dim: Dimension of input/output space (x and the velocity output)
        condition_dim: Dimension of the condition f. Defaults to input_dim when
            using ConcatConditioner (the most common case).
        hidden_dim: Number of hidden units in each layer (default: 128)
        num_layers: Number of residual blocks (default: 4)
        activation: Activation function name (default: 'swish')
        dropout: Dropout probability (default: 0.0)
        conditioner: Conditioning mechanism. Defaults to ConcatConditioner,
            which concatenates f to x before the first linear layer.
    """
    def __init__(
        self,
        input_dim: int,
        condition_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "swish",
        dropout: float = 0.0,
        conditioner: Optional[BaseConditioner] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conditioner = conditioner if conditioner is not None else ConcatConditioner(dim=1)

        # Resolve the first linear layer's input size based on conditioner type
        _condition_dim = condition_dim if condition_dim is not None else input_dim
        if isinstance(self.conditioner, NullConditioner):
            proj_in_dim = input_dim
        else:
            # ConcatConditioner or unknown: input_dim + condition_dim
            proj_in_dim = input_dim + _condition_dim

        # Fourier time embedding
        fourier_dim = hidden_dim
        self.time_fourier = FourierTimeEmbedding(dim=fourier_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            get_activation(activation)
        )

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, hidden_dim),
            get_activation(activation)
        )
        
        # Middle layers with residual connections
        self.middle_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                get_activation(activation),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])
        
        # Output projection (zero-initialized for stable training)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        init_weights(self, zero_init_last=True)
    
    def forward(self, x: Tensor, f: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field for flow matching.
        
        Args:
            x: State x_t ∈ R^d, shape (B, d) or (B, *spatial_shape)
            f: Condition, shape (B, d) or (B, *spatial_shape)
            t: Time t ∈ [0, 1], shape (B,) or (B, 1)
        
        Returns:
            Velocity field v(x, f, t) ∈ R^d, same shape as input x
        """
        # Save original shape for output
        original_shape = x.shape
        batch_size = x.shape[0]
        
        # Flatten to (B, input_dim)
        x = x.reshape(batch_size, -1)
        f = f.reshape(batch_size, -1)
        
        # Ensure consistent input dimension
        assert x.shape[1] == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {x.shape[1]}"
        
        # Time embedding
        t_emb = self.time_fourier(t)
        t_emb = self.time_proj(t_emb)
        
        # Apply conditioner (concat by default)
        h = self.conditioner(x, f)
        h = self.input_proj(h)
        
        # Add time embedding
        h = h + t_emb
        
        # Apply middle layers with residual connections
        for layer in self.middle_layers:
            h = h + layer(h)
        
        # Output projection
        output = self.output_proj(h)
        
        # Restore original shape
        return output.reshape(original_shape)
    
    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}"