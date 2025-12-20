"""
Simple CNN architectures for 1D and 2D.
"""

import math
import torch
from torch import nn, Tensor
from typing import Optional

from flowpde.utils.activation_functions import Swish


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for temporal conditioning.
    
    Maps scalar time $t$ to high-dimensional feature vector using sinusoids.
    
    Args:
        dim: Embedding dimension (should be even)
        max_period: Maximum period for sinusoids
    """
    def __init__(self, dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class ResidualBlock1D(nn.Module):
    """Residual convolutional block for 1D data.
    
    Args:
        channels: Number of input/output channels
        kernel_size: Convolution kernel size
        time_emb_dim: Dimension of time embedding (for conditioning)
    """
    def __init__(self, channels: int, kernel_size: int = 3, time_emb_dim: int = 64):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = Swish()
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, channels)
    
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, C, L)
            t_emb: Time embedding (B, time_emb_dim)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_proj(t_emb)[:, :, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return x + h


class ResidualBlock2D(nn.Module):
    """Residual convolutional block for 2D data.
    
    Args:
        channels: Number of input/output channels
        kernel_size: Convolution kernel size
        time_emb_dim: Dimension of time embedding (for conditioning)
    """
    def __init__(self, channels: int, kernel_size: int = 3, time_emb_dim: int = 64):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = Swish()
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, channels)
    
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
            t_emb: Time embedding (B, time_emb_dim)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_proj(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return x + h


class CNN1D(nn.Module):
    """Simple 1D CNN for flow matching on sequential/1D spatial data.
    
    A straightforward convolutional architecture for 1D problems like Burgers equation.
    Uses residual blocks with time conditioning.
    
    Args:
        input_dim: Length of 1D spatial domain
        hidden_channels: Number of hidden channels
        num_blocks: Number of residual blocks
        condition_channels: Number of channels in condition
        kernel_size: Convolution kernel size (default: 3)
        solution_channels: Number of channels in solution (default: 1)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        num_blocks: int,
        condition_channels: int,
        kernel_size: int = 3,
        solution_channels: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.solution_channels = solution_channels
        self.condition_channels = condition_channels
        
        # Time embedding
        time_emb_dim = hidden_channels
        self.time_fourier = FourierTimeEmbedding(dim=time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            Swish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Input projection: x + condition concatenated
        in_channels = solution_channels + condition_channels
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, kernel_size, time_emb_dim)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.output_conv = nn.Conv1d(hidden_channels, solution_channels, 3, padding=1)
    
    def forward(self, x: Tensor, f: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: State $x_t$ (B, solution_channels, L) or flattened (B, L)
            f: Condition (B, condition_channels, L) or flattened (B, L)
            t: Time $t \in [0,1]$ (B, 1)
        
        Returns:
            $v_\\theta(x_t, f, t)$: Velocity field, same shape as x
        """
        # Handle flattened input
        if x.dim() == 2:
            B = x.shape[0]
            L = x.shape[1] // self.solution_channels
            x = x.view(B, self.solution_channels, L)
        
        if f.dim() == 2:
            B = f.shape[0]
            L = f.shape[1] // self.condition_channels
            f = f.view(B, self.condition_channels, L)
        
        original_shape = x.shape
        
        # Time embedding
        t_emb = self.time_fourier(t)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate x and condition
        h = torch.cat([x, f], dim=1)
        h = self.input_conv(h)
        
        # Apply residual blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output projection
        h = self.output_norm(h)
        h = Swish()(h)
        out = self.output_conv(h)
        
        # Return flattened if input was flattened
        return out.view(original_shape[0], -1)


class CNN2D(nn.Module):
    """Simple 2D CNN for flow matching on 2D spatial data.
    
    A straightforward convolutional architecture for 2D problems like Poisson equation.
    Uses residual blocks with time conditioning.
    
    Args:
        input_dim: Spatial resolution (assumes square grid) or flattened dimension
        hidden_channels: Number of hidden channels
        num_blocks: Number of residual blocks
        condition_channels: Number of channels in condition
        kernel_size: Convolution kernel size (default: 3)
        solution_channels: Number of channels in solution (default: 1)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        num_blocks: int,
        condition_channels: int,
        kernel_size: int = 3,
        solution_channels: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.solution_channels = solution_channels
        self.condition_channels = condition_channels
        
        # Compute spatial size if input_dim is flattened
        if input_dim > 256:  # Likely flattened
            self.spatial_size = int((input_dim / solution_channels) ** 0.5)
        else:
            self.spatial_size = input_dim
        
        # Time embedding
        time_emb_dim = hidden_channels
        self.time_fourier = FourierTimeEmbedding(dim=time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            Swish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Input projection: x + condition concatenated
        in_channels = solution_channels + condition_channels
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock2D(hidden_channels, kernel_size, time_emb_dim)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.output_conv = nn.Conv2d(hidden_channels, solution_channels, 3, padding=1)
    
    def forward(self, x: Tensor, f: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: State $x_t$ (B, C, H, W) or flattened (B, C*H*W)
            f: Condition (B, C', H, W) or flattened (B, C'*H*W)
            t: Time $t \in [0,1]$ (B, 1)
        
        Returns:
            $v_\\theta(x_t, f, t)$: Velocity field, same shape as x
        """
        B = x.shape[0]
        
        # Handle flattened input
        if x.dim() == 2:
            H = W = self.spatial_size
            x = x.view(B, self.solution_channels, H, W)
        
        if f.dim() == 2:
            H = W = self.spatial_size
            f = f.view(B, self.condition_channels, H, W)
        
        # Time embedding
        t_emb = self.time_fourier(t)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate x and condition
        h = torch.cat([x, f], dim=1)
        h = self.input_conv(h)
        
        # Apply residual blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output projection
        h = self.output_norm(h)
        h = Swish()(h)
        out = self.output_conv(h)
        
        # Return flattened
        return out.view(B, -1)


class CNN(nn.Module):
    """Unified CNN that selects 1D or 2D based on spatial_dim parameter.
    
    A convenience wrapper that uses CNN1D or CNN2D based on the specified
    spatial dimensionality.
    
    Args:
        input_dim: Spatial resolution or flattened dimension
        hidden_channels: Number of hidden channels
        num_blocks: Number of residual blocks
        spatial_dim: Spatial dimensionality (1 or 2)
        condition_channels: Number of channels in condition
        kernel_size: Convolution kernel size (default: 3)
        solution_channels: Number of channels in solution (default: 1)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        num_blocks: int,
        spatial_dim: int,
        condition_channels: int,
        kernel_size: int = 3,
        solution_channels: int = 1,
    ):
        super().__init__()
        
        if spatial_dim not in [1, 2]:
            raise ValueError(f"spatial_dim must be 1 or 2, got {spatial_dim}")
        
        self.spatial_dim = spatial_dim
        
        # Create appropriate CNN
        if spatial_dim == 1:
            self.cnn = CNN1D(
                input_dim=input_dim,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                solution_channels=solution_channels,
                condition_channels=condition_channels
            )
        else:
            self.cnn = CNN2D(
                input_dim=input_dim,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                solution_channels=solution_channels,
                condition_channels=condition_channels
            )
    
    def forward(self, x: Tensor, f: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: State $x_t$
            f: Condition
            t: Time $t \in [0,1]$ (B, 1)
        
        Returns:
            $v_\\theta(x_t, f, t)$: Velocity field
        """
        return self.cnn(x, f, t)
