"""
U-Net architecture for FlowPDE.

Encoder-decoder architecture with skip connections for high-resolution
PDE solving where multi-scale feature extraction is beneficial.
"""

import math
from typing import List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .components import (
    FourierTimeEmbedding,
    TimeMLPEmbedding,
    get_conv_layer,
    get_conv_transpose_layer,
    get_pool_layer,
    get_norm_layer,
    get_activation,
    get_num_groups,
    expand_time_embedding,
    init_weights,
)
from flowpde.core.base_conditioner import BaseConditioner, ConcatConditioner, FiLMConditioner, NullConditioner
from flowpde.models.convnet import _input_channels_for_conditioner


class ConvBlock(nn.Module):
    """
    Double convolution block used in UNet.
    
    Structure: Conv → Norm → Act → Conv → Norm → Act
    
    Args:
        spatial_dim: Spatial dimensionality (1 or 2)
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        norm_type: Normalization type
        activation: Activation function name
    """
    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_type: str = "group",
        activation: str = "swish",
    ):
        super().__init__()
        
        Conv = get_conv_layer(spatial_dim)
        padding = kernel_size // 2
        
        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, padding=padding),
            get_norm_layer(norm_type, out_channels, spatial_dim),
            get_activation(activation),
            Conv(out_channels, out_channels, kernel_size, padding=padding),
            get_norm_layer(norm_type, out_channels, spatial_dim),
            get_activation(activation),
        )
        
        # Store out_channels for external access
        self.out_channels = out_channels
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing global spatial dependencies.
    
    Used at the UNet bottleneck to allow long-range interactions,
    important for PDEs with non-local behavior.
    
    Args:
        channels: Number of input channels
        num_heads: Number of attention heads
        norm_type: Normalization type
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        norm_type: str = "group",
    ):
        super().__init__()
        
        # Ensure channels is divisible by num_heads
        self.num_heads = min(num_heads, channels)
        while channels % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        
        self.norm = get_norm_layer(norm_type, channels, spatial_dim=2)
        self.attn = nn.MultiheadAttention(
            channels,
            num_heads=self.num_heads,
            batch_first=True
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W) for 2D or (B, C, L) for 1D
        
        Returns:
            Output with residual connection
        """
        if x.dim() == 4:  # 2D: (B, C, H, W)
            B, C, H, W = x.shape
            h = self.norm(x)
            h = h.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
            h = self.attn(h, h, h, need_weights=False)[0]
            h = h.transpose(1, 2).view(B, C, H, W)
        else:  # 1D: (B, C, L)
            B, C, L = x.shape
            h = self.norm(x)
            h = h.transpose(1, 2)  # (B, L, C)
            h = self.attn(h, h, h, need_weights=False)[0]
            h = h.transpose(1, 2)  # (B, C, L)
        
        return x + h


class UNet(nn.Module):
    """
    U-Net architecture for flow matching on spatial PDE domains.
    
    Encoder-decoder architecture with skip connections that enables
    multi-scale feature extraction. Particularly effective for problems
    where both local and global patterns matter.
    
    Architecture:
        Encoder: [ConvBlock → Pool] × depth
        Bottleneck: ConvBlock [+ Attention]
        Decoder: [Upsample → Concat(skip) → ConvBlock] × depth
    
    Features:
        - Automatic depth calculation based on spatial resolution
        - Fourier time embeddings with residual conditioning
        - Optional self-attention at bottleneck
        - Dimension-agnostic (1D/2D)
    
    Args:
        spatial_dim: Spatial dimensionality (1 or 2)
        spatial_size: Size of spatial domain (assumes square for 2D)
        base_channels: Base number of channels (doubled at each level)
        solution_channels: Number of channels in solution (default: 1)
        condition_channels: Number of channels in condition (default: 1)
        max_channels: Maximum channels at any level (default: 512)
        use_attention: Whether to use attention at bottleneck (default: True)
        norm_type: Normalization type ('group', 'batch', 'instance')
        activation: Activation function name
        return_spatial: If True, return spatial tensor; if False, flatten
    """
    def __init__(
        self,
        spatial_dim: int,
        spatial_size: int,
        base_channels: int = 64,
        solution_channels: int = 1,
        condition_channels: int = 1,
        max_channels: int = 512,
        use_attention: bool = True,
        norm_type: str = "group",
        activation: str = "silu",
        return_spatial: bool = False,
        conditioner: Optional[BaseConditioner] = None,
    ):
        super().__init__()

        if spatial_dim not in [1, 2]:
            raise ValueError(f"spatial_dim must be 1 or 2, got {spatial_dim}")

        self.spatial_dim = spatial_dim
        self.spatial_size = spatial_size
        self.base_channels = base_channels
        self.solution_channels = solution_channels
        self.condition_channels = condition_channels
        self.use_attention = use_attention
        self.return_spatial = return_spatial
        self.conditioner = conditioner if conditioner is not None else ConcatConditioner(dim=1)

        # Calculate depth based on spatial resolution
        # We want to downsample until spatial size is ~4-8
        max_depth = int(math.floor(math.log2(spatial_size))) - 2
        self.depth = max(1, min(max_depth, 5))  # Cap at 5 levels

        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_embed = TimeMLPEmbedding(dim=time_emb_dim, activation=activation)

        # Build encoder
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.time_projs_encoder = nn.ModuleList()

        in_ch = _input_channels_for_conditioner(
            self.conditioner, solution_channels, condition_channels
        )
        Pool = get_pool_layer(spatial_dim, "max")
        
        encoder_channels = []
        for i in range(self.depth):
            out_ch = min(base_channels * (2 ** i), max_channels)
            self.encoder_blocks.append(
                ConvBlock(spatial_dim, in_ch, out_ch, norm_type=norm_type, activation=activation)
            )
            self.pools.append(Pool(2))
            self.time_projs_encoder.append(nn.Linear(time_emb_dim, out_ch))
            encoder_channels.append(out_ch)
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = min(in_ch * 2, max_channels)
        self.bottleneck = ConvBlock(
            spatial_dim, in_ch, bottleneck_ch, norm_type=norm_type, activation=activation
        )
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, bottleneck_ch)
        
        if use_attention and spatial_dim == 2:
            self.bottleneck_attn = AttentionBlock(bottleneck_ch, norm_type=norm_type)
        else:
            self.bottleneck_attn = None
        
        # Build decoder
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.time_projs_decoder = nn.ModuleList()
        
        ConvT = get_conv_transpose_layer(spatial_dim)
        in_ch = bottleneck_ch
        
        for i, skip_ch in enumerate(reversed(encoder_channels)):
            self.upsamples.append(ConvT(in_ch, skip_ch, kernel_size=2, stride=2))
            self.decoder_blocks.append(
                ConvBlock(spatial_dim, skip_ch * 2, skip_ch, norm_type=norm_type, activation=activation)
            )
            self.time_projs_decoder.append(nn.Linear(time_emb_dim, skip_ch))
            in_ch = skip_ch
        
        # Output projection
        Conv = get_conv_layer(spatial_dim)
        self.output_conv = Conv(base_channels, solution_channels, 1)
        
        # Initialize weights
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
            Velocity field v(x, f, t)
        """
        # Reshape inputs
        x = self._reshape_input(x, self.solution_channels)
        f = self._reshape_input(f, self.condition_channels)
        
        B = x.shape[0]
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Apply conditioner (concat by default)
        h = self.conditioner(x, f)

        # Encoder path
        skips = []
        for i, (block, pool, t_proj) in enumerate(
            zip(self.encoder_blocks, self.pools, self.time_projs_encoder)
        ):
            h = block(h)
            h = h + expand_time_embedding(t_proj(t_emb), self.spatial_dim)
            skips.append(h)
            # Only pool if spatial size > 1
            if h.shape[-1] > 1:
                h = pool(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        h = h + expand_time_embedding(self.time_proj_bottleneck(t_emb), self.spatial_dim)
        
        if self.bottleneck_attn is not None:
            h = self.bottleneck_attn(h)
        
        # Decoder path
        for i, (up, block, t_proj, skip) in enumerate(
            zip(self.upsamples, self.decoder_blocks, self.time_projs_decoder, reversed(skips))
        ):
            h = up(h)
            
            # Handle size mismatch from non-power-of-2 dimensions
            if h.shape[-1] != skip.shape[-1] or (self.spatial_dim == 2 and h.shape[-2] != skip.shape[-2]):
                h = F.interpolate(h, size=skip.shape[-self.spatial_dim:], mode='nearest')
            
            h = torch.cat([h, skip], dim=1)
            h = block(h)
            h = h + expand_time_embedding(t_proj(t_emb), self.spatial_dim)
        
        # Output
        out = self.output_conv(h)
        
        # Return format
        if self.return_spatial:
            return out
        else:
            return out.view(B, -1)
    
    def extra_repr(self) -> str:
        return (
            f"spatial_dim={self.spatial_dim}, spatial_size={self.spatial_size}, "
            f"depth={self.depth}, base_channels={self.base_channels}, "
            f"solution_channels={self.solution_channels}, "
            f"condition_channels={self.condition_channels}"
        )
