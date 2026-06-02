"""
ResNet architecture for FlowPDE.

Residual Network optimized for PDE solving with flow matching.
Unlike classification ResNets, this preserves spatial resolution
(no global pooling) to output full velocity fields.
"""

from typing import Optional, List

import torch
from torch import nn, Tensor

from .components import (
    TimeMLPEmbedding,
    get_conv_layer,
    get_norm_layer,
    get_activation,
    expand_time_embedding,
    init_weights,
)
from flowpde.core.base_conditioner import BaseConditioner, ConcatConditioner, FiLMConditioner, NullConditioner
from flowpde.models.convnet import _input_channels_for_conditioner


class BasicBlock(nn.Module):
    """
    Basic residual block with time conditioning.
    
    Structure: Conv → Norm → Act → Conv → Norm → (+skip) → Act
    
    This is the standard ResNet BasicBlock adapted for:
    1. Dimension-agnostic operation (1D/2D)
    2. Time conditioning via additive embedding
    3. Optional channel expansion for skip connection
    
    Args:
        spatial_dim: Spatial dimensionality (1 or 2)
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        time_emb_dim: Dimension of time embedding
        stride: Convolution stride (default: 1)
        norm_type: Normalization type
        activation: Activation function name
        use_film: Whether to apply FiLM conditioning inside the block
        film_condition_dim: Conditioning vector dimension for FiLM
        film_hidden_dim: Hidden dimension for FiLM parameter generation
    """
    expansion = 1  # BasicBlock doesn't expand channels
    
    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        time_emb_dim: int = 128,
        stride: int = 1,
        norm_type: str = "group",
        activation: str = "swish",
        use_film: bool = False,
        film_condition_dim: Optional[int] = None,
        film_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        padding = kernel_size // 2
        
        Conv = get_conv_layer(spatial_dim)
        
        # Main path
        self.conv1 = Conv(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding, bias=False)
        self.norm1 = get_norm_layer(norm_type, out_channels, spatial_dim)
        self.act1 = get_activation(activation)
        
        self.conv2 = Conv(out_channels, out_channels, kernel_size,
                         stride=1, padding=padding, bias=False)
        self.norm2 = get_norm_layer(norm_type, out_channels, spatial_dim)
        self.act2 = get_activation(activation)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.use_film = use_film
        if self.use_film:
            if film_condition_dim is None:
                raise ValueError("film_condition_dim must be set when use_film is True")
            self.film = FiLMConditioner(
                condition_dim=film_condition_dim,
                feature_dim=out_channels,
                hidden_dim=film_hidden_dim,
            )
        else:
            self.film = None
        
        # Skip connection (identity or projection)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                Conv(in_channels, out_channels, 1, stride=stride, bias=False),
                get_norm_layer(norm_type, out_channels, spatial_dim)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: Tensor, t_emb: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input tensor
            t_emb: Time embedding (B, time_emb_dim)
            condition: Optional conditioning tensor for FiLM
        """
        if self.use_film and condition is None:
            raise ValueError("condition must be provided when use_film is True")

        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        # Add time conditioning
        out = out + expand_time_embedding(self.time_proj(t_emb), self.spatial_dim)

        if self.film is not None:
            out = self.film(out, condition)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + identity
        out = self.act2(out)
        
        return out


class ResNetStage(nn.Module):
    """
    A stage (group of blocks) in ResNet.
    
    Args:
        spatial_dim: Spatial dimensionality (1 or 2)
        in_channels: Input channels to stage
        out_channels: Output channels from stage
        num_blocks: Number of BasicBlocks in stage
        time_emb_dim: Dimension of time embedding
        stride: Stride for first block (for downsampling)
        norm_type: Normalization type
        activation: Activation function name
        use_film: Whether to apply FiLM conditioning inside blocks
        film_condition_dim: Conditioning vector dimension for FiLM
        film_hidden_dim: Hidden dimension for FiLM parameter generation
    """
    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        time_emb_dim: int = 128,
        stride: int = 1,
        norm_type: str = "group",
        activation: str = "swish",
        use_film: bool = False,
        film_condition_dim: Optional[int] = None,
        film_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        blocks = []
        
        # First block may change channels and/or downsample
        blocks.append(BasicBlock(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
            stride=stride,
            norm_type=norm_type,
            activation=activation,
            use_film=use_film,
            film_condition_dim=film_condition_dim,
            film_hidden_dim=film_hidden_dim,
        ))
        
        # Remaining blocks maintain channels
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(
                spatial_dim=spatial_dim,
                in_channels=out_channels,
                out_channels=out_channels,
                time_emb_dim=time_emb_dim,
                stride=1,
                norm_type=norm_type,
                activation=activation,
                use_film=use_film,
                film_condition_dim=film_condition_dim,
                film_hidden_dim=film_hidden_dim,
            ))
        
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x: Tensor, t_emb: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            x = block(x, t_emb, condition)
        return x


class ResNet(nn.Module):
    """
    ResNet architecture for flow matching on PDE data.
    
    This is a fully convolutional ResNet designed for PDE solving:
    - No global average pooling (preserves spatial structure)
    - No classification head (outputs full velocity field)
    - Time conditioning at every residual block
    - Configurable depth and width
    
    Architecture:
        Stem → [Stage1 → Stage2 → ... → StageN] → Output Conv
    
    For PDE solving, we typically don't downsample (stride=1 everywhere)
    to preserve spatial resolution for the velocity field output.
    
    Args:
        spatial_dim: Spatial dimensionality (1 or 2)
        spatial_size: Size of spatial domain
        base_channels: Base number of channels (doubled at each stage if downsample)
        blocks_per_stage: Number of blocks in each stage (list or int)
        solution_channels: Number of channels in solution (default: 1)
        condition_channels: Number of channels in condition (default: 1)
        kernel_size: Convolution kernel size (default: 3)
        norm_type: Normalization type ('group', 'batch', 'instance')
        activation: Activation function name ('swish', 'relu', 'gelu')
        downsample: Whether to downsample between stages (default: False for PDEs)
        return_spatial: If True, return spatial tensor; if False, flatten
    
    Example configurations:
        ResNet-8:  blocks_per_stage=[1, 1, 1, 1] with base_channels=32
        ResNet-14: blocks_per_stage=[2, 2, 2] with base_channels=64
        ResNet-18: blocks_per_stage=[2, 2, 2, 2] with base_channels=64
    """
    def __init__(
        self,
        spatial_dim: int,
        spatial_size: int,
        base_channels: int = 64,
        blocks_per_stage: List[int] = [2, 2, 2, 2],
        solution_channels: int = 1,
        condition_channels: int = 1,
        kernel_size: int = 3,
        norm_type: str = "group",
        activation: str = "swish",
        downsample: bool = False,
        return_spatial: bool = False,
        conditioner: Optional[BaseConditioner] = None,
    ):
        super().__init__()

        if spatial_dim not in [1, 2]:
            raise ValueError(f"spatial_dim must be 1 or 2, got {spatial_dim}")

        self.spatial_dim = spatial_dim
        self.spatial_size = spatial_size
        self.solution_channels = solution_channels
        self.condition_channels = condition_channels
        self.return_spatial = return_spatial
        self.num_stages = len(blocks_per_stage)
        self.conditioner = conditioner if conditioner is not None else ConcatConditioner(dim=1)
        self.use_film = isinstance(self.conditioner, FiLMConditioner)

        film_condition_dim = None
        film_hidden_dim = None
        if self.use_film:
            film_config = self.conditioner.get_config()
            film_condition_dim = film_config.get("condition_dim")
            film_hidden_dim = film_config.get("hidden_dim")
            if film_condition_dim is None:
                raise ValueError("FiLMConditioner requires condition_dim")

        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_embed = TimeMLPEmbedding(dim=time_emb_dim, activation=activation)

        Conv = get_conv_layer(spatial_dim)
        if self.use_film:
            in_channels = solution_channels
        else:
            in_channels = _input_channels_for_conditioner(
                self.conditioner, solution_channels, condition_channels
            )

        # Stem: initial convolution
        self.stem = nn.Sequential(
            Conv(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            get_norm_layer(norm_type, base_channels, spatial_dim),
            get_activation(activation),
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for i, num_blocks in enumerate(blocks_per_stage):
            # Determine output channels for this stage
            if downsample and i > 0:
                out_channels = min(current_channels * 2, base_channels * 8)  # Cap at 8x
                stride = 2
            else:
                out_channels = current_channels
                stride = 1
            
            self.stages.append(ResNetStage(
                spatial_dim=spatial_dim,
                in_channels=current_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                time_emb_dim=time_emb_dim,
                stride=stride,
                norm_type=norm_type,
                activation=activation,
                use_film=self.use_film,
                film_condition_dim=film_condition_dim,
                film_hidden_dim=film_hidden_dim,
            ))
            
            current_channels = out_channels
        
        # Output projection
        self.output_norm = get_norm_layer(norm_type, current_channels, spatial_dim)
        self.output_act = get_activation(activation)
        
        # If downsampled, need to upsample back
        if downsample:
            # Build upsampling path
            self.upsample = self._build_upsample_path(
                current_channels, solution_channels, 
                blocks_per_stage, base_channels, spatial_dim
            )
            self.output_conv = None
        else:
            self.upsample = None
            self.output_conv = Conv(current_channels, solution_channels, 3, padding=1)
        
        # Initialize weights
        init_weights(self, zero_init_last=True)
    
    def _build_upsample_path(
        self, 
        in_channels: int, 
        out_channels: int,
        blocks_per_stage: List[int],
        base_channels: int,
        spatial_dim: int
    ) -> nn.Sequential:
        """Build upsampling path to restore original resolution."""
        from .components import get_conv_transpose_layer
        
        ConvT = get_conv_transpose_layer(spatial_dim)
        Conv = get_conv_layer(spatial_dim)
        
        layers = []
        current_ch = in_channels
        
        # Upsample for each downsampling stage (except first)
        for i in range(len(blocks_per_stage) - 1):
            next_ch = max(current_ch // 2, base_channels)
            layers.extend([
                ConvT(current_ch, next_ch, kernel_size=2, stride=2),
                get_norm_layer("group", next_ch, spatial_dim),
                get_activation("swish"),
            ])
            current_ch = next_ch
        
        # Final projection
        layers.append(Conv(current_ch, out_channels, 3, padding=1))
        
        return nn.Sequential(*layers)
    
    def _reshape_input(self, tensor: Tensor, channels: int) -> Tensor:
        """Reshape flattened input to spatial format."""
        if tensor.dim() == 2:
            B = tensor.shape[0]
            if self.spatial_dim == 1:
                return tensor.view(B, channels, self.spatial_size)
            else:
                return tensor.view(B, channels, self.spatial_size, self.spatial_size)
        return tensor
    
    def forward(self, x: Tensor, f: Optional[Tensor], t: Tensor) -> Tensor:
        """
        Predict velocity field for flow matching.
        
        Args:
            x: State x_t, shape (B, C, *spatial) or flattened (B, C*spatial)
            f: Condition, shape (B, C', *spatial) or flattened (B, C'*spatial).
               Optional when using NullConditioner.
            t: Time t ∈ [0, 1], shape (B,) or (B, 1)
        
        Returns:
            Velocity field v(x, f, t)
        """
        # Reshape inputs
        x = self._reshape_input(x, self.solution_channels)
        if self.use_film:
            if f is None:
                raise ValueError("f must be provided when using FiLMConditioner")
        elif isinstance(self.conditioner, NullConditioner):
            f = None
        else:
            if f is None:
                raise ValueError("f must be provided when conditioner is not NullConditioner")
            f = self._reshape_input(f, self.condition_channels)
        
        B = x.shape[0]
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Apply conditioner (concat by default)
        if self.use_film:
            h = x
        else:
            h = self.conditioner(x, f)

        # Stem
        h = self.stem(h)
        
        # Apply stages
        for stage in self.stages:
            h = stage(h, t_emb, f)
        
        # Output projection
        h = self.output_norm(h)
        h = self.output_act(h)
        
        if self.upsample is not None:
            out = self.upsample(h)
        else:
            out = self.output_conv(h)
        
        # Return format
        if self.return_spatial:
            return out
        else:
            return out.view(B, -1)
    
    def extra_repr(self) -> str:
        return (
            f"spatial_dim={self.spatial_dim}, spatial_size={self.spatial_size}, "
            f"num_stages={self.num_stages}, "
            f"solution_channels={self.solution_channels}, "
            f"condition_channels={self.condition_channels}"
        )


def resnet8(spatial_dim: int, spatial_size: int, **kwargs) -> ResNet:
    """ResNet-8: Lightweight model for small grids."""
    return ResNet(
        spatial_dim=spatial_dim,
        spatial_size=spatial_size,
        base_channels=32,
        blocks_per_stage=[1, 1, 1, 1],
        **kwargs
    )


def resnet14(spatial_dim: int, spatial_size: int, **kwargs) -> ResNet:
    """ResNet-14: Medium model for moderate complexity."""
    return ResNet(
        spatial_dim=spatial_dim,
        spatial_size=spatial_size,
        base_channels=64,
        blocks_per_stage=[2, 2, 2],
        **kwargs
    )


def resnet18(spatial_dim: int, spatial_size: int, **kwargs) -> ResNet:
    """ResNet-18: Standard model for most PDE problems."""
    return ResNet(
        spatial_dim=spatial_dim,
        spatial_size=spatial_size,
        base_channels=64,
        blocks_per_stage=[2, 2, 2, 2],
        **kwargs
    )


def resnet26(spatial_dim: int, spatial_size: int, **kwargs) -> ResNet:
    """ResNet-26: Deeper model for complex PDEs."""
    return ResNet(
        spatial_dim=spatial_dim,
        spatial_size=spatial_size,
        base_channels=64,
        blocks_per_stage=[3, 3, 3, 3],
        **kwargs
    )
