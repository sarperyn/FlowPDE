import math
import torch
from torch import nn, Tensor

from flowpde.utils.activation_functions import Swish


# --- Fourier Time Embedding (better than linear for flows) ---
class FourierTimeEmbedding(nn.Module):
    """Sinusoidal time embedding used in diffusion models and flow matching.
    
    Maps scalar time t to high-dimensional feature vector using sinusoids
    at different frequencies. This is more expressive than linear embeddings.
    
    Args:
        dim: Embedding dimension (should be even)
        max_period: Maximum period for sinusoids (default: 10000)
    """
    def __init__(self, dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Time tensor (batch_size, 1) or (batch_size,)
        Returns:
            Embedding (batch_size, dim)
        """
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


# --- Self-Attention Block for global context ---
class AttentionBlock(nn.Module):
    """Self-attention block for U-Net bottleneck.
    
    Allows the model to capture global spatial dependencies,
    which is important for PDEs with long-range interactions.
    
    Args:
        channels: Number of input channels
        num_heads: Number of attention heads (default: 4)
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        # Ensure channels is divisible by num_heads
        self.num_heads = min(num_heads, channels)
        if channels % self.num_heads != 0:
            self.num_heads = 1
        
        num_groups = 32 if channels >= 32 else max(1, channels // 4)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(
            channels, 
            num_heads=self.num_heads, 
            batch_first=True
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C, H, W) with residual connection
        """
        B, C, H, W = x.shape
        
        # Normalize and reshape to sequence
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Self-attention
        h = self.attn(h, h, h, need_weights=False)[0]
        
        # Reshape back and add residual
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


# --- Basic conv block ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Use GroupNorm for stability with small batches (common in PDE training)
        # Dynamically set num_groups based on out_ch to handle varying channel counts
        num_groups = self._get_num_groups(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            Swish(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            Swish(),
        )

    def _get_num_groups(self, channels):
        """Dynamically determine number of groups for GroupNorm."""
        # Try common group sizes: 32, 16, 8, 4, 2, 1
        for num_groups in [32, 16, 8, 4, 2, 1]:
            if channels % num_groups == 0:
                return num_groups
        return 1  # fallback

    def forward(self, x): 
        return self.block(x)


# --- UNet with conditioning (x, f) and time embedding (t) ---
class UNet(nn.Module):
    """U-Net architecture for flow matching on spatial domains.
    
    A convolutional neural network with encoder-decoder architecture that
    automatically adjusts depth based on spatial resolution. Ideal for
    2D PDE problems.
    
    Improvements over basic U-Net:
    - Fourier time embeddings (sinusoidal features, better than linear)
    - Optional self-attention at bottleneck (captures global context)
    - Residual time conditioning at each layer
    
    Args:
        input_dim: Spatial resolution (assumes square grid) OR total flattened dimension
        base_ch: Base number of channels (doubled at each downsampling)
        time_dim: Dimension of time input (default: 1, embedded to higher dim)
        use_attention: Whether to use attention at bottleneck (default: True)
        solution_channels: Number of channels in solution (default: 1)
        condition_channels: Number of channels in condition (default: 2)
    """
    def __init__(
        self, 
        input_dim: int = 64, 
        base_ch: int = 64, 
        time_dim: int = 1, 
        use_attention: bool = False,
        solution_channels: int = 1,
        condition_channels: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_ch   = base_ch
        self.use_attention = use_attention
        self.solution_channels = solution_channels
        self.condition_channels = condition_channels

        # --- Fourier time embedding (better than linear) ---
        fourier_dim = base_ch * 2
        self.time_fourier = FourierTimeEmbedding(dim=fourier_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(fourier_dim, base_ch * 4),
            Swish(),
            nn.Linear(base_ch * 4, base_ch)
        )

        # --- Determine depth automatically ---
        # If input_dim is very large, it's likely flattened - calculate spatial size
        # Otherwise use it as spatial resolution
        if input_dim > 1024:  # Likely flattened (e.g., 3*32*32 = 3072)
            total_channels = solution_channels + condition_channels
            spatial_dim = int((input_dim / total_channels) ** 0.5)
        else:
            spatial_dim = input_dim
        
        max_depth = int(math.floor(math.log2(spatial_dim))) - 1  # e.g. 64→5, 128→6
        self.max_depth = max_depth

        # --- Encoder ---
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        in_ch = solution_channels + condition_channels  # x + f channels concatenated
        for i in range(max_depth):
            out_ch = base_ch * min(2 ** i, 16)  # cap growth
            self.downs.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(in_ch, in_ch * 2)
        
        # --- Optional attention at bottleneck for global context ---
        if self.use_attention:
            self.bottleneck_attn = AttentionBlock(in_ch * 2, num_heads=4)

        # --- Decoder ---
        self.ups, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        rev_chs = [block.block[0].out_channels for block in self.downs[::-1]]
        curr_ch = self.bottleneck.block[0].out_channels
        for skip_ch in rev_chs:
            self.ups.append(nn.ConvTranspose2d(curr_ch, skip_ch, 2, stride=2))
            self.up_blocks.append(ConvBlock(skip_ch * 2, skip_ch))
            curr_ch = skip_ch

        self.out_conv = nn.Conv2d(base_ch, self.solution_channels, 3, padding=1)

        # --- time projection for residual addition ---
        self.time_proj = nn.ModuleList([
            nn.Linear(base_ch, ch.block[0].out_channels) for ch in self.downs + [self.bottleneck] + self.up_blocks
        ])

    def forward(self, x: Tensor, f: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: State x_t ∈ ℝ^d (B, C, H, W) 
            f: Condition (B, C', H, W) - PDE parameters (forcing, coefficients, etc.)
            t: Time t ∈ [0,1] (B, 1)
            
        Returns:
            v_θ(x_t, f, t): Velocity field ∈ ℝ^d
        """
        # Reshape from flattened to spatial if needed
        if x.dim() == 2:
            total_size = x.size(1)
            spatial_size = total_size // self.solution_channels
            side = int(spatial_size ** 0.5)
            x = x.view(x.size(0), self.solution_channels, side, side)
        
        if f.dim() == 2:
            total_size = f.size(1)
            spatial_size = total_size // self.condition_channels
            side = int(spatial_size ** 0.5)
            f = f.view(f.size(0), self.condition_channels, side, side)

        B, _, H, W = x.shape
        
        # Fourier time embedding
        t_fourier = self.time_fourier(t)  # (B, fourier_dim)
        t_emb = self.time_mlp(t_fourier)  # (B, base_ch)

        # --- Encoder ---
        h = torch.cat([x, f], dim=1)  # Concatenate x_t and f
        skips = []
        for i, (down, pool) in enumerate(zip(self.downs, self.pools)):
            h = down(h)
            h = h + self.time_proj[i](t_emb)[:, :, None, None]  # add time conditioning
            skips.append(h)
            if h.shape[-2] > 1 and h.shape[-1] > 1:
                h = pool(h)

        # --- Bottleneck ---
        h = self.bottleneck(h)
        h = h + self.time_proj[len(self.downs)](t_emb)[:, :, None, None]
        
        # Apply attention at bottleneck for global context
        if self.use_attention:
            h = self.bottleneck_attn(h)

        # --- Decoder ---
        for j, (up, block, skip) in enumerate(zip(self.ups, self.up_blocks, reversed(skips))):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)
            idx = len(self.downs) + 1 + j
            h = h + self.time_proj[idx](t_emb)[:, :, None, None]

        out = self.out_conv(h)
        return out.view(B, -1)
