import math
import torch
from torch import nn, Tensor

from utils.activation_functions import Swish


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
    """U-Net that automatically sets depth based on input_dim, conditioned on f and time t."""
    def __init__(self, input_dim: int = 64, base_ch: int = 64, time_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.base_ch   = base_ch

        # --- time embedding ---
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, base_ch),
            Swish(),
            nn.Linear(base_ch, base_ch)
        )

        # --- Determine depth automatically ---
        max_depth = int(math.floor(math.log2(input_dim))) - 1  # e.g. 64→5, 128→6
        self.max_depth = max_depth

        # --- Encoder ---
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        in_ch = 2  # x + f channels
        for i in range(max_depth):
            out_ch = base_ch * min(2 ** i, 16)  # cap growth
            self.downs.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(in_ch, in_ch * 2)

        # --- Decoder ---
        self.ups, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        rev_chs = [block.block[0].out_channels for block in self.downs[::-1]]
        curr_ch = self.bottleneck.block[0].out_channels
        for skip_ch in rev_chs:
            self.ups.append(nn.ConvTranspose2d(curr_ch, skip_ch, 2, stride=2))
            self.up_blocks.append(ConvBlock(skip_ch * 2, skip_ch))
            curr_ch = skip_ch

        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

        # --- time projection for residual addition ---
        self.time_proj = nn.ModuleList([
            nn.Linear(base_ch, ch.block[0].out_channels) for ch in self.downs + [self.bottleneck] + self.up_blocks
        ])

    def forward(self, x: Tensor, f: Tensor, t: Tensor) -> Tensor:
        if x.dim() == 2:
            side = int(x.size(1) ** 0.5)
            x = x.view(x.size(0), 1, side, side)
        if f.dim() == 2:
            side = int(f.size(1) ** 0.5)
            f = f.view(f.size(0), 1, side, side)

        B, _, H, W = x.shape
        t_emb = self.time_mlp(t)  # (B, base_ch)

        # --- Encoder ---
        h = torch.cat([x, f], dim=1) # The dimensions --> (B, 2, H, W)
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
