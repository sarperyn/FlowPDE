import math
import torch
from torch import nn, Tensor

from flowpde.utils.activation_functions import Swish


# --- Fourier Time Embedding for MLP ---
class FourierTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for better temporal representation.
    
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


class MLP(nn.Module):
    """Multi-Layer Perceptron for flow matching.
    
    A fully-connected neural network that predicts velocity fields for
    continuous normalizing flows. Suitable for low-dimensional problems.
    
    Improvements over basic MLP:
    - Fourier time embeddings (sinusoidal features, better than raw time)
    - Residual connections in middle layers (better gradient flow)
    - Configurable depth
    
    Args:
        input_dim: Dimension of input/output space
        time_dim: Dimension of time input (default: 1, embedded to higher dim)
        hidden_dim: Number of hidden units in each layer
        num_layers: Number of residual blocks (default: 4)
    """
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        
        # Fourier time embedding (better than raw time)
        fourier_dim = hidden_dim
        self.time_fourier = FourierTimeEmbedding(dim=fourier_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            Swish()
        )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),  # x + cond
            Swish()
        )
        
        # Middle layers with residual connections
        self.middle_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                Swish(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    

    def forward(self, x: Tensor, f: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: State $x_t \in \mathbb{R}^d$ (B, d)
            f: Condition (B, d) - PDE parameters
            t: Time $t \in [0,1]$ (B, 1)
        
        Returns:
            $v_\theta(x_t, f, t)$: Velocity field $\in \mathbb{R}^d$
        """
        # Save original shape
        sz = x.size()
        batch_size = sz[0]
        
        # Flatten
        x = x.reshape(batch_size, self.input_dim)
        f = f.reshape(batch_size, self.input_dim)
        t = t.reshape(batch_size, self.time_dim).float()
        
        # Fourier time embedding
        t_emb = self.time_fourier(t)
        t_emb = self.time_proj(t_emb)
        
        # Concatenate x_t and f
        h = torch.cat([x, f], dim=1)
        h = self.input_proj(h)  # (batch, hidden_dim)
        
        # Add time embedding
        h = h + t_emb
        
        # Apply middle layers with residual connections
        for layer in self.middle_layers:
            h = h + layer(h)  # Residual connection
        
        # Project to output
        output = self.output_proj(h)
        
        # Restore original shape
        return output.reshape(*sz)