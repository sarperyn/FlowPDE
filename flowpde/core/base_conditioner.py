"""
Base Conditioner Abstract Class

Defines the interface for conditioning mechanisms in FlowPDE.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import torch
from torch import nn, Tensor


class BaseConditioner(ABC, nn.Module):
    """
    Abstract base class for conditioning mechanisms.
    
    Conditioners process and inject conditioning information (e.g., PDE coefficients,
    boundary conditions, observations) into the flow model.
    
    Examples:
    - Concatenation: Simply concatenate condition with input
    - Cross-attention: Use attention mechanism to condition
    - FiLM: Feature-wise Linear Modulation
    - Adaptive Instance Normalization
    """
    
    def __init__(self, **kwargs):
        """
        Initialize conditioner.
        
        Args:
            **kwargs: Conditioner-specific parameters
        """
        super().__init__()
        self._config = kwargs
    
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        **kwargs
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Apply conditioning to input.
        
        Args:
            x: Input tensor (batch_size, dim) or (batch_size, channels, H, W)
            condition: Conditioning information
            **kwargs: Additional parameters
            
        Returns:
            conditioned_x: Conditioned tensor (same shape as x or transformed)
            OR
            Dictionary with conditioned features and auxiliary outputs
        """
        raise NotImplementedError
    
    def preprocess_condition(self, condition: Tensor) -> Tensor:
        """
        Preprocess conditioning information before use.
        
        Override this to implement custom preprocessing (e.g., normalization,
        embedding, feature extraction).
        
        Args:
            condition: Raw conditioning tensor
            
        Returns:
            Preprocessed conditioning tensor
        """
        return condition
    
    def get_config(self) -> Dict[str, Any]:
        """Get conditioner configuration."""
        return {
            'type': self.__class__.__name__,
            **self._config
        }
    
    def extra_repr(self) -> str:
        """Extra information for repr."""
        config_str = ', '.join(f'{k}={v}' for k, v in self._config.items())
        return config_str


class ConcatConditioner(BaseConditioner):
    """
    Simple concatenation-based conditioning.
    
    Concatenates condition with input along specified dimension.
    """
    
    def __init__(self, dim: int = 1):
        """
        Initialize concatenation conditioner.
        
        Args:
            dim: Dimension along which to concatenate (default: 1 for feature dimension)
        """
        super().__init__(dim=dim)
        self.dim = dim
    
    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Concatenate condition with input.
        
        Args:
            x: Input tensor
            condition: Conditioning tensor
            
        Returns:
            Concatenated tensor
        """
        condition = self.preprocess_condition(condition)
        
        # Ensure compatible shapes
        if x.dim() != condition.dim():
            # Try to reshape condition to match x
            if condition.dim() == 2 and x.dim() == 4:
                # Condition is (B, C), x is (B, C, H, W)
                # Reshape condition to (B, C, 1, 1) and broadcast
                condition = condition.view(condition.shape[0], condition.shape[1], 1, 1)
                condition = condition.expand(-1, -1, x.shape[2], x.shape[3])
        
        return torch.cat([x, condition], dim=self.dim)


class FiLMConditioner(BaseConditioner):
    """
    Feature-wise Linear Modulation (FiLM) conditioning.
    
    Generates scale and shift parameters from condition to modulate features.
    """
    
    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize FiLM conditioner.
        
        Args:
            condition_dim: Dimension of conditioning input
            feature_dim: Dimension of features to modulate
            hidden_dim: Hidden dimension for parameter generation (default: condition_dim)
        """
        super().__init__(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
        hidden_dim = hidden_dim or condition_dim
        
        # Network to generate scale and shift parameters
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim)  # 2x for scale and shift
        )
    
    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Apply FiLM conditioning.
        
        Args:
            x: Input features (batch_size, feature_dim, ...)
            condition: Conditioning information (batch_size, condition_dim)
            
        Returns:
            Modulated features: scale * x + shift
        """
        condition = self.preprocess_condition(condition)
        
        # Flatten condition if needed
        if condition.dim() > 2:
            condition = condition.flatten(start_dim=1)
        
        # Generate scale and shift parameters
        params = self.film_generator(condition)  # (batch_size, 2 * feature_dim)
        scale, shift = params.chunk(2, dim=1)    # Each (batch_size, feature_dim)
        
        # Reshape for broadcasting if x is spatial
        if x.dim() == 4:  # (B, C, H, W)
            scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
            shift = shift.view(shift.shape[0], shift.shape[1], 1, 1)
        elif x.dim() == 3:  # (B, L, C)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        # Apply FiLM: scale * x + shift
        return scale * x + shift


class NullConditioner(BaseConditioner):
    """
    No-op conditioner that returns input unchanged.
    
    Useful for unconditional models or as a placeholder.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Return input unchanged."""
        return x
