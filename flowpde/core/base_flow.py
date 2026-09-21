"""
Base Flow Abstract Class

Defines the interface for all normalizing flow types in FlowPDE.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union
import torch
from torch import nn, Tensor


class BaseFlow(ABC, nn.Module):
    """
    Abstract base class for all normalizing flows.
    
    This defines the common interface for invertible flow objects. Training
    objectives such as flow matching or maximum likelihood live outside the
    flow.
    
    Key Concepts:
    - Forward: Map from data distribution to base distribution (e.g., training)
    - Inverse: Map from base distribution to data distribution (e.g., sampling)
    - Log probability computation for density estimation
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_distribution: str = 'gaussian',
        target_key: str = 'u',
        condition_key: str = 'f',
        **kwargs: Any
    ):
        """
        Initialize base flow.
        
        Args:
            model: Neural network that parameterizes the flow
            base_distribution: Type of base distribution ('gaussian', 'uniform', etc.)
            target_key: Default batch key for target data
            condition_key: Default batch key for conditioning data
            **kwargs: Additional flow-specific parameters
        """
        super().__init__()
        self.model = model
        self.base_distribution = base_distribution
        self.target_key = target_key
        self.condition_key = condition_key
        self._extra_kwargs = kwargs
    
    @property
    def model_device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.model.parameters()).device

    def _extract_target_condition(
        self,
        batch: Dict[str, Tensor],
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Extract target and condition tensors from a training batch."""
        target_key = target_key or self.target_key
        condition_key = condition_key or self.condition_key

        missing = [key for key in (target_key, condition_key) if key not in batch]
        if missing:
            available = ", ".join(sorted(batch.keys()))
            expected = f"target_key='{target_key}', condition_key='{condition_key}'"
            raise KeyError(
                f"Batch is missing required key(s): {missing}. "
                f"Expected {expected}. Available keys: [{available}]"
            )

        target = batch[target_key].flatten(start_dim=1).to(self.model_device)
        condition = batch[condition_key].flatten(start_dim=1).to(self.model_device)
        return target, condition
    
    @abstractmethod
    def forward_transform(
        self,
        x: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs: Any
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Transform from data space to latent space (forward direction).
        
        This is typically used during training to map data to the base distribution.
        
        Args:
            x: Input tensor from data distribution (batch_size, dim)
            condition: Optional conditioning information (batch_size, cond_dim)
            **kwargs: Additional arguments
            
        Returns:
            z: Transformed tensor in latent space
            log_det (optional): Log determinant of Jacobian for probability computation
        """
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(
        self,
        z: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs: Any
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Transform from latent space to data space (inverse/sampling direction).
        
        This is typically used during inference to generate samples.
        
        Args:
            z: Input tensor from base distribution (batch_size, dim)
            condition: Optional conditioning information (batch_size, cond_dim)
            **kwargs: Additional arguments
            
        Returns:
            x: Transformed tensor in data space
            log_det (optional): Log determinant of Jacobian for probability computation
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(
        self,
        condition: Tensor,
        **kwargs: Any
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Generate samples from the flow, given conditioning information.

        Every flow in this package is conditional -- it models p(x | c) for a
        PDE condition c -- so the condition is the first argument and the
        batch size comes from it.  Subclasses define the rest of their
        sampling controls (solver, step count, initial noise).

        Args:
            condition: Conditioning information (batch_size, cond_dim)
            **kwargs: Flow-specific sampling parameters

        Returns:
            Generated samples
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(
        self,
        x: Tensor,
        condition: Tensor,
        **kwargs: Any
    ) -> Tensor:
        """
        Compute log probability of data under the flow.

        The base measure and how its density is evaluated are the subclass's
        business: they have to agree with whatever that subclass samples
        from, and a generic implementation here could only guess.

        Args:
            x: Data points (batch_size, dim)
            condition: Conditioning information
            **kwargs: Additional parameters

        Returns:
            Log probabilities (batch_size,)
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for the flow."""
        return {
            'type': self.__class__.__name__,
            'base_distribution': self.base_distribution,
            **self._extra_kwargs
        }
    
    def extra_repr(self) -> str:
        """Extra information for repr."""
        return f"base_distribution={self.base_distribution}"
