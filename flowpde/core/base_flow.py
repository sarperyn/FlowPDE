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
        **kwargs
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
        **kwargs
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
        **kwargs
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
    
    def sample(
        self,
        n_samples: int,
        condition: Optional[Tensor] = None,
        device: str = 'cuda',
        **kwargs
    ) -> Tensor:
        """
        Generate samples from the flow.
        
        Default implementation: sample from base distribution and apply inverse transform.
        Can be overridden for custom sampling strategies.
        
        Args:
            n_samples: Number of samples to generate
            condition: Optional conditioning information
            device: Device for computation
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated samples (n_samples, dim)
        """
        # Sample from base distribution
        if condition is not None:
            batch_size = condition.shape[0]
            if condition.dim() > 2:
                dim = condition.flatten(start_dim=1).shape[1]
            else:
                dim = condition.shape[1]
        else:
            batch_size = n_samples
            # Need to infer dimension - subclasses should override if needed
            dim = self._get_data_dim()
        
        z = self._sample_base_distribution(batch_size, dim, device)
        
        # Transform to data space
        x = self.inverse_transform(z, condition, **kwargs)
        if isinstance(x, tuple):
            x = x[0]  # Discard log_det if returned
        
        return x
    
    def log_prob(
        self,
        x: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Compute log probability of data under the flow.
        
        Default implementation using change of variables formula.
        Can be overridden for flows with different probability computation.
        
        Args:
            x: Data points (batch_size, dim)
            condition: Optional conditioning information
            **kwargs: Additional parameters
            
        Returns:
            Log probabilities (batch_size,)
        """
        # Transform to latent space
        result = self.forward_transform(x, condition, **kwargs)
        if isinstance(result, tuple):
            z, log_det = result
        else:
            z = result
            log_det = torch.zeros(x.shape[0], device=x.device)
        
        # Base distribution log prob
        log_pz = self._base_log_prob(z)
        
        # Change of variables: log p(x) = log p(z) + log |det J|
        log_px = log_pz + log_det
        
        return log_px
    
    def _sample_base_distribution(
        self,
        batch_size: int,
        dim: int,
        device: str
    ) -> Tensor:
        """Sample from the base distribution."""
        if self.base_distribution == 'gaussian':
            return torch.randn(batch_size, dim, device=device)
        elif self.base_distribution == 'uniform':
            return torch.rand(batch_size, dim, device=device)
        else:
            raise ValueError(f"Unknown base distribution: {self.base_distribution}")
    
    def _base_log_prob(self, z: Tensor) -> Tensor:
        """Compute log probability under base distribution."""
        if self.base_distribution == 'gaussian':
            # Standard normal log prob
            return -0.5 * (z.pow(2).sum(dim=1) + z.shape[1] * torch.log(2 * torch.tensor(torch.pi)))
        elif self.base_distribution == 'uniform':
            # Uniform [0,1] log prob
            return torch.zeros(z.shape[0], device=z.device)
        else:
            raise ValueError(f"Unknown base distribution: {self.base_distribution}")
    
    def _get_data_dim(self) -> int:
        """
        Get the dimension of the data space.
        Subclasses should override this if they track the data dimension.
        """
        raise NotImplementedError(
            "Data dimension unknown. Please provide condition or override _get_data_dim()"
        )
    
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
