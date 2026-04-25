"""
Continuous Normalizing Flow Trainer

Trainer for continuous normalizing flows using neural ODEs with
exact log probability computation via trace estimation.
"""

import torch
from typing import Dict, Any, Optional
from torch import Tensor

from flowpde.trainers.trainer import Trainer
from flowpde.flows.cnf import ContinuousNormalizingFlow


class CNFTrainer(Trainer):
    """
    Trainer for Continuous Normalizing Flows (CNF).
    
    CNF learns invertible transformations using neural ODEs and computes exact
    log probabilities via the instantaneous change of variables formula.
    The loss is maximum likelihood (negative log likelihood).
    
    CNF is more expensive than flow matching but provides:
    - Exact density estimation
    - Invertible transformations
    - Flexibility in velocity field design
    
    The training loss is:
    
    $$\mathcal{L} = -\mathbb{E}[\log p(x)] = -\mathbb{E}\left[\log p(z) - \int \text{tr}\left(\frac{\partial v}{\partial x}\right) dt\right]$$
    
    where the trace is computed using exact computation or Hutchinson estimator.
    
    Args:
        flow: ContinuousNormalizingFlow instance
        optimizer: PyTorch optimizer
        scheduler: Optional learning rate scheduler
        device: Device for training ('cuda' or 'cpu')
        
    Example:
        >>> from flowpde.flows import ContinuousNormalizingFlow
        >>> from flowpde.models import MLP
        >>> 
        >>> # Create model and flow
        >>> model = MLP(input_dim=128, time_dim=32, hidden_dim=256)
        >>> flow = ContinuousNormalizingFlow(
        ...     model, 
        ...     trace_estimator='hutchinson',
        ...     n_trace_samples=1
        ... )
        >>> 
        >>> # Create trainer
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> trainer = CNFTrainer(flow, optimizer, device='cuda')
        >>> 
        >>> # Train on any dataset (forward or inverse problems)
        >>> trainer.train(train_loader, epochs=100)
    """
    
    def __init__(
        self,
        flow: ContinuousNormalizingFlow,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: str = 'cuda',
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
    ):
        """Initialize CNF trainer."""
        super().__init__(
            model=flow.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        self.flow = flow.to(device)
        self.target_key = target_key or flow.target_key
        self.condition_key = condition_key or flow.condition_key
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute CNF loss (negative log likelihood).
        
        The batch should contain target and condition tensors, configured by
        target_key and condition_key.
        
        For forward problems: condition on parameters, learn solution distribution
        For inverse problems: condition on observations, learn parameter posterior
        
        CNF integrates backward in time to compute exact log probabilities:
        
        $$\log p(x) = \log p(z) - \int_0^1 \text{tr}\left(\frac{\partial v}{\partial x}\right) dt$$
        
        Args:
            batch: Dictionary with target and condition tensors
            
        Returns:
            Negative log likelihood loss tensor
        """
        # CNF already implements compute_loss (negative log likelihood)
        loss = self.flow.compute_loss(
            batch,
            target_key=self.target_key,
            condition_key=self.condition_key,
        )
        return loss
    
    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with loss value
        """
        self.optimizer.zero_grad()
        
        # Compute loss (NLL with trace estimation)
        loss = self.compute_loss(batch)
        
        # Backward pass (includes gradients through ODE solver)
        loss.backward()
        
        # Gradient clipping (important for ODE stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def get_flow_config(self) -> Dict[str, Any]:
        """Get flow configuration for saving."""
        return {
            'flow_type': 'cnf',
            'base_distribution': self.flow.base_distribution,
            'trace_estimator': self.flow.trace_estimator,
            'n_trace_samples': self.flow.n_trace_samples,
            'regularization': self.flow.regularization,
            'target_key': self.target_key,
            'condition_key': self.condition_key,
        }
