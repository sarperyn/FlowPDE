"""
Flow Matching Trainer

Trainer for flow matching methods including conditional flow matching with
linear or optimal transport interpolation paths.
"""

import torch
from typing import Dict, Any
from torch import Tensor

from flowpde.trainers.trainer import Trainer
from flowpde.flows.flow_matching import FlowMatching


class FlowMatchingTrainer(Trainer):
    """
    Trainer for Flow Matching algorithms.
    
    Flow matching learns the conditional probability path between base and 
    target distributions. This trainer supports:
    - Linear interpolation paths
    - Conditional optimal transport (OT) paths
    - Custom conditioning for inverse problems
    
    The flow matching loss minimizes:
    
    $$\mathcal{L} = \mathbb{E}\left[\|v_\theta(x_t, \text{condition}, t) - v_{\text{target}}(t)\|^2\right]$$
    
    where $v_{\text{target}}$ is derived from the interpolation path.
    
    Args:
        flow: FlowMatching instance
        optimizer: PyTorch optimizer
        scheduler: Optional learning rate scheduler
        device: Device for training ('cuda' or 'cpu')
        
    Example:
        >>> from flowpde.flows import FlowMatching
        >>> from flowpde.models import UNet
        >>> 
        >>> # Create model and flow
        >>> model = UNet(input_dim=128, base_channels=64, time_dim=32)
        >>> flow = FlowMatching(model, path='conditional_optimal_transport')
        >>> 
        >>> # Create trainer
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> trainer = FlowMatchingTrainer(flow, optimizer, device='cuda')
        >>> 
        >>> # Train on any dataset (forward or inverse problems)
        >>> trainer.train(train_loader, epochs=100)
    """
    
    def __init__(
        self,
        flow: FlowMatching,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: str = 'cuda'
    ):
        """Initialize Flow Matching trainer."""
        # FlowMatching flow wraps the model
        super().__init__(
            model=flow.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        self.flow = flow.to(device)
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute flow matching loss.
        
        The batch should contain:
        - 'u': Target data (e.g., PDE solutions)
        - 'f': Conditioning information (e.g., PDE parameters, boundary conditions)
        
        For forward problems: condition on parameters, learn solution
        For inverse problems: condition on observations, learn parameter distribution
        
        Args:
            batch: Dictionary with 'u' and 'f' tensors
            
        Returns:
            Training loss tensor (MSE between predicted and target velocity)
        """
        # Flow matching already implements compute_loss in the flow object
        loss = self.flow.compute_loss(batch)
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
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional, can be parameterized)
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def get_flow_config(self) -> Dict[str, Any]:
        """Get flow configuration for saving."""
        return {
            'flow_type': 'flow_matching',
            'path': self.flow.path,
            'sigma': self.flow.sigma
        }
