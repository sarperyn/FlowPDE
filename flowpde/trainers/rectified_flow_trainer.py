"""
Rectified Flow Trainer

Trainer for rectified flows that learn straight transport paths
through iterative straightening procedures.
"""

import torch
from typing import Dict, Any
from torch import Tensor

from flowpde.trainers.trainer import Trainer
from flowpde.flows.rectified_flow import RectifiedFlow


class RectifiedFlowTrainer(Trainer):
    """
    Trainer for Rectified Flows.
    
    Rectified flow learns to transport between distributions along straight lines,
    leading to faster sampling and better generation quality through an iterative
    straightening procedure.
    
    The method minimizes:
    
    $$\mathcal{L} = \mathbb{E}\left[\|v_\theta(x_t, \text{condition}, t) - (x_1 - x_0)\|^2\right]$$
    
    where $x_t = (1-t)x_0 + t \cdot x_1$ and the target velocity is constant.
    
    Advantages over standard flow matching:
    - Faster sampling (requires fewer ODE steps)
    - Better generation quality  
    - More stable training
    - Can be iteratively refined (reflow)
    
    Args:
        flow: RectifiedFlow instance
        optimizer: PyTorch optimizer
        scheduler: Optional learning rate scheduler
        device: Device for training ('cuda' or 'cpu')
        
    Example:
        >>> from flowpde.flows import RectifiedFlow
        >>> from flowpde.models import UNet
        >>> 
        >>> # Create model and flow
        >>> model = UNet(input_dim=128, base_channels=64, time_dim=32)
        >>> flow = RectifiedFlow(model, time_sampling='uniform')
        >>> 
        >>> # Create trainer
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> trainer = RectifiedFlowTrainer(flow, optimizer, device='cuda')
        >>> 
        >>> # Train on any dataset (forward or inverse problems)
        >>> trainer.train(train_loader, epochs=100)
        >>> 
        >>> # Optionally: perform reflow iterations for straighter paths
        >>> flow.current_iteration += 1
        >>> trainer.train(train_loader, epochs=50)
    """
    
    def __init__(
        self,
        flow: RectifiedFlow,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: str = 'cuda'
    ):
        """Initialize Rectified Flow trainer."""
        super().__init__(
            model=flow.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        self.flow = flow.to(device)
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute rectified flow loss.
        
        The batch should contain:
        - 'u': Target data (e.g., PDE solutions)
        - 'f': Conditioning information (e.g., PDE parameters, boundary conditions)
        
        For forward problems: condition on parameters, learn straight paths to solutions
        For inverse problems: condition on observations, learn straight paths to parameters
        
        The loss encourages constant velocity along straight interpolation paths:
        
        $$\mathcal{L} = \mathbb{E}\left[\|v_\theta(x_t, \text{cond}, t) - (x_1 - x_0)\|^2\right]$$
        
        Args:
            batch: Dictionary with 'u' and 'f' tensors
            
        Returns:
            MSE loss tensor between predicted and target velocities
        """
        # Rectified flow already implements compute_loss
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
        
        # Compute loss (MSE on straight path velocities)
        loss = self.compute_loss(batch)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional, can be parameterized)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def get_flow_config(self) -> Dict[str, Any]:
        """Get flow configuration for saving."""
        return {
            'flow_type': 'rectified_flow',
            'time_sampling': self.flow.time_sampling,
            'reflow_iterations': self.flow.reflow_iterations,
            'current_iteration': self.flow.current_iteration
        }
