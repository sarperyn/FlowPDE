"""
Flow Matching implementation as a BaseFlow.

Flow matching provides a simple and efficient way to train continuous
normalizing flows by matching velocity fields along straight paths.
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Optional, Tuple

from flowpde.core.base_flow import BaseFlow


class FlowMatching(BaseFlow):
    """Flow Matching for Continuous Normalizing Flows.
    
    Implements conditional flow matching (Lipman et al., 2023) which learns
    to transport samples from a base distribution (Gaussian noise) to the
    target distribution by minimizing the difference between predicted and
    target velocity fields.
    
    Args:
        model: Neural network that predicts velocity v(x, condition, t)
        path: Interpolation path type ('linear' or 'conditional_optimal_transport')
        sigma: Noise level for conditional probability paths (default: 0.0)
    """
    
    def __init__(
        self,
        model: nn.Module,
        path: str = "linear",
        sigma: float = 0.0
    ):
        super().__init__(model)
        self.path = path
        self.sigma = sigma
        
        if path not in ['linear', 'conditional_optimal_transport']:
            raise ValueError(f"Unknown path type: {path}")
    
    def sample_base_distribution(
        self,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> Tensor:
        """Sample from base distribution (standard Gaussian)."""
        return torch.randn(*shape, device=device)
    
    def compute_conditional_flow(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute conditional flow path and target velocity.
        
        Args:
            x_0: Samples from base distribution (noise)
            x_1: Samples from target distribution (data)
            t: Time values in [0, 1]
        
        Returns:
            x_t: Interpolated samples at time t
            v_t: Target velocity at time t
        """
        if self.path == "linear":
            return self._linear_flow(x_0, x_1, t)
        elif self.path == "conditional_optimal_transport":
            return self._ot_flow(x_0, x_1, t)
        else:
            raise ValueError(f"Unknown path: {self.path}")
    
    def _linear_flow(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Linear interpolation path: $x_t = (1-t)x_0 + t \cdot x_1$."""
        # t shape: (batch_size, 1)
        # Expand t to match x dimensions
        t_expanded = t.view(-1, *([1] * (x_0.dim() - 1)))
        
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        v_t = x_1 - x_0  # Constant velocity along straight lines
        
        return x_t, v_t
    
    def _ot_flow(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Optimal transport conditional flow with Gaussian noise.
        
        Based on conditional optimal transport formulation:
        
        $$x_t = t \cdot x_1 + (1-t) \cdot x_0 + \sigma \sqrt{t(1-t)} \cdot \text{noise}$$
        """
        t_expanded = t.view(-1, *([1] * (x_0.dim() - 1)))
        
        if self.sigma > 0:
            noise = torch.randn_like(x_0)
            noise_coeff = self.sigma * torch.sqrt(t_expanded * (1 - t_expanded))
            x_t = t_expanded * x_1 + (1 - t_expanded) * x_0 + noise_coeff * noise
        else:
            x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        
        v_t = x_1 - x_0
        
        return x_t, v_t
    
    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        """
        Compute flow matching loss.
        
        Args:
            batch: Dictionary containing:
                - 'u': Target data (solutions)
                - 'f': Conditioning data (source terms)
        
        Returns:
            MSE loss between predicted and target velocities
        """
        # Extract data
        x_1 = batch["u"].flatten(start_dim=1).to(self.model_device)
        condition = batch["f"].flatten(start_dim=1).to(self.model_device)
        
        # Sample base distribution
        x_0 = self.sample_base_distribution(x_1.shape, self.model_device)
        
        # Sample random times
        batch_size = x_1.shape[0]
        t = torch.rand(batch_size, 1, device=self.model_device)
        
        # Compute conditional flow
        x_t, v_target = self.compute_conditional_flow(x_0, x_1, t)
        
        # Predict velocity with model
        v_pred = self.model(x_t, condition, t)
        
        # Compute loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
    
    def sample(
        self,
        condition: Tensor,
        n_steps: int = 50,
        solver: str = 'dopri5',
        x_init: Optional[Tensor] = None,
        **solver_kwargs
    ) -> Tensor:
        """
        Sample from the learned flow.
        
        Args:
            condition: Conditioning tensor
            n_steps: Number of integration steps
            solver: ODE solver name ()
            x_init: Optional initial noise
            **solver_kwargs: Additional arguments for solver
        
        Returns:
            Generated samples
        """
        from flowpde.solvers import ODEFlowSolver
        
        # Create ODE solver
        ode_solver = ODEFlowSolver(model=self.model, method=solver, **solver_kwargs)
        
        # Sample
        samples = ode_solver.sample(
            condition=condition,
            x_init=x_init,
            n_steps=n_steps
        )
        
        return samples
    
    def log_prob(
        self,
        x: Tensor,
        condition: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Compute log probability (requires trace computation).
        
        Note: This requires instantaneous change of variables and trace
        estimation, which is computationally expensive. For most applications,
        use the sample() method instead.
        """
        raise NotImplementedError(
            "Log probability computation requires trace estimation. "
            "Use sample() for generation or implement with FFJORD."
        )
    
    def forward_transform(self, x: Tensor, condition: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Forward transformation (data -> noise).
        
        For flow matching, this would require solving the ODE backward in time,
        which is typically not needed during training.
        """
        raise NotImplementedError(
            "Forward transform not implemented for flow matching. "
            "Use sample() for generation (noise -> data)."
        )
    
    def inverse_transform(self, z: Tensor, condition: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Inverse transformation (noise -> data).
        
        This is equivalent to the sample() method.
        """
        return self.sample(condition=condition, **kwargs)
    
    def get_config(self) -> Dict:
        """Return configuration dictionary."""
        return {
            'flow_type': 'flow_matching',
            'path': self.path,
            'sigma': self.sigma,
            'model_type': self.model.__class__.__name__
        }
