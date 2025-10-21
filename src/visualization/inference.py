"""Inference utilities for flow matching models.

Provides sampling, generation, and evaluation functions for trained flow matching models.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Union
from torch import nn, Tensor


@torch.no_grad()

def sample_flow_matching(
    model: nn.Module,
    condition: Tensor,
    n_steps: int = 50,
    device: str = 'cuda',
    integration_method: str = 'euler',
    return_trajectory: bool = False,
    x_init: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Sample from a trained flow matching model given conditioning.
    
    Args:
        model: Trained flow matching model (expects: model(x, cond, t))
        condition: Conditioning tensor (B, D) or (B, H, W)
        n_steps: Number of integration steps from t=0 to t=1
        device: Device for computation
        integration_method: 'euler' or 'midpoint' for ODE integration
        return_trajectory: If True, return full trajectory (n_steps+1, B, D)
        x_init: Optional initial noise (B, D). If None, sample from N(0, I)
    
    Returns:
        samples: Final samples at t=1, shape (B, D) or (B, H, W)
        trajectory: (optional) Full trajectory if return_trajectory=True
    """
    model.eval()
    
    # Flatten condition if needed
    original_shape = condition.shape
    if condition.dim() > 2:
        condition = condition.flatten(start_dim=1)
    
    condition = condition.float().to(device)
    batch_size, dim = condition.shape
    
    # Initialize from noise
    if x_init is None:
        x_t = torch.randn(batch_size, dim, device=device)
    else:
        x_t = x_init.flatten(start_dim=1).float().to(device)
    
    # Time steps from 0 to 1
    t_vals = torch.linspace(0, 1, n_steps + 1, device=device)
    dt = 1.0 / n_steps
    
    # Store trajectory if requested
    if return_trajectory:
        trajectory = [x_t.clone().cpu()]
    
    # Integrate ODE
    for i in range(n_steps):
        t_current = t_vals[i]
        t_next = t_vals[i + 1]
        
        if integration_method == 'midpoint':
            # Midpoint method
            t_mid = (t_current + t_next) / 2
            t_mid_batch = t_mid.repeat(batch_size, 1)
            v_t = model(x_t, condition, t_mid_batch)
        elif integration_method == 'rk4':
            # RK4 for even better accuracy
            t_batch = t_current.repeat(batch_size, 1)
            t_mid = (t_current + t_next) / 2
            t_mid_batch = t_mid.repeat(batch_size, 1)
            t_next_batch = t_next.repeat(batch_size, 1)
            
            k1 = model(x_t, condition, t_batch)
            k2 = model(x_t + 0.5 * dt * k1, condition, t_mid_batch)
            k3 = model(x_t + 0.5 * dt * k2, condition, t_mid_batch)
            k4 = model(x_t + dt * k3, condition, t_next_batch)
            
            v_t = (k1 + 2*k2 + 2*k3 + k4) / 6
        else:  # euler (default)
            t_batch = t_current.repeat(batch_size, 1)
            v_t = model(x_t, condition, t_batch)
        
        x_t = x_t + dt * v_t
        
        if return_trajectory:
            trajectory.append(x_t.clone().cpu())
    
    # Reshape to original spatial dimensions if needed
    if len(original_shape) > 2:
        x_t = x_t.view(original_shape)
    
    if return_trajectory:
        trajectory = torch.stack(trajectory, dim=0)  # (n_steps+1, B, D)
        return x_t, trajectory
    
    return x_t


@torch.no_grad()
def compute_prediction_error(
    model: nn.Module,
    dataloader,
    device: str = 'cuda',
    n_steps: int = 50,
    integration_method: str = 'euler',
    metric: str = 'mse'
) -> Dict[str, float]:
    """
    Compute prediction error on a dataset.
    
    Args:
        model: Trained flow matching model
        dataloader: DataLoader yielding (u, f) or {'u': ..., 'f': ...}
        device: Device for computation
        n_steps: Number of integration steps
        integration_method: ODE solver to use
        metric: 'mse', 'mae', or 'relative_l2'
    
    Returns:
        Dictionary with error metrics
    """
    model.eval()
    
    total_error = 0.0
    total_samples = 0
    
    for batch in dataloader:
        # Handle both formats
        if isinstance(batch, dict):
            u_true = batch['u'].to(device)
            f = batch['f'].to(device)
        else:
            u_true, f = batch
            u_true = u_true.to(device)
            f = f.to(device)
        
        # Sample predictions
        u_pred = sample_flow_matching(
            model=model,
            condition=f,
            n_steps=n_steps,
            device=device,
            integration_method=integration_method
        )
        
        # Flatten for error computation
        u_true_flat = u_true.flatten(start_dim=1)
        u_pred_flat = u_pred.flatten(start_dim=1)
        
        # Compute error
        if metric == 'mse':
            error = ((u_pred_flat - u_true_flat) ** 2).mean(dim=1)
        elif metric == 'relative_l2':
            error = ((u_pred_flat - u_true_flat).norm(dim=1) / 
                     (u_true_flat.norm(dim=1) + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        total_error += error.sum().item()
        total_samples += u_true.shape[0]
    
    mean_error = total_error / total_samples
    
    return {
        f'mean_{metric}': mean_error,
        'n_samples': total_samples
    }


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    conditions: Tensor,
    n_steps: int = 50,
    device: str = 'cuda',
    integration_method: str = 'euler',
    batch_size: Optional[int] = None
) -> Tensor:
    """
    Generate samples for a batch of conditions.
    
    Args:
        model: Trained flow matching model
        conditions: Conditioning tensor (N, D) or (N, H, W)
        n_steps: Number of integration steps
        device: Device for computation
        integration_method: ODE solver to use
        batch_size: Process in batches if memory is limited
    
    Returns:
        Generated samples, same shape as conditions
    """
    model.eval()
    
    if batch_size is None:
        # Process all at once
        return sample_flow_matching(
            model=model,
            condition=conditions,
            n_steps=n_steps,
            device=device,
            integration_method=integration_method
        )
    
    # Process in batches
    all_samples = []
    n_total = conditions.shape[0]
    
    for i in range(0, n_total, batch_size):
        batch_cond = conditions[i:i+batch_size]
        batch_samples = sample_flow_matching(
            model=model,
            condition=batch_cond,
            n_steps=n_steps,
            device=device,
            integration_method=integration_method
        )
        all_samples.append(batch_samples.cpu())
    
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def interpolate_conditions(
    model: nn.Module,
    condition_a: Tensor,
    condition_b: Tensor,
    n_interpolations: int = 10,
    n_steps: int = 50,
    device: str = 'cuda',
    integration_method: str = 'euler'
) -> Tensor:
    """
    Generate samples for interpolated conditions between two endpoints.
    
    Args:
        model: Trained flow matching model
        condition_a: First condition (1, D) or (D,) or (H, W)
        condition_b: Second condition, same shape as condition_a
        n_interpolations: Number of interpolation steps
        n_steps: ODE integration steps
        device: Device
        integration_method: ODE solver
    
    Returns:
        Interpolated samples (n_interpolations, D) or (n_interpolations, H, W)
    """
    model.eval()
    
    # Ensure batch dimension
    if condition_a.dim() == 1 or (condition_a.dim() == 2 and condition_a.shape[0] != 1):
        condition_a = condition_a.unsqueeze(0)
        condition_b = condition_b.unsqueeze(0)
    
    # Create interpolation weights
    alphas = torch.linspace(0, 1, n_interpolations, device=device)
    
    # Interpolate conditions
    interpolated_conditions = []
    for alpha in alphas:
        cond_interp = (1 - alpha) * condition_a + alpha * condition_b
        interpolated_conditions.append(cond_interp)
    
    interpolated_conditions = torch.cat(interpolated_conditions, dim=0)
    
    # Generate samples
    return sample_flow_matching(
        model=model,
        condition=interpolated_conditions,
        n_steps=n_steps,
        device=device,
        integration_method=integration_method
    )
