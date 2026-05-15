"""
Flow matching objective for neural ODE flows.

This module provides a single, configurable objective that encompasses:
- Standard Flow Matching (Lipman et al., 2023)
- Rectified Flow (Liu et al., 2023)
- OT-Conditional Flow Matching (Tong et al., 2023)

All variants are achieved through configuration, not inheritance.
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Optional, Tuple, Union, Any

from flowpde.flows import NeuralODEFlow
from flowpde.flows.components import (
    PathInterpolant,
    TimeSampler,
    Coupling,
    get_path,
    get_time_sampler,
    get_coupling,
)


class FlowMatchingObjective(nn.Module):
    """
    Flow-matching objective for ``NeuralODEFlow``.
    
    This objective trains a neural ODE flow by supervised velocity regression
    along interpolation paths instead of maximum likelihood. Multiple flow
    matching variants are configured through modular components:
    
    - **Path**: How to interpolate between noise and data
    - **Time Sampler**: Distribution for sampling training times
    - **Coupling**: How to pair noise and data samples
    
    Standard Configurations:
    
    1. **Flow Matching** (default):
       ```
       FlowMatchingObjective(flow, path='linear', time_sampler='uniform')
       ```
    
    2. **Rectified Flow**:
       ```
       FlowMatchingObjective(flow, path='linear', time_sampler='logit_normal')
       ```
    
    3. **OT-Conditional Flow Matching**:
       ```
       FlowMatchingObjective(flow, path='ot_conditional', sigma=0.01)
       ```
    
    Args:
        flow: Neural ODE flow whose model predicts velocity
            v(x_t, condition, t)
        path: Interpolation path ('linear', 'ot_conditional') or PathInterpolant
        time_sampler: Time distribution ('uniform', 'logit_normal') or TimeSampler
        coupling: Coupling strategy ('independent', 'minibatch_ot') or Coupling
        sigma: Noise level for OT-conditional path (default: 0.0)
        target_key: Default batch key for target tensors (default: 'u')
        condition_key: Default batch key for condition tensors (default: 'f')
    
    References:
        - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
        - Liu et al., "Flow Straight and Fast: Rectified Flow", ICLR 2023
        - Tong et al., "Conditional Flow Matching", NeurIPS 2023
    """
    
    def __init__(
        self,
        flow: NeuralODEFlow,
        path: Union[str, PathInterpolant] = "linear",
        time_sampler: Union[str, TimeSampler] = "uniform",
        coupling: Union[str, Coupling] = "independent",
        sigma: float = 0.0,
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
    ):
        super().__init__()
        self.flow = flow
        self.model = flow.model
        self.target_key = target_key or flow.target_key
        self.condition_key = condition_key or flow.condition_key
        
        # Initialize components
        # Pass sigma to OT path if needed
        if isinstance(path, str) and path in ['ot_conditional', 'conditional_optimal_transport', 'ot']:
            self.path = get_path(path, sigma=sigma)
        else:
            self.path = get_path(path)
        
        self.time_sampler = get_time_sampler(time_sampler)
        self.coupling = get_coupling(coupling)
        self.sigma = sigma
        
        # Store string names for config
        self._path_name = path if isinstance(path, str) else path.__class__.__name__
        self._time_sampler_name = time_sampler if isinstance(time_sampler, str) else time_sampler.__class__.__name__
        self._coupling_name = coupling if isinstance(coupling, str) else coupling.__class__.__name__
    
    def sample_base_distribution(
        self,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> Tensor:
        """Sample from base distribution (standard Gaussian)."""
        return torch.randn(*shape, device=device)

    @property
    def model_device(self) -> torch.device:
        return self.flow.model_device
    
    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
        **kwargs
    ) -> Tensor:
        """
        Compute flow matching loss.
        
        The loss minimizes the MSE between predicted and target velocities:
        
        $$\\mathcal{L} = \\mathbb{E}_{t, x_0, x_1}[\\|v_\\theta(x_t, f, t) - v_t\\|^2]$$
        
        where:
        - $x_t$ is the interpolated point on the path
        - $v_t$ is the target velocity (derivative of path)
        - $f$ is the conditioning information
        
        Args:
            batch: Dictionary containing target and condition tensors.
            target_key: Batch key for target data. Defaults to this flow's
                configured target key ('u' by default).
            condition_key: Batch key for conditioning data. Defaults to this
                flow's configured condition key ('f' by default).
        
        Returns:
            MSE loss tensor (scalar)
        """
        # Extract and prepare data
        x_1, condition = self.flow._extract_target_condition(
            batch,
            target_key=target_key or self.target_key,
            condition_key=condition_key or self.condition_key,
        )
        self.flow._target_dim = x_1.shape[1]
        batch_size = x_1.shape[0]
        
        # Sample from base distribution (noise)
        x_0 = self.sample_base_distribution(x_1.shape, self.model_device)
        
        # Apply coupling strategy
        x_0, x_1 = self.coupling(x_0, x_1)
        
        # Sample time
        t = self.time_sampler(batch_size, self.model_device)
        
        # Compute path interpolation and target velocity
        x_t, v_target = self.path(x_0, x_1, t)
        
        # Predict velocity with model
        v_pred = self.model(x_t, condition, t)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
    
    def sample(
        self,
        condition: Tensor,
        n_steps: int = 50,
        solver: str = 'euler',
        x_init: Optional[Tensor] = None,
        target_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        return_trajectory: bool = False,
        **solver_kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Generate samples by solving the flow ODE.
        
        Integrates the learned velocity field from t=0 (noise) to t=1 (data):
        
        $$\\frac{dx}{dt} = v_\\theta(x_t, f, t), \\quad x_0 \\sim \\mathcal{N}(0, I)$$
        
        Args:
            condition: Conditioning tensor (B, *)
            n_steps: Number of integration steps
            solver: ODE solver ('euler', 'midpoint', 'rk4', 'dopri5')
            x_init: Optional initial noise (default: sample from N(0,I))
            target_shape: Shape of generated targets excluding batch, or a
                flattened target dimension. Required before training when the
                target and condition dimensions differ.
            return_trajectory: If True, return full trajectory
            **solver_kwargs: Additional solver arguments
        
        Returns:
            Generated samples (B, dim)
            If return_trajectory: (samples, trajectory) where trajectory is (n_steps+1, B, dim)
        """
        from flowpde.solvers import ODEFlowSolver
        
        # Flatten condition
        condition_flat = condition.flatten(start_dim=1).to(self.model_device)
        batch_size = condition_flat.shape[0]
        if x_init is not None:
            dim = x_init.flatten(start_dim=1).shape[1]
        elif target_shape is not None:
            if isinstance(target_shape, int):
                dim = target_shape
            else:
                dim = 1
                for size in target_shape:
                    dim *= size
        else:
            dim = getattr(self.flow, "_target_dim", condition_flat.shape[1])
        
        # Sample initial noise if not provided
        if x_init is None:
            x_init = self.sample_base_distribution((batch_size, dim), self.model_device)
        else:
            x_init = x_init.flatten(start_dim=1).to(self.model_device)
        
        # Create solver and integrate
        ode_solver = ODEFlowSolver(model=self.model, method=solver, **solver_kwargs)
        
        samples = ode_solver.sample(
            condition=condition_flat,
            x_init=x_init,
            n_steps=n_steps,
        )
        
        return samples
    
    def estimate_straightness(
        self,
        batch: Dict[str, Tensor],
        n_time_points: int = 10,
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Estimate how straight the learned transport paths are.
        
        Straighter paths have more uniform velocities, meaning the model
        predicts similar v at different time points for the same (x_0, x_1) pair.
        
        Args:
            batch: Batch with target and condition tensors
            n_time_points: Number of time points to evaluate
            target_key: Batch key for target data
            condition_key: Batch key for conditioning data
        
        Returns:
            Dictionary with:
            - 'velocity_std': Std of velocity norms across time (lower = straighter)
            - 'straightness_score': 1 / (1 + velocity_std)
        """
        self.model.eval()
        
        x_1, condition = self.flow._extract_target_condition(
            batch,
            target_key=target_key or self.target_key,
            condition_key=condition_key or self.condition_key,
        )
        batch_size = x_1.shape[0]
        
        x_0 = self.sample_base_distribution(x_1.shape, self.model_device)
        
        with torch.no_grad():
            velocities = []
            time_points = torch.linspace(0.01, 0.99, n_time_points, device=self.model_device)
            
            for t_val in time_points:
                t = t_val.expand(batch_size, 1)
                x_t, _ = self.path(x_0, x_1, t)
                v_pred = self.model(x_t, condition, t)
                vel_norm = v_pred.norm(dim=1)
                velocities.append(vel_norm)
            
            # Stack: (n_time_points, batch_size)
            velocities = torch.stack(velocities, dim=0)
            
            # Std across time for each sample
            vel_std = velocities.std(dim=0).mean().item()
        
        return {
            'velocity_std': vel_std,
            'straightness_score': 1.0 / (1.0 + vel_std),
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        return {
            'objective': 'flow_matching',
            'flow': self.flow.get_config(),
            'path': self._path_name,
            'time_sampler': self._time_sampler_name,
            'coupling': self._coupling_name,
            'sigma': self.sigma,
            'target_key': self.target_key,
            'condition_key': self.condition_key,
            'model_type': self.model.__class__.__name__,
        }
    
    def __repr__(self) -> str:
        return (
            f"FlowMatchingObjective(\n"
            f"  path={self._path_name},\n"
            f"  time_sampler={self._time_sampler_name},\n"
            f"  coupling={self._coupling_name},\n"
            f"  sigma={self.sigma}\n"
            f")"
        )


def create_flow_matching(
    flow: NeuralODEFlow,
    variant: str = "standard",
    **kwargs
) -> FlowMatchingObjective:
    """
    Create a flow matching objective with preset configurations.
    
    Args:
        flow: Neural ODE flow
        variant: Preset name:
            - 'standard': Standard flow matching (linear, uniform)
            - 'rectified': Rectified flow (linear, logit-normal)
            - 'ot_cfm': OT-Conditional FM (ot_conditional, uniform)
        **kwargs: Override any default parameters
    
    Returns:
        Configured flow matching objective
    """
    presets = {
        'standard': {
            'path': 'linear',
            'time_sampler': 'uniform',
            'coupling': 'independent',
            'sigma': 0.0,
        },
        'rectified': {
            'path': 'linear',
            'time_sampler': 'logit_normal',
            'coupling': 'independent',
            'sigma': 0.0,
        },
        'ot_cfm': {
            'path': 'ot_conditional',
            'time_sampler': 'uniform',
            'coupling': 'independent',
            'sigma': 0.01,
        },
        'ot_cfm_coupled': {
            'path': 'ot_conditional',
            'time_sampler': 'uniform',
            'coupling': 'minibatch_ot',
            'sigma': 0.01,
        },
    }
    
    if variant not in presets:
        raise ValueError(f"Unknown variant: '{variant}'. Available: {list(presets.keys())}")
    
    config = presets[variant].copy()
    config.update(kwargs)
    
    return FlowMatchingObjective(flow, **config)
