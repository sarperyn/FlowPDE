"""
Continuous Normalizing Flows (CNF) with FFJORD-style trace estimation.

Continuous normalizing flows learn invertible transformations using neural ODEs
and can compute exact log probabilities via the instantaneous change of variables.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, Callable, Union
import warnings

from flowpde.core.base_flow import BaseFlow


class CNFVectorField(nn.Module):
    """
    Vector field wrapper for CNF that includes log probability computation.
    
    Augments state with log probability and computes trace of Jacobian
    for the instantaneous change of variables formula.
    """
    
    def __init__(
        self,
        model: nn.Module,
        condition: Tensor,
        trace_estimator: str = 'hutchinson',
        n_trace_samples: int = 1
    ):
        super().__init__()
        self.model = model
        self.condition = condition
        self.trace_estimator = trace_estimator
        self.n_trace_samples = n_trace_samples
    
    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        """
        Compute augmented dynamics: $[dx/dt, d(\log p)/dt]$.
        
        Args:
            t: Current time (scalar)
            state: Augmented state $[x, \log p_x]$ with shapes:
                   x: (batch_size, dim)
                   $\log p_x$: (batch_size, 1)
        
        Returns:
            Augmented dynamics $[dx/dt, d(\log p_x)/dt]$
        """
        batch_size = state.shape[0]
        
        # Split augmented state
        x = state[:, :-1]  # (batch_size, dim)
        log_px = state[:, -1:]  # (batch_size, 1)
        
        # Prepare time for model
        t_batch = t.expand(batch_size, 1)
        
        # Enable gradients for trace computation
        with torch.enable_grad():
            x_requires_grad = x.requires_grad
            x = x.requires_grad_(True)
            
            # Compute velocity
            v = self.model(x, self.condition, t_batch)
            
            # Compute trace of Jacobian
            trace = self._compute_trace(v, x)
            
            # Restore gradient state
            x = x.requires_grad_(x_requires_grad)
        
        # Change of variables: $d(\log p)/dt = -\text{tr}(\partial v/\partial x)$
        dlogpx_dt = -trace.view(batch_size, 1)
        
        # Combine into augmented dynamics
        dstate_dt = torch.cat([v, dlogpx_dt], dim=1)
        
        return dstate_dt
    
    def _compute_trace(self, v: Tensor, x: Tensor) -> Tensor:
        """
        Compute trace of Jacobian $\partial v/\partial x$.
        
        Args:
            v: Velocity field (batch_size, dim)
            x: State (batch_size, dim)
        
        Returns:
            Trace (batch_size,)
        """
        if self.trace_estimator == 'exact':
            return self._exact_trace(v, x)
        elif self.trace_estimator == 'hutchinson':
            return self._hutchinson_trace(v, x)
        else:
            raise ValueError(f"Unknown trace estimator: {self.trace_estimator}")
    
    def _exact_trace(self, v: Tensor, x: Tensor) -> Tensor:
        """Compute exact trace by summing diagonal of Jacobian."""
        batch_size, dim = v.shape
        trace = torch.zeros(batch_size, device=v.device, dtype=v.dtype)
        
        for i in range(dim):
            grad_outputs = torch.zeros_like(v)
            grad_outputs[:, i] = 1
            
            dvi_dx = torch.autograd.grad(
                v, x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            trace += dvi_dx[:, i]
        
        return trace
    
    def _hutchinson_trace(self, v: Tensor, x: Tensor) -> Tensor:
        """
        Hutchinson trace estimator: $\mathbb{E}[\varepsilon^T (\partial v/\partial x) \varepsilon]$ where $\varepsilon \sim \mathcal{N}(0, I)$.
        
        Unbiased estimator that only requires one Jacobian-vector product.
        """
        batch_size, dim = v.shape
        
        # Sample random vectors
        epsilon = torch.randn(
            self.n_trace_samples, batch_size, dim,
            device=v.device, dtype=v.dtype
        )
        
        traces = []
        for eps in epsilon:
            # Compute Jacobian-vector product
            jvp = torch.autograd.grad(
                v, x,
                grad_outputs=eps,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Trace estimate: ε^T * jvp
            trace = (eps * jvp).sum(dim=1)
            traces.append(trace)
        
        # Average over samples
        return torch.stack(traces).mean(dim=0)


class ContinuousNormalizingFlow(BaseFlow):
    """
    Continuous Normalizing Flow (CNF) with exact log probability.
    
    CNF uses neural ODEs to learn invertible transformations and can compute
    exact log probabilities via the instantaneous change of variables formula:
    
    $$\log p(x_1) = \log p(x_0) - \int_0^1 \text{tr}\left(\frac{\partial f}{\partial x}\right) dt$$
    
    This is more expensive than flow matching but provides density estimation.
    
    Args:
        model: Neural network that computes velocity $v(x, \text{condition}, t)$
        base_distribution: Base distribution for sampling ('gaussian' or 'uniform')
        trace_estimator: Method for trace computation ('exact' or 'hutchinson')
        n_trace_samples: Number of samples for Hutchinson estimator
        regularization: Weight for regularization term (optional)
        target_key: Default batch key for target tensors (default: 'u')
        condition_key: Default batch key for condition tensors (default: 'f')
    
    References:
        - Grathwohl et al., "FFJORD: Free-form Continuous Dynamics for Scalable
          Reversible Generative Models", ICLR 2019
        - Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_distribution: str = 'gaussian',
        trace_estimator: str = 'hutchinson',
        n_trace_samples: int = 1,
        regularization: float = 0.0,
        target_key: str = "u",
        condition_key: str = "f",
    ):
        super().__init__(
            model,
            target_key=target_key,
            condition_key=condition_key,
        )
        self.base_distribution = base_distribution
        self.trace_estimator = trace_estimator
        self.n_trace_samples = n_trace_samples
        self.regularization = regularization
        
        if trace_estimator not in ['exact', 'hutchinson']:
            raise ValueError(f"Unknown trace estimator: {trace_estimator}")
    
    def sample_base_distribution(
        self,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> Tensor:
        """Sample from base distribution."""
        if self.base_distribution == 'gaussian':
            return torch.randn(*shape, device=device)
        elif self.base_distribution == 'uniform':
            return torch.rand(*shape, device=device) * 2 - 1
        else:
            raise ValueError(f"Unknown base distribution: {self.base_distribution}")
    
    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
        **kwargs
    ) -> Tensor:
        """
        Compute CNF loss via maximum likelihood.
        
        The loss is the negative log likelihood:
        
        $$\mathcal{L} = -\log p(x_1) = -\log p(x_0) + \int_0^1 \text{tr}\left(\frac{\partial f}{\partial x}\right) dt$$
        
        Args:
            batch: Dictionary containing target and condition tensors.
            target_key: Batch key for target data. Defaults to this flow's
                configured target key ('u' by default).
            condition_key: Batch key for conditioning data. Defaults to this
                flow's configured condition key ('f' by default).
        
        Returns:
            Negative log likelihood loss
        """
        # Extract data
        x_1, condition = self._extract_target_condition(
            batch,
            target_key=target_key,
            condition_key=condition_key,
        )
        self._target_dim = x_1.shape[1]
        batch_size = x_1.shape[0]
        
        # Forward pass: transform data to base distribution
        # We integrate backward in time from t=1 to t=0
        x_0, delta_logp = self._integrate_ode(
            x_1, condition, 
            t_span=(1.0, 0.0),  # Backward
            compute_logp=True
        )
        
        # Compute log probability at base
        if self.base_distribution == 'gaussian':
            log_p0 = -0.5 * (x_0 ** 2).sum(dim=1) - 0.5 * x_0.shape[1] * torch.log(
                torch.tensor(2 * 3.14159265, device=x_0.device)
            )
        elif self.base_distribution == 'uniform':
            # Uniform on [-1, 1]
            in_support = ((x_0 >= -1) & (x_0 <= 1)).all(dim=1).float()
            log_p0 = torch.log(in_support / (2 ** x_0.shape[1]))
        
        # Log probability at data: log p(x_1) = log p(x_0) + delta_logp
        log_px = log_p0 + delta_logp
        
        # Negative log likelihood loss
        nll = -log_px.mean()
        
        # Optional: Add regularization on velocity field norm
        if self.regularization > 0:
            t_reg = torch.rand(batch_size, 1, device=self.model_device)
            x_reg = x_1.detach()
            v_reg = self.model(x_reg, condition, t_reg)
            reg_loss = self.regularization * (v_reg ** 2).mean()
            nll = nll + reg_loss
        
        return nll
    
    def _integrate_ode(
        self,
        x: Tensor,
        condition: Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        compute_logp: bool = False,
        n_steps: Optional[int] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Integrate ODE with optional log probability tracking.
        
        Args:
            x: Initial state
            condition: Conditioning tensor
            t_span: Time interval (t_start, t_end)
            compute_logp: Whether to track log probability
            n_steps: Number of steps (optional)
        
        Returns:
            Final state and change in log probability
        """
        from torchdiffeq import odeint
        
        batch_size = x.shape[0]
        device = x.device
        
        if compute_logp:
            # Augment state with log probability
            log_px = torch.zeros(batch_size, 1, device=device)
            state = torch.cat([x, log_px], dim=1)
            
            # Create augmented vector field
            vector_field = CNFVectorField(
                self.model, condition,
                trace_estimator=self.trace_estimator,
                n_trace_samples=self.n_trace_samples
            )
        else:
            state = x
            # Regular vector field (no log probability)
            from flowpde.solvers.ode_solvers import VelocityField
            vector_field = VelocityField(self.model, condition)
        
        # Time points
        t_start, t_end = t_span
        t_eval = torch.tensor([t_start, t_end], device=device)
        
        # Integrate
        trajectory = odeint(
            vector_field,
            state,
            t_eval,
            method='dopri5',
            rtol=1e-5,
            atol=1e-7
        )
        
        final_state = trajectory[-1]
        
        if compute_logp:
            x_final = final_state[:, :-1]
            delta_logp = final_state[:, -1]
            return x_final, delta_logp
        else:
            return final_state, None
    
    def sample(
        self,
        condition: Tensor,
        n_steps: int = 50,
        solver: str = 'dopri5',
        x_init: Optional[Tensor] = None,
        target_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        **solver_kwargs
    ) -> Tensor:
        """
        Sample from the learned distribution.
        
        Args:
            condition: Conditioning tensor
            n_steps: Number of integration steps (ignored for adaptive solvers)
            solver: ODE solver name
            x_init: Optional initial noise
            target_shape: Shape of generated targets excluding batch. Required
                before training when target and condition dimensions differ.
            **solver_kwargs: Additional solver arguments
        
        Returns:
            Generated samples
        """
        condition = condition.flatten(start_dim=1).to(self.model_device)
        batch_size = condition.shape[0]
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
            dim = getattr(self, "_target_dim", condition.shape[1])
        
        # Sample from base distribution
        if x_init is None:
            x_0 = self.sample_base_distribution((batch_size, dim), self.model_device)
        else:
            x_0 = x_init.flatten(start_dim=1).to(self.model_device)
        
        # Integrate forward from t=0 to t=1
        x_1, _ = self._integrate_ode(
            x_0, condition,
            t_span=(0.0, 1.0),
            compute_logp=False,
            n_steps=n_steps
        )
        
        return x_1
    
    def log_prob(
        self,
        x: Tensor,
        condition: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Compute log probability of data.
        
        Args:
            x: Data samples
            condition: Conditioning tensor
        
        Returns:
            Log probabilities (batch_size,)
        """
        x = x.flatten(start_dim=1).to(self.model_device)
        condition = condition.flatten(start_dim=1).to(self.model_device)
        
        # Transform to base distribution
        x_0, delta_logp = self._integrate_ode(
            x, condition,
            t_span=(1.0, 0.0),
            compute_logp=True
        )
        
        # Compute base log probability
        if self.base_distribution == 'gaussian':
            log_p0 = -0.5 * (x_0 ** 2).sum(dim=1) - 0.5 * x_0.shape[1] * torch.log(
                torch.tensor(2 * 3.14159265, device=x_0.device)
            )
        else:
            in_support = ((x_0 >= -1) & (x_0 <= 1)).all(dim=1).float()
            log_p0 = torch.log(in_support / (2 ** x_0.shape[1]) + 1e-10)
        
        # Log probability at data
        log_px = log_p0 + delta_logp
        
        return log_px
    
    def forward_transform(
        self,
        x: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Forward transformation: data -> latent (backward ODE).
        
        Args:
            x: Data samples
            condition: Conditioning tensor
        
        Returns:
            Latent samples in base distribution
        """
        if condition is None:
            raise ValueError("CNF requires conditioning")
        
        x = x.flatten(start_dim=1).to(self.model_device)
        condition = condition.flatten(start_dim=1).to(self.model_device)
        
        # Integrate backward
        z, _ = self._integrate_ode(
            x, condition,
            t_span=(1.0, 0.0),
            compute_logp=False
        )
        
        return z
    
    def inverse_transform(
        self,
        z: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Inverse transformation: latent -> data (forward ODE).
        
        Args:
            z: Latent samples from base distribution
            condition: Conditioning tensor
        
        Returns:
            Data samples
        """
        return self.sample(condition=condition, x_init=z, **kwargs)
    
    def get_config(self) -> Dict:
        """Return configuration dictionary."""
        return {
            'flow_type': 'cnf',
            'base_distribution': self.base_distribution,
            'trace_estimator': self.trace_estimator,
            'n_trace_samples': self.n_trace_samples,
            'regularization': self.regularization,
            'target_key': self.target_key,
            'condition_key': self.condition_key,
            'model_type': self.model.__class__.__name__
        }
