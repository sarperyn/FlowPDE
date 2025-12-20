"""
ODE Solvers for Flow Matching Inference using torchdiffeq.

This module provides a unified interface for using various ODE solvers
from the torchdiffeq library for flow matching inference. The solvers
integrate the velocity field from t=0 (noise) to t=1 (solution).

Available solvers:
- Explicit methods: euler, midpoint, rk4, dopri5 (adaptive)
- Implicit methods: dopri8, adaptive_heun
- Specialized: bosh3, tsit5

Refactored to inherit from flowpde.core.base_solver.ODESolver
"""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from torchdiffeq import odeint, odeint_adjoint

from flowpde.core.base_solver import ODESolver


class VelocityField(nn.Module):
    """
    Wraps a flow matching model as an ODE velocity field.
    
    The flow matching model predicts $v(x_t, \text{condition}, t)$, which is the
    time derivative $dx/dt$ at time $t$. This wrapper makes it compatible
    with torchdiffeq's interface.
    """
    
    def __init__(self, model: nn.Module, condition: Tensor):
        """
        Args:
            model: Flow matching model with signature model(x, condition, t)
            condition: Conditioning tensor (batch_size, dim) - fixed during integration
        """
        super().__init__()
        self.model = model
        self.condition = condition
        
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        Compute velocity field at time $t$.
        
        Args:
            t: Current time (scalar tensor)
            x: Current state (batch_size, dim)
            
        Returns:
            Velocity $dx/dt$ at time $t$
        """
        # torchdiffeq passes scalar t, but model expects (batch_size, 1)
        batch_size = x.shape[0]
        t_batch = t.expand(batch_size, 1)
        
        # Compute velocity
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            v = self.model(x, self.condition, t_batch)
        
        return v


class ODEFlowSolver(ODESolver):
    """
    ODE solver for flow matching inference using torchdiffeq.
    
    This class provides a high-level interface for sampling from flow
    matching models using various ODE solvers from torchdiffeq.
    
    Inherits from flowpde.core.base_solver.ODESolver
    """
    
    # Available solvers and their properties
    ADAPTIVE_SOLVERS = ['dopri5', 'dopri8', 'bosh3', 'adaptive_heun', 'tsit5']
    FIXED_STEP_SOLVERS = ['euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']
    
    def __init__(
        self,
        model: nn.Module,
        method: str = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-7,
        adjoint: bool = False,
        method_options: Optional[dict] = None
    ):
        """
        Initialize ODE solver.
        
        Args:
            model: Flow matching model with signature model(x, condition, t)
            method: ODE solver method. Options:
                - 'dopri5': Runge-Kutta 4(5) (adaptive, recommended)
                - 'dopri8': Runge-Kutta 7(8) (high accuracy)
                - 'bosh3': Bogacki-Shampine 2(3) (faster, less accurate)
                - 'tsit5': Tsitouras 5(4) (good balance)
                - 'euler': Explicit Euler (simple, fast, less accurate)
                - 'midpoint': Explicit midpoint
                - 'rk4': Classic 4th order Runge-Kutta
                - 'adaptive_heun': Adaptive Heun's method
            rtol: Relative tolerance (for adaptive solvers)
            atol: Absolute tolerance (for adaptive solvers)
            adjoint: If True, use adjoint method for memory-efficient backprop
            method_options: Additional options for the solver
        """
        super().__init__(
            method=method,
            rtol=rtol,
            atol=atol,
            method_options=method_options
        )
        
        self.model = model
        self.adjoint = adjoint
        
        # Select integration function
        self.odeint_fn = odeint_adjoint if adjoint else odeint
        
        # Validate method
        all_methods = self.ADAPTIVE_SOLVERS + self.FIXED_STEP_SOLVERS
        if method not in all_methods:
            raise ValueError(
                f"Unknown method '{method}'. Available: {all_methods}"
            )
    
    @property
    def is_adaptive(self) -> bool:
        """Whether this is an adaptive step-size solver."""
        return self.method in self.ADAPTIVE_SOLVERS
    
    @property
    def supports_adjoint(self) -> bool:
        """Whether this solver supports adjoint method for backprop."""
        return True
    
    def solve(
        self,
        func: Callable[[Tensor, Tensor], Tensor],
        y0: Tensor,
        t_span: Tuple[float, float],
        **kwargs
    ) -> Tensor:
        """
        Solve the ODE $dy/dt = f(t, y)$.
        
        Args:
            func: Function computing $dy/dt$ given $(t, y)$
            y0: Initial state (batch_size, dim)
            t_span: Time interval $(t_{\text{start}}, t_{\text{end}})$
            **kwargs: Additional solving parameters
            
        Returns:
            y_final: Final state at t_end (batch_size, dim)
        """
        device = y0.device
        t_eval = torch.tensor(list(t_span), device=device)
        
        # Build options
        options = {'dtype': torch.float32}
        options.update(self.method_options)
        options.update(kwargs)
        
        # Solve
        trajectory = self.odeint_fn(
            func,
            y0,
            t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options
        )
        
        return trajectory[-1]  # Return final state
    
    def solve_trajectory(
        self,
        func: Callable[[Tensor, Tensor], Tensor],
        y0: Tensor,
        t_eval: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Solve and return trajectory at specified time points.
        
        Args:
            func: Function computing dy/dt
            y0: Initial state (batch_size, dim)
            t_eval: Time points for evaluation (n_steps,)
            **kwargs: Additional parameters
            
        Returns:
            trajectory: States at each time point (n_steps, batch_size, dim)
        """
        # Build options
        options = {'dtype': torch.float32}
        if self.method in self.FIXED_STEP_SOLVERS:
            n_steps = len(t_eval) - 1
            dt = (t_eval[-1] - t_eval[0]) / n_steps
            options['step_size'] = dt.item()
        
        options.update(self.method_options)
        options.update(kwargs)
        
        # Solve
        trajectory = self.odeint_fn(
            func,
            y0,
            t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options
        )
        
        return trajectory
    
    @torch.no_grad()
    def sample(
        self,
        condition: Tensor,
        x_init: Optional[Tensor] = None,
        t_span: Tuple[float, float] = (0.0, 1.0),
        return_trajectory: bool = False,
        n_steps: Optional[int] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Sample from flow matching model by solving ODE.
        
        Args:
            condition: Conditioning tensor (batch_size, dim) or (batch_size, H, W)
            x_init: Initial noise (batch_size, dim). If None, sample from N(0, I)
            t_span: Time interval (t_start, t_end), typically (0.0, 1.0)
            return_trajectory: If True, return full trajectory
            n_steps: Number of evaluation points (for fixed-step solvers or trajectory)
                     If None, adaptive solvers choose steps automatically
        
        Returns:
            samples: Final samples at t_end (batch_size, dim) or (batch_size, H, W)
            trajectory: (optional) Full trajectory if return_trajectory=True
        """
        self.model.eval()
        
        # Store original shapes for reshaping
        condition_original_shape = condition.shape
        if condition.dim() > 2:
            condition = condition.flatten(start_dim=1)
        
        condition = condition.float()
        batch_size = condition.shape[0]
        device = condition.device
        
        # Initialize from noise and store its shape
        if x_init is None:
            # Default: same spatial shape as condition but need to infer solution channels
            # For now, assume solution has 1 channel (common for PDEs)
            if len(condition_original_shape) > 2:
                # Spatial input: (B, C, H, W) -> solution is (B, 1, H, W)
                solution_shape = (batch_size, 1, *condition_original_shape[2:])
                x_init = torch.randn(solution_shape, device=device)
                x_init_original_shape = solution_shape
            else:
                # Flattened input: same dim as condition
                x_init = torch.randn_like(condition)
                x_init_original_shape = condition_original_shape
        else:
            x_init_original_shape = x_init.shape
        
        # Flatten x_init for ODE integration
        if x_init.dim() > 2:
            x_init = x_init.flatten(start_dim=1).float()
        else:
            x_init = x_init.float()
        
        # Create time span
        t_start, t_end = t_span
        if return_trajectory or (self.method in self.FIXED_STEP_SOLVERS and n_steps is not None):
            # Fixed evaluation points
            if n_steps is None:
                n_steps = 50  # Default
            t_eval = torch.linspace(t_start, t_end, n_steps + 1, device=device)
        else:
            # For adaptive solvers, just specify endpoints
            t_eval = torch.tensor([t_start, t_end], device=device)
        
        # Create velocity field
        velocity_field = VelocityField(self.model, condition)
        
        # Solve ODE
        trajectory = self.solve_trajectory(
            velocity_field,
            x_init,
            t_eval
        )
        
        # trajectory shape: (n_steps+1, batch_size, dim) or (2, batch_size, dim)
        samples = trajectory[-1]  # Final state
        
        # Reshape to original solution shape (not condition shape!)
        if len(x_init_original_shape) > 2:
            samples = samples.view(x_init_original_shape)
            if return_trajectory:
                trajectory = trajectory.view(-1, *x_init_original_shape)
        
        if return_trajectory:
            return samples, trajectory
        
        return samples
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get information about current solver configuration."""
        info = super().get_solver_info()
        info['adjoint'] = self.adjoint
        return info


@torch.no_grad()
def sample_with_ode_solver(
    model: nn.Module,
    condition: Tensor,
    solver: str = 'dopri5',
    rtol: float = 1e-5,
    atol: float = 1e-7,
    n_steps: Optional[int] = None,
    device: str = 'cuda',
    return_trajectory: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Convenience function for sampling with ODE solver.
    
    This is a simpler interface to ODEFlowSolver for one-off sampling.
    
    Args:
        model: Flow matching model
        condition: Conditioning tensor
        solver: ODE solver method (see ODEFlowSolver for options)
        rtol: Relative tolerance
        atol: Absolute tolerance
        n_steps: Number of steps (for fixed-step solvers)
        device: Device for computation
        return_trajectory: If True, return full trajectory
    
    Returns:
        samples: Final samples
        trajectory: (optional) Full trajectory if return_trajectory=True
    
    Example:
        >>> samples = sample_with_ode_solver(
        ...     model=trained_model,
        ...     condition=f,
        ...     solver='dopri5',
        ...     device='cuda'
        ... )
    """
    condition = condition.to(device)
    
    solver_instance = ODEFlowSolver(
        model=model,
        solver=solver,
        rtol=rtol,
        atol=atol
    )
    
    return solver_instance.sample(
        condition=condition,
        return_trajectory=return_trajectory,
        n_steps=n_steps
    )


def compare_solvers(
    model: nn.Module,
    condition: Tensor,
    ground_truth: Optional[Tensor] = None,
    solvers: Optional[List[str]] = None,
    device: str = 'cuda',
    n_steps: int = 50
) -> dict:
    """
    Compare different ODE solvers on the same input.
    
    Args:
        model: Flow matching model
        condition: Conditioning tensor
        ground_truth: Optional ground truth for error computation
        solvers: List of solver names to compare. If None, uses default set
        device: Device for computation
        n_steps: Number of steps for fixed-step solvers
    
    Returns:
        Dictionary with results for each solver including:
        - samples: Generated samples
        - time: Computation time
        - error: L2 error vs ground truth (if provided)
    
    Example:
        >>> results = compare_solvers(
        ...     model=trained_model,
        ...     condition=f,
        ...     ground_truth=u_true,
        ...     solvers=['euler', 'rk4', 'dopri5']
        ... )
        >>> for solver, info in results.items():
        ...     print(f"{solver}: error={info['error']:.6f}, time={info['time']:.3f}s")
    """
    import time
    
    if solvers is None:
        solvers = ['euler', 'midpoint', 'rk4', 'dopri5']
    
    model.eval()
    condition = condition.to(device)
    if ground_truth is not None:
        ground_truth = ground_truth.to(device)
    
    results = {}
    
    for solver_name in solvers:
        # Create solver
        ode_solver = ODEFlowSolver(
            model=model,
            solver=solver_name,
            rtol=1e-5,
            atol=1e-7
        )
        
        # Time the sampling
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        samples = ode_solver.sample(
            condition=condition,
            n_steps=n_steps if solver_name in ODEFlowSolver.FIXED_STEP_SOLVERS else None
        )
        
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start_time
        
        # Compute error if ground truth provided
        error = None
        if ground_truth is not None:
            samples_flat = samples.flatten(start_dim=1)
            gt_flat = ground_truth.flatten(start_dim=1)
            error = (samples_flat - gt_flat).norm(dim=1).mean().item()
        
        results[solver_name] = {
            'samples': samples.cpu(),
            'time': elapsed,
            'error': error,
            'solver_info': ode_solver.get_solver_info()
        }
    
    return results
