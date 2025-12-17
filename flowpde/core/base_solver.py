"""
Base Solver Abstract Class

Defines the interface for all ODE/SDE solvers in FlowPDE.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Union, Tuple, Dict, Any
import torch
from torch import nn, Tensor


class BaseSolver(ABC):
    """
    Abstract base class for ODE/SDE solvers.
    
    This defines the common interface for numerical integration methods
    used to solve differential equations in normalizing flows.
    
    Solvers can be:
    - Fixed-step (e.g., Euler, RK4)
    - Adaptive step-size (e.g., Dopri5, Dopri8)
    - Stochastic (SDE solvers) (NOT IMPLEMENTED YET)
    """
    
    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        method_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize solver.
        
        Args:
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
            method_options: Additional solver-specific options
            **kwargs: Additional parameters
        """
        self.rtol = rtol
        self.atol = atol
        self.method_options = method_options or {}
        self._extra_kwargs = kwargs
    
    @abstractmethod
    def solve(
        self,
        func: Callable[[Tensor, Tensor], Tensor],
        y0: Tensor,
        t_span: Tuple[float, float],
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """
        Solve the differential equation dy/dt = func(t, y).
        
        Args:
            func: Function computing dy/dt given (t, y)
                  Signature: func(t: Tensor, y: Tensor) -> Tensor
            y0: Initial state (batch_size, dim)
            t_span: Time interval (t_start, t_end)
            **kwargs: Additional solving parameters
            
        Returns:
            y_final: Final state at t_end (batch_size, dim)
            info (optional): Dictionary with solving statistics
        """
        raise NotImplementedError
    
    def solve_trajectory(
        self,
        func: Callable[[Tensor, Tensor], Tensor],
        y0: Tensor,
        t_eval: Tensor,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """
        Solve and return trajectory at specified time points.
        
        Args:
            func: Function computing dy/dt
            y0: Initial state (batch_size, dim)
            t_eval: Time points for evaluation (n_steps,)
            **kwargs: Additional parameters
            
        Returns:
            trajectory: States at each time point (n_steps, batch_size, dim)
            info (optional): Dictionary with solving statistics
        """
        # Default implementation: solve multiple intervals
        # Subclasses can override for more efficient implementations
        trajectory = [y0]
        y_current = y0
        
        for i in range(len(t_eval) - 1):
            t_span = (t_eval[i].item(), t_eval[i+1].item())
            result = self.solve(func, y_current, t_span, **kwargs)
            if isinstance(result, tuple):
                y_current, _ = result
            else:
                y_current = result
            trajectory.append(y_current)
        
        return torch.stack(trajectory, dim=0)
    
    @abstractmethod
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get information about the solver configuration.
        
        Returns:
            Dictionary with solver properties
        """
        raise NotImplementedError
    
    def set_tolerance(self, rtol: float, atol: float):
        """Update solver tolerances."""
        self.rtol = rtol
        self.atol = atol
    
    @property
    @abstractmethod
    def is_adaptive(self) -> bool:
        """Whether this is an adaptive step-size solver."""
        raise NotImplementedError
    
    @property
    def supports_adjoint(self) -> bool:
        """Whether this solver supports adjoint method for backprop."""
        return False
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"rtol={self.rtol}, atol={self.atol}, "
            f"adaptive={self.is_adaptive})"
        )


class ODESolver(BaseSolver):
    """
    Base class specifically for ODE solvers.
    
    Ordinary Differential Equation solvers for deterministic flows.
    """
    
    def __init__(
        self,
        method: str,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        **kwargs
    ):
        """
        Initialize ODE solver.
        
        Args:
            method: Integration method name
            rtol: Relative tolerance
            atol: Absolute tolerance
            **kwargs: Additional parameters
        """
        super().__init__(rtol=rtol, atol=atol, **kwargs)
        self.method = method
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get ODE solver information."""
        return {
            'type': 'ODE',
            'method': self.method,
            'rtol': self.rtol,
            'atol': self.atol,
            'adaptive': self.is_adaptive,
            'adjoint_support': self.supports_adjoint,
        }


class SDESolver(BaseSolver):
    """
    Base class specifically for SDE solvers.
    
    Stochastic Differential Equation solvers for stochastic flows.
    """
    
    def __init__(
        self,
        method: str,
        noise_type: str = 'diagonal',
        rtol: float = 1e-5,
        atol: float = 1e-7,
        **kwargs
    ):
        """
        Initialize SDE solver.
        
        Args:
            method: Integration method name
            noise_type: Type of noise ('diagonal', 'general', 'scalar')
            rtol: Relative tolerance
            atol: Absolute tolerance
            **kwargs: Additional parameters
        """
        super().__init__(rtol=rtol, atol=atol, **kwargs)
        self.method = method
        self.noise_type = noise_type
    
    @abstractmethod
    def solve(
        self,
        drift: Callable[[Tensor, Tensor], Tensor],
        diffusion: Callable[[Tensor, Tensor], Tensor],
        y0: Tensor,
        t_span: Tuple[float, float],
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """
        Solve SDE: dy = drift(t, y)dt + diffusion(t, y)dW.
        
        Args:
            drift: Drift function (deterministic part)
            diffusion: Diffusion function (stochastic part)
            y0: Initial state
            t_span: Time interval
            **kwargs: Additional parameters
            
        Returns:
            y_final: Final state
            info (optional): Solver statistics
        """
        raise NotImplementedError
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get SDE solver information."""
        return {
            'type': 'SDE',
            'method': self.method,
            'noise_type': self.noise_type,
            'rtol': self.rtol,
            'atol': self.atol,
            'adaptive': self.is_adaptive,
        }
