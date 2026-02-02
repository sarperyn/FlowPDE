"""
Poisson Equation Scenario Wrapper
==================================

Wraps APEBench's Poisson equation solver for FlowPDE integration.

The Poisson equation:
    -∇²u = f

where f is the source term and u is the solution.

Note: APEBench's Poisson solver is for periodic domains and uses spectral
methods. This is suitable for learning mappings f → u or u → f.
"""

import torch
from typing import Dict, Any, Optional, Literal, Union
from pathlib import Path
from dataclasses import dataclass

from ..base import BaseAPEBenchScenario, ScenarioConfig, APEBenchDataset
from ..converters import jax_to_torch


@dataclass
class PoissonConfig(ScenarioConfig):
    """
    Configuration specific to Poisson equation scenarios.
    """
    # Poisson-specific parameters
    domain_extent: float = 10.0
    order: int = 2  # Order of the finite difference stencil
    
    # Override temporal settings (Poisson is static)
    train_temporal_horizon: int = 1
    test_temporal_horizon: int = 1
    num_warmup_steps: int = 0
    
    def get_cache_params(self) -> Dict[str, Any]:
        """Include order in cache key."""
        params = super().get_cache_params()
        params['order'] = self.order
        return params


class PoissonScenario(BaseAPEBenchScenario):
    """
    Poisson equation scenario wrapper for APEBench.
    
    Generates source-solution pairs for the Poisson equation using
    APEBench's spectral solver, converted to FlowPDE format.
    
    Note: The Poisson equation is static (no time evolution), so
    APEBench treats it as a single-step "stepper" that maps
    source terms to solutions.
    
    Output format:
        {
            'source': (N, 1, *spatial),      # Source terms (f)
            'solution': (N, 1, *spatial),    # Solutions (u)
            'stats': {...},                   # Normalization statistics
            'config': {...},                  # Generation config
        }
    
    Example:
        >>> scenario = PoissonScenario(
        ...     num_spatial_dims=2,
        ...     num_points=64,
        ...     num_train_samples=500,
        ... )
        >>> data = scenario.generate(mode='train', cache=True)
        >>> print(data['source'].shape)  # (500, 1, 64, 64)
    """
    
    def __init__(
        self,
        num_spatial_dims: int = 2,
        num_points: int = 64,
        domain_extent: float = 10.0,
        order: int = 2,
        num_train_samples: int = 50,
        num_test_samples: int = 30,
        train_seed: int = 0,
        test_seed: int = 773,
        jax_device: Literal['cpu', 'gpu', 'auto'] = 'cpu',
        torch_device: str = 'cpu',
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize Poisson scenario.
        
        Args:
            num_spatial_dims: Spatial dimensions (1, 2, or 3)
            num_points: Grid resolution per dimension
            domain_extent: Physical domain size
            order: Order of finite difference stencil
            num_train_samples: Number of training pairs
            num_test_samples: Number of test pairs
            train_seed: Random seed for training data
            test_seed: Random seed for test data
            jax_device: JAX computation device
            torch_device: Target PyTorch device
            cache_dir: Optional cache directory override
        """
        config = PoissonConfig(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            domain_extent=domain_extent,
            order=order,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            train_seed=train_seed,
            test_seed=test_seed,
            jax_device=jax_device,
            torch_device=torch_device,
            # Poisson is static
            train_temporal_horizon=1,
            test_temporal_horizon=1,
        )
        
        self.num_spatial_dims = num_spatial_dims
        self.order = order
        
        super().__init__(config=config, cache_dir=cache_dir, **kwargs)
    
    def _create_apebench_scenario(self):
        """Create the APEBench Poisson scenario."""
        import apebench
        
        scenario = apebench.scenarios.physical.Poisson(
            num_spatial_dims=self.config.num_spatial_dims,
            num_points=self.config.num_points,
            domain_extent=self.config.domain_extent,
            order=self.order,
            num_train_samples=self.config.num_train_samples,
            num_test_samples=self.config.num_test_samples,
            train_seed=self.config.train_seed,
            test_seed=self.config.test_seed,
            # Poisson requires these to be 1
            train_temporal_horizon=1,
            test_temporal_horizon=1,
            num_warmup_steps=0,
        )
        
        return scenario
    
    def _convert_to_flowpde_format(
        self,
        jax_data,
        mode: Literal['train', 'test'],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert APEBench Poisson data to FlowPDE format.
        
        APEBench Poisson format: (N, 2, C, *spatial)
        where index 0 is the source (input) and index 1 is the solution (output)
        
        Args:
            jax_data: JAX array from APEBench
            mode: 'train' or 'test' split
            
        Returns:
            Dictionary with 'source' and 'solution' tensors
        """
        # Convert to PyTorch
        data = jax_to_torch(
            jax_data,
            device=self.config.torch_device,
            dtype=torch.float32,
        )
        
        # APEBench Poisson returns trajectory with 2 time steps:
        # - t=0: source term (input to "stepper")
        # - t=1: solution (output of "stepper")
        source = data[:, 0]    # (N, C, *spatial) - source/RHS
        solution = data[:, 1]  # (N, C, *spatial) - solution
        
        return {
            'source': source,
            'solution': solution,
            'domain_extent': self.config.domain_extent,
            'order': self.order,
        }
    
    def get_pde_name(self) -> str:
        """Get PDE identifier string."""
        return f"poisson_{self.num_spatial_dims}d"
    
    def get_cache_params(self) -> Dict[str, Any]:
        """Get parameters for cache key generation."""
        params = self.config.get_cache_params()
        params['order'] = self.order
        params['domain_extent'] = self.config.domain_extent
        return params


class PoissonDataset(APEBenchDataset):
    """
    Specialized dataset for Poisson equation data.
    
    Provides convenient access to source and solution fields.
    For use with FlowPDE's flow training pipeline.
    """
    
    def __init__(
        self,
        data: Dict[str, Any],
        problem: Literal['forward', 'inverse'] = 'forward',
        return_dict: bool = True,
    ):
        """
        Initialize Poisson dataset.
        
        Args:
            data: Dictionary from PoissonScenario.generate()
            problem: 'forward' (source→solution) or 'inverse' (solution→source)
            return_dict: If True, return dict; if False, return tuple
        """
        super().__init__(data, problem=problem, return_dict=return_dict)
    
    @property
    def source(self) -> torch.Tensor:
        """Get source term tensor."""
        return self.data['source']
    
    @property
    def solution(self) -> torch.Tensor:
        """Get solution tensor."""
        return self.data['solution']
    
    @property
    def domain_extent(self) -> float:
        """Get domain extent."""
        return self.data.get('domain_extent', 10.0)
