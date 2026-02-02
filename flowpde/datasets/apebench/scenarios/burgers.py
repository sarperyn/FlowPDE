"""
Burgers Equation Scenario Wrapper
==================================

Wraps APEBench's Burgers equation solvers for FlowPDE integration.

Supports:
- 1D Burgers equation with configurable viscosity
- 2D Burgers equation (single-channel)
- Both physical and difficulty-based parameterizations

The Burgers equation:
    ∂u/∂t + u · ∇u = ν ∇²u

where ν is the viscosity (diffusion coefficient).
"""

import torch
from typing import Dict, Any, Optional, Literal, Union
from pathlib import Path
from dataclasses import dataclass, field

from ..base import BaseAPEBenchScenario, ScenarioConfig, APEBenchDataset
from ..converters import jax_to_torch


@dataclass
class BurgersConfig(ScenarioConfig):
    """
    Configuration specific to Burgers equation scenarios.
    
    Inherits all base config options and adds Burgers-specific parameters.
    """
    # Burgers-specific parameters
    viscosity: float = 0.0003  # Diffusion coefficient (ν)
    convection_coef: float = -0.125  # Convection coefficient
    
    # Initial condition config (APEBench format)
    ic_config: str = "fourier;5;true;true"
    
    # Warmup steps (for letting transients settle)
    num_warmup_steps: int = 0
    
    def get_cache_params(self) -> Dict[str, Any]:
        """Include viscosity in cache key."""
        params = super().get_cache_params()
        params['viscosity'] = self.viscosity
        return params


class BurgersScenario(BaseAPEBenchScenario):
    """
    Burgers equation scenario wrapper for APEBench.
    
    Generates trajectories of the 1D or 2D Burgers equation using
    APEBench's spectral solvers, converted to FlowPDE format.
    
    Output format:
        {
            'initial': (N, 1, *spatial),     # Initial conditions
            'final': (N, 1, *spatial),       # Final states
            'trajectory': (N, T, 1, *spatial), # Full trajectories
            'time': (T,),                    # Time points
            'viscosity': float,              # Viscosity value
            'stats': {...},                  # Normalization statistics
            'config': {...},                 # Generation config
        }
    
    Example:
        >>> scenario = BurgersScenario(
        ...     num_points=160,
        ...     viscosity=0.0003,
        ...     num_train_samples=500,
        ... )
        >>> data = scenario.generate(mode='train', cache=True)
        >>> print(data['initial'].shape)  # (500, 1, 160)
    """
    
    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 160,
        domain_extent: float = 1.0,
        dt: float = 0.1,
        viscosity: float = 0.0003,
        num_train_samples: int = 50,
        num_test_samples: int = 30,
        train_temporal_horizon: int = 50,
        test_temporal_horizon: int = 200,
        train_seed: int = 0,
        test_seed: int = 773,
        jax_device: Literal['cpu', 'gpu', 'auto'] = 'cpu',
        torch_device: str = 'cpu',
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize Burgers scenario.
        
        Args:
            num_spatial_dims: Spatial dimensions (1 or 2)
            num_points: Grid resolution per dimension
            domain_extent: Physical domain size
            dt: Time step size
            viscosity: Diffusion coefficient (ν)
            num_train_samples: Number of training trajectories
            num_test_samples: Number of test trajectories
            train_temporal_horizon: Training trajectory length
            test_temporal_horizon: Test trajectory length
            train_seed: Random seed for training data
            test_seed: Random seed for test data
            jax_device: JAX computation device ('cpu', 'gpu', 'auto')
            torch_device: Target PyTorch device
            cache_dir: Optional cache directory override
        """
        # Create config
        config = BurgersConfig(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            domain_extent=domain_extent,
            dt=dt,
            viscosity=viscosity,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            train_temporal_horizon=train_temporal_horizon,
            test_temporal_horizon=test_temporal_horizon,
            train_seed=train_seed,
            test_seed=test_seed,
            jax_device=jax_device,
            torch_device=torch_device,
        )
        
        # Store for easy access
        self.viscosity = viscosity
        self.num_spatial_dims = num_spatial_dims
        
        super().__init__(config=config, cache_dir=cache_dir, **kwargs)
    
    def _create_apebench_scenario(self):
        """Create the APEBench Burgers scenario."""
        import apebench
        
        # Use physical Burgers scenario
        scenario = apebench.scenarios.physical.Burgers(
            num_spatial_dims=self.config.num_spatial_dims,
            num_points=self.config.num_points,
            domain_extent=self.config.domain_extent,
            dt=self.config.dt,
            diffusion_coef=self.config.viscosity,
            num_train_samples=self.config.num_train_samples,
            num_test_samples=self.config.num_test_samples,
            train_temporal_horizon=self.config.train_temporal_horizon,
            test_temporal_horizon=self.config.test_temporal_horizon,
            train_seed=self.config.train_seed,
            test_seed=self.config.test_seed,
        )
        
        return scenario
    
    def _convert_to_flowpde_format(
        self,
        jax_data,
        mode: Literal['train', 'test'],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert APEBench trajectory data to FlowPDE format.
        
        APEBench format: (N, T+1, C, *spatial)
        FlowPDE format: separate initial, final, trajectory tensors
        
        Args:
            jax_data: JAX array from APEBench get_train_data/get_test_data
            mode: 'train' or 'test' split
            
        Returns:
            Dictionary with PyTorch tensors
        """
        # Convert full trajectory to PyTorch
        trajectory = jax_to_torch(
            jax_data,
            device=self.config.torch_device,
            dtype=torch.float32,
        )
        
        # Extract initial and final states
        # APEBench includes initial condition at index 0
        initial = trajectory[:, 0]    # (N, C, *spatial)
        final = trajectory[:, -1]     # (N, C, *spatial)
        
        # Full trajectory excluding initial (for consistency with some uses)
        traj_no_init = trajectory[:, 1:]  # (N, T, C, *spatial)
        
        # Create time array
        temporal_horizon = (
            self.config.train_temporal_horizon 
            if mode == 'train' 
            else self.config.test_temporal_horizon
        )
        time = torch.linspace(
            self.config.dt,
            self.config.dt * temporal_horizon,
            temporal_horizon,
            dtype=torch.float32,
        )
        
        return {
            'initial': initial,
            'final': final,
            'trajectory': traj_no_init,
            'time': time,
            'viscosity': self.viscosity,
        }
    
    def get_pde_name(self) -> str:
        """Get PDE identifier string."""
        return f"burgers_{self.num_spatial_dims}d"
    
    def get_cache_params(self) -> Dict[str, Any]:
        """Get parameters for cache key generation."""
        params = self.config.get_cache_params()
        params['viscosity'] = self.viscosity
        return params


class BurgersSingleChannelScenario(BaseAPEBenchScenario):
    """
    Single-channel Burgers equation (alternative formulation).
    
    Uses the BurgersSingleChannel scenario from APEBench which
    has a slightly different parameterization using normalized coefficients.
    """
    
    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 160,
        diffusion_alpha: float = 0.00003,
        convection_beta: float = -0.0125,
        num_train_samples: int = 50,
        num_test_samples: int = 30,
        train_temporal_horizon: int = 50,
        test_temporal_horizon: int = 200,
        train_seed: int = 0,
        test_seed: int = 773,
        jax_device: Literal['cpu', 'gpu', 'auto'] = 'cpu',
        torch_device: str = 'cpu',
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize single-channel Burgers scenario.
        
        Uses normalized coefficients (alpha, beta) instead of physical
        parameters (viscosity).
        
        Args:
            diffusion_alpha: Normalized diffusion coefficient
            convection_beta: Normalized convection coefficient
            (other args same as BurgersScenario)
        """
        config = ScenarioConfig(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            train_temporal_horizon=train_temporal_horizon,
            test_temporal_horizon=test_temporal_horizon,
            train_seed=train_seed,
            test_seed=test_seed,
            jax_device=jax_device,
            torch_device=torch_device,
        )
        
        self.diffusion_alpha = diffusion_alpha
        self.convection_beta = convection_beta
        self.num_spatial_dims = num_spatial_dims
        
        super().__init__(config=config, cache_dir=cache_dir, **kwargs)
    
    def _create_apebench_scenario(self):
        """Create the APEBench normalized Burgers scenario."""
        import apebench
        
        scenario = apebench.scenarios.normalized.BurgersSingleChannel(
            num_spatial_dims=self.config.num_spatial_dims,
            num_points=self.config.num_points,
            diffusion_alpha=self.diffusion_alpha,
            convection_sc_beta=self.convection_beta,
            num_train_samples=self.config.num_train_samples,
            num_test_samples=self.config.num_test_samples,
            train_temporal_horizon=self.config.train_temporal_horizon,
            test_temporal_horizon=self.config.test_temporal_horizon,
            train_seed=self.config.train_seed,
            test_seed=self.config.test_seed,
        )
        
        return scenario
    
    def _convert_to_flowpde_format(
        self,
        jax_data,
        mode: Literal['train', 'test'],
    ) -> Dict[str, torch.Tensor]:
        """Convert to FlowPDE format (same as standard Burgers)."""
        trajectory = jax_to_torch(
            jax_data,
            device=self.config.torch_device,
            dtype=torch.float32,
        )
        
        initial = trajectory[:, 0]
        final = trajectory[:, -1]
        traj_no_init = trajectory[:, 1:]
        
        temporal_horizon = (
            self.config.train_temporal_horizon 
            if mode == 'train' 
            else self.config.test_temporal_horizon
        )
        time = torch.linspace(
            0.0, 1.0, temporal_horizon, dtype=torch.float32
        )
        
        return {
            'initial': initial,
            'final': final,
            'trajectory': traj_no_init,
            'time': time,
            'diffusion_alpha': self.diffusion_alpha,
            'convection_beta': self.convection_beta,
        }
    
    def get_pde_name(self) -> str:
        return f"burgers_sc_{self.num_spatial_dims}d"
    
    def get_cache_params(self) -> Dict[str, Any]:
        params = self.config.get_cache_params()
        params['diffusion_alpha'] = self.diffusion_alpha
        params['convection_beta'] = self.convection_beta
        return params
