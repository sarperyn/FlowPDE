"""
APEBench Dataset Provider
==========================

High-level factory API for generating PDE datasets from APEBench.
Provides a unified interface for creating datasets with caching support.
"""

from typing import Dict, Any, Optional, Type, Literal, Union
from pathlib import Path
import warnings

from .base import BaseAPEBenchScenario, APEBenchDataset
from .cache import CacheManager
from .scenarios import (
    BurgersScenario,
    BurgersSingleChannelScenario,
    PoissonScenario,
)


class APEBenchProvider:
    """
    Factory class for creating APEBench-based datasets.
    
    Provides a unified, user-friendly API for generating PDE datasets
    using APEBench's JAX-based spectral solvers.
    
    Features:
    - Simple string-based PDE selection
    - Automatic scenario configuration
    - Opt-in disk caching (recommended)
    - PyTorch Dataset compatibility
    
    Supported PDEs:
    - 'burgers_1d': 1D Burgers equation
    - 'burgers_2d': 2D Burgers equation
    - 'poisson_1d': 1D Poisson equation
    - 'poisson_2d': 2D Poisson equation
    
    Example:
        >>> from flowpde.datasets.apebench import APEBenchProvider
        >>> 
        >>> # Create a Burgers 1D dataset
        >>> dataset = APEBenchProvider.create(
        ...     pde='burgers_1d',
        ...     problem='forward',
        ...     num_points=160,
        ...     viscosity=0.0003,
        ...     num_train_samples=500,
        ...     cache=True,
        ... )
        >>> 
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    # Registry of available PDE scenarios
    _registry: Dict[str, Type[BaseAPEBenchScenario]] = {
        'burgers_1d': BurgersScenario,
        'burgers_2d': BurgersScenario,
        'burgers_sc_1d': BurgersSingleChannelScenario,
        'burgers_sc_2d': BurgersSingleChannelScenario,
        'poisson_1d': PoissonScenario,
        'poisson_2d': PoissonScenario,
        'poisson_3d': PoissonScenario,
    }
    
    # Default parameters for each PDE type
    _defaults: Dict[str, Dict[str, Any]] = {
        'burgers_1d': {
            'num_spatial_dims': 1,
            'num_points': 160,
            'viscosity': 0.0003,
            'train_temporal_horizon': 50,
        },
        'burgers_2d': {
            'num_spatial_dims': 2,
            'num_points': 64,
            'viscosity': 0.0003,
            'train_temporal_horizon': 50,
        },
        'burgers_sc_1d': {
            'num_spatial_dims': 1,
            'num_points': 160,
            'diffusion_alpha': 0.00003,
            'convection_beta': -0.0125,
        },
        'burgers_sc_2d': {
            'num_spatial_dims': 2,
            'num_points': 64,
            'diffusion_alpha': 0.00003,
            'convection_beta': -0.0125,
        },
        'poisson_1d': {
            'num_spatial_dims': 1,
            'num_points': 160,
            'domain_extent': 10.0,
        },
        'poisson_2d': {
            'num_spatial_dims': 2,
            'num_points': 64,
            'domain_extent': 10.0,
        },
        'poisson_3d': {
            'num_spatial_dims': 3,
            'num_points': 32,
            'domain_extent': 10.0,
        },
    }
    
    @classmethod
    def register(cls, name: str, scenario_class: Type[BaseAPEBenchScenario]):
        """
        Register a new PDE scenario.
        
        Use this to add custom PDE scenarios to the provider.
        
        Args:
            name: String identifier for the PDE (e.g., 'navier_stokes_2d')
            scenario_class: Subclass of BaseAPEBenchScenario
            
        Example:
            >>> class NavierStokesScenario(BaseAPEBenchScenario):
            ...     ...
            >>> APEBenchProvider.register('navier_stokes_2d', NavierStokesScenario)
        """
        cls._registry[name] = scenario_class
    
    @classmethod
    def list_available(cls) -> list:
        """
        List all available PDE scenarios.
        
        Returns:
            List of registered PDE names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_defaults(cls, pde: str) -> Dict[str, Any]:
        """
        Get default parameters for a PDE.
        
        Args:
            pde: PDE identifier string
            
        Returns:
            Dictionary of default parameters
        """
        if pde not in cls._defaults:
            return {}
        return cls._defaults[pde].copy()
    
    @classmethod
    def create(
        cls,
        pde: str,
        problem: Literal['forward', 'inverse'] = 'forward',
        mode: Literal['train', 'test'] = 'train',
        cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        force_regenerate: bool = False,
        torch_device: str = 'cpu',
        jax_device: Literal['cpu', 'gpu', 'auto'] = 'cpu',
        **kwargs,
    ) -> APEBenchDataset:
        """
        Create a dataset for the specified PDE.
        
        Args:
            pde: PDE identifier (e.g., 'burgers_1d', 'poisson_2d')
            problem: Problem type - 'forward' or 'inverse'
            mode: Data split - 'train' or 'test'
            cache: Whether to cache generated data (recommended)
            cache_dir: Cache directory (default: .cache/apebench/)
            force_regenerate: If True, regenerate even if cache exists
            torch_device: Target PyTorch device ('cpu', 'cuda', etc.)
            jax_device: JAX computation device ('cpu', 'gpu', 'auto')
            **kwargs: Additional scenario-specific parameters
            
        Returns:
            APEBenchDataset instance ready for DataLoader
            
        Raises:
            ValueError: If PDE is not registered
            ImportError: If APEBench/JAX is not installed
            
        Example:
            >>> # Basic usage
            >>> dataset = APEBenchProvider.create('burgers_1d')
            >>> 
            >>> # With custom parameters
            >>> dataset = APEBenchProvider.create(
            ...     pde='burgers_1d',
            ...     viscosity=0.001,
            ...     num_points=256,
            ...     num_train_samples=1000,
            ...     cache=True,
            ... )
        """
        # Validate PDE selection
        if pde not in cls._registry:
            available = ', '.join(cls.list_available())
            raise ValueError(
                f"Unknown PDE: '{pde}'. Available options: {available}"
            )
        
        # Get scenario class and defaults
        scenario_class = cls._registry[pde]
        defaults = cls.get_defaults(pde)
        
        # Merge defaults with provided kwargs
        params = {**defaults, **kwargs}
        params['torch_device'] = torch_device
        params['jax_device'] = jax_device
        
        if cache_dir is not None:
            params['cache_dir'] = cache_dir
        
        # Create scenario
        scenario = scenario_class(**params)
        
        # Generate data
        data = scenario.generate(
            mode=mode,
            cache=cache,
            force_regenerate=force_regenerate,
        )
        
        # Return as dataset
        return APEBenchDataset(data, problem=problem)
    
    @classmethod
    def create_train_test(
        cls,
        pde: str,
        problem: Literal['forward', 'inverse'] = 'forward',
        cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> tuple:
        """
        Create both train and test datasets in one call.
        
        Args:
            pde: PDE identifier
            problem: Problem type
            cache: Whether to cache data
            cache_dir: Cache directory
            **kwargs: Scenario parameters
            
        Returns:
            Tuple of (train_dataset, test_dataset)
            
        Example:
            >>> train_ds, test_ds = APEBenchProvider.create_train_test(
            ...     pde='burgers_1d',
            ...     cache=True,
            ... )
        """
        train_dataset = cls.create(
            pde=pde,
            problem=problem,
            mode='train',
            cache=cache,
            cache_dir=cache_dir,
            **kwargs,
        )
        
        test_dataset = cls.create(
            pde=pde,
            problem=problem,
            mode='test',
            cache=cache,
            cache_dir=cache_dir,
            **kwargs,
        )
        
        return train_dataset, test_dataset
    
    @classmethod
    def get_scenario(
        cls,
        pde: str,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> BaseAPEBenchScenario:
        """
        Get a scenario instance without generating data.
        
        Useful for advanced use cases where you need direct
        access to the scenario object.
        
        Args:
            pde: PDE identifier
            cache_dir: Cache directory
            **kwargs: Scenario parameters
            
        Returns:
            Scenario instance
        """
        if pde not in cls._registry:
            available = ', '.join(cls.list_available())
            raise ValueError(
                f"Unknown PDE: '{pde}'. Available options: {available}"
            )
        
        scenario_class = cls._registry[pde]
        defaults = cls.get_defaults(pde)
        params = {**defaults, **kwargs}
        
        if cache_dir is not None:
            params['cache_dir'] = cache_dir
        
        return scenario_class(**params)
    
    @classmethod
    def clear_cache(cls, pde: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Clear cached datasets.
        
        Args:
            pde: If provided, only clear cache for this PDE
            cache_dir: Cache directory (default: .cache/apebench/)
            
        Returns:
            Number of cache directories cleared
        """
        cache = CacheManager(cache_dir)
        return cache.clear(pde)
    
    @classmethod
    def list_cached(cls, cache_dir: Optional[str] = None) -> list:
        """
        List all cached datasets.
        
        Args:
            cache_dir: Cache directory
            
        Returns:
            List of cache info dictionaries
        """
        cache = CacheManager(cache_dir)
        return cache.list_cached()


# Convenience alias
create_dataset = APEBenchProvider.create
