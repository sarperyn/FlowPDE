"""
Base Abstract Class for APEBench Scenario Wrappers
===================================================

Defines the interface that all PDE scenario wrappers must implement,
providing common functionality for data generation, conversion, and caching.
"""

import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union, Tuple
from dataclasses import dataclass, field, asdict

from .converters import (
    jax_to_torch,
    check_apebench_available,
    compute_normalization_stats,
)
from .cache import CacheManager


@dataclass
class ScenarioConfig:
    """
    Configuration for a PDE scenario.
    
    This dataclass holds all parameters needed to configure an APEBench
    scenario, providing a clean interface for parameter passing.
    """
    # Spatial discretization
    num_spatial_dims: int = 1
    num_points: int = 160
    domain_extent: float = 1.0
    
    # Temporal discretization
    dt: float = 0.1
    train_temporal_horizon: int = 50
    test_temporal_horizon: int = 200
    
    # Sample counts
    num_train_samples: int = 50
    num_test_samples: int = 30
    
    # Seeds for reproducibility
    train_seed: int = 0
    test_seed: int = 773
    
    # Device configuration
    jax_device: Literal['cpu', 'gpu', 'auto'] = 'cpu'
    torch_device: str = 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def get_cache_params(self) -> Dict[str, Any]:
        """Get parameters relevant for cache key generation."""
        return {
            'num_points': self.num_points,
            'num_train_samples': self.num_train_samples,
            'train_temporal_horizon': self.train_temporal_horizon,
            'train_seed': self.train_seed,
            'domain_extent': self.domain_extent,
        }


class BaseAPEBenchScenario(ABC):
    """
    Abstract base class for wrapping APEBench scenarios.
    
    Subclasses must implement:
    - _create_apebench_scenario(): Create the APEBench scenario object
    - _convert_to_flowpde_format(): Convert JAX output to FlowPDE dict format
    - get_pde_name(): Return the PDE identifier string
    
    This class provides:
    - Automatic JAX → PyTorch conversion
    - Normalization statistics computation
    - Disk caching support
    - Dataset wrapper for PyTorch DataLoader compatibility
    
    Example:
        >>> scenario = BurgersScenario(num_points=160, viscosity=0.0003)
        >>> data = scenario.generate(mode='train')
        >>> print(data['initial'].shape)  # (N, 1, 160)
    """
    
    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize the scenario wrapper.
        
        Args:
            config: ScenarioConfig object with all parameters
            cache_dir: Optional cache directory override
            **kwargs: Parameters to override in config
        """
        # Check dependencies
        if not check_apebench_available():
            raise ImportError(
                "APEBench is required for this module. "
                "Install with: pip install apebench exponax jax"
            )
        
        # Initialize config
        if config is None:
            config = ScenarioConfig()
        
        # Override config with any kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        
        # Initialize cache manager
        self.cache = CacheManager(cache_dir) if cache_dir else CacheManager()
        
        # Lazily created APEBench scenario
        self._apebench_scenario = None
    
    @property
    def apebench_scenario(self):
        """Lazily create and cache the APEBench scenario object."""
        if self._apebench_scenario is None:
            self._apebench_scenario = self._create_apebench_scenario()
        return self._apebench_scenario
    
    @abstractmethod
    def _create_apebench_scenario(self):
        """
        Create the APEBench scenario object.
        
        Returns:
            An APEBench BaseScenario subclass instance
        """
        pass
    
    @abstractmethod
    def _convert_to_flowpde_format(
        self,
        jax_data,
        mode: Literal['train', 'test'],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert JAX array output to FlowPDE dictionary format.
        
        Args:
            jax_data: Raw JAX array from APEBench
            mode: 'train' or 'test' split
            
        Returns:
            Dictionary with PyTorch tensors in FlowPDE format
        """
        pass
    
    @abstractmethod
    def get_pde_name(self) -> str:
        """
        Get the PDE identifier string.
        
        Returns:
            String like 'burgers_1d', 'poisson_2d', etc.
        """
        pass
    
    def get_cache_params(self) -> Dict[str, Any]:
        """
        Get parameters for cache key generation.
        
        Override in subclasses to include PDE-specific parameters.
        
        Returns:
            Dictionary of parameters for cache key
        """
        return self.config.get_cache_params()
    
    def generate(
        self,
        mode: Literal['train', 'test'] = 'train',
        cache: bool = False,
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate dataset using APEBench.
        
        Args:
            mode: Data split to generate ('train' or 'test')
            cache: Whether to cache the result (opt-in)
            force_regenerate: If True, regenerate even if cache exists
            
        Returns:
            Dictionary containing:
            - Tensor data (format depends on PDE type)
            - 'stats': Normalization statistics for each field
            - 'config': Generation configuration
        """
        pde_name = self.get_pde_name()
        cache_params = self.get_cache_params()
        
        # Check cache first
        if cache and not force_regenerate:
            if self.cache.exists(pde_name, cache_params):
                try:
                    cached_data = self.cache.load(pde_name, cache_params, split=mode)
                    print(f"✓ Loaded {mode} data from cache: {pde_name}")
                    return cached_data
                except Exception as e:
                    print(f"⚠ Cache load failed, regenerating: {e}")
        
        # Generate using APEBench
        print(f"⏳ Generating {mode} data for {pde_name}...")
        
        scenario = self.apebench_scenario
        
        if mode == 'train':
            jax_data = scenario.get_train_data()
        else:
            jax_data = scenario.get_test_data()
        
        # Convert to FlowPDE format
        data = self._convert_to_flowpde_format(jax_data, mode)
        
        # Compute normalization statistics
        data['stats'] = self._compute_stats(data)
        
        # Store config for reference
        data['config'] = self.config.to_dict()
        
        # Cache if requested
        if cache:
            self.cache.save(pde_name, cache_params, data, split=mode)
            print(f"✓ Cached {mode} data to: {self.cache._get_cache_path(pde_name, cache_params)}")
        
        print(f"✓ Generated {mode} data: {self._summarize_data(data)}")
        
        return data
    
    def _compute_stats(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Compute normalization statistics for all tensor fields."""
        stats = {}
        
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                stats[key] = compute_normalization_stats(value)
        
        return stats
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """Create a summary string of the generated data."""
        summaries = []
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                summaries.append(f"{key}: {tuple(value.shape)}")
        return ', '.join(summaries)
    
    def as_dataset(
        self,
        mode: Literal['train', 'test'] = 'train',
        cache: bool = False,
        problem: Literal['forward', 'inverse'] = 'forward',
    ) -> 'APEBenchDataset':
        """
        Get a PyTorch Dataset wrapper for the generated data.
        
        Args:
            mode: Data split ('train' or 'test')
            cache: Whether to use caching
            problem: Problem type ('forward' or 'inverse')
            
        Returns:
            APEBenchDataset instance compatible with DataLoader
        """
        data = self.generate(mode=mode, cache=cache)
        return APEBenchDataset(data, problem=problem)


class APEBenchDataset(Dataset):
    """
    PyTorch Dataset wrapper for APEBench-generated data.
    
    Provides a standard Dataset interface for use with DataLoader.
    Returns data in FlowPDE format: {'input': ..., 'target': ...}
    
    The actual normalization is NOT applied here - that's delegated to
    the FlowPDE training pipeline for consistency.
    """
    
    def __init__(
        self,
        data: Dict[str, Any],
        problem: Literal['forward', 'inverse'] = 'forward',
        return_dict: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Dictionary from scenario.generate()
            problem: 'forward' (input→output) or 'inverse' (output→input)
            return_dict: If True, return dict; if False, return tuple
        """
        self.data = data
        self.problem = problem
        self.return_dict = return_dict
        
        # Store stats for external access (normalization delegated to FlowPDE)
        self.stats = data.get('stats', {})
        self.config = data.get('config', {})
        
        # Determine input/target keys based on data structure
        self._setup_keys()
    
    def _setup_keys(self):
        """Determine which keys to use for input and target."""
        # For time-dependent PDEs (Burgers, etc.)
        if 'initial' in self.data and 'final' in self.data:
            if self.problem == 'forward':
                self.input_key = 'initial'
                self.target_key = 'final'
            else:  # inverse
                self.input_key = 'final'
                self.target_key = 'initial'
        
        # For static PDEs (Poisson, etc.)
        elif 'source' in self.data and 'solution' in self.data:
            if self.problem == 'forward':
                self.input_key = 'source'
                self.target_key = 'solution'
            else:  # inverse
                self.input_key = 'solution'
                self.target_key = 'source'
        
        else:
            raise ValueError(
                f"Unrecognized data format. Keys: {list(self.data.keys())}"
            )
    
    def __len__(self) -> int:
        return len(self.data[self.input_key])
    
    def __getitem__(self, idx: int) -> Union[Dict[str, torch.Tensor], Tuple]:
        input_data = self.data[self.input_key][idx]
        target_data = self.data[self.target_key][idx]
        
        if self.return_dict:
            return {
                'input': input_data,
                'target': target_data,
            }
        else:
            return input_data, target_data
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics for all fields."""
        return self.stats
    
    def get_config(self) -> Dict[str, Any]:
        """Get generation configuration."""
        return self.config
    
    def get_raw_data(self) -> Dict[str, Any]:
        """
        Get the raw underlying data dictionary.
        
        Returns all generated data including trajectories, time points,
        physics parameters, and metadata.
        
        Returns:
            Dictionary with all tensor data and metadata
        """
        return self.data
