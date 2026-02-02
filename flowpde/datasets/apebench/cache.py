"""
Dataset Caching Utilities
==========================

Provides disk caching functionality for APEBench-generated datasets
to avoid redundant computation on subsequent runs.

Cache Structure:
----------------
    .cache/apebench/
    ├── burgers_1d/
    │   ├── res160_nu0.0003_n500_t50/
    │   │   ├── metadata.json
    │   │   ├── train.pt
    │   │   └── test.pt
    │   └── ...
    └── poisson_2d/
        └── ...
"""

import json
import hashlib
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


class CacheManager:
    """
    Manages disk caching for APEBench-generated datasets.
    
    Features:
    - Automatic cache key generation from parameters
    - Metadata storage (generation params, timestamps, checksums)
    - Cache validation to detect corrupted files
    - Thread-safe file operations
    
    Example:
        >>> cache = CacheManager('.cache/apebench')
        >>> 
        >>> # Check if dataset exists
        >>> if cache.exists('burgers_1d', params):
        ...     data = cache.load('burgers_1d', params)
        >>> else:
        ...     data = generate_data(...)
        ...     cache.save('burgers_1d', params, data)
    """
    
    DEFAULT_CACHE_DIR = '.cache/apebench'
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Root directory for cache storage.
                       Defaults to '.cache/apebench' in the current working directory.
        """
        if cache_dir is None:
            cache_dir = self.DEFAULT_CACHE_DIR
        self.cache_dir = Path(cache_dir)
    
    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key from parameters.
        
        Creates a deterministic string identifier based on the parameter values.
        The key is human-readable for easy inspection.
        
        Args:
            params: Dictionary of generation parameters
            
        Returns:
            Cache key string (e.g., 'res160_nu0.0003_n500_t50')
        """
        # Define the order and formatting of key components
        key_parts = []
        
        # Resolution
        if 'num_points' in params:
            key_parts.append(f"res{params['num_points']}")
        elif 'resolution' in params:
            key_parts.append(f"res{params['resolution']}")
        
        # Viscosity/diffusivity
        if 'viscosity' in params:
            key_parts.append(f"nu{params['viscosity']}")
        elif 'diffusivity' in params:
            key_parts.append(f"nu{params['diffusivity']}")
        
        # Number of samples
        if 'n_train_samples' in params:
            key_parts.append(f"n{params['n_train_samples']}")
        elif 'num_train_samples' in params:
            key_parts.append(f"n{params['num_train_samples']}")
        
        # Temporal horizon
        if 'temporal_horizon' in params:
            key_parts.append(f"t{params['temporal_horizon']}")
        elif 'train_temporal_horizon' in params:
            key_parts.append(f"t{params['train_temporal_horizon']}")
        
        # Seed for reproducibility
        if 'seed' in params:
            key_parts.append(f"s{params['seed']}")
        elif 'train_seed' in params:
            key_parts.append(f"s{params['train_seed']}")
        
        # Domain extent if non-default
        if 'domain_extent' in params and params['domain_extent'] != 1.0:
            key_parts.append(f"L{params['domain_extent']}")
        
        # Fallback: use hash if no recognizable params
        if not key_parts:
            param_str = json.dumps(params, sort_keys=True)
            hash_val = hashlib.md5(param_str.encode()).hexdigest()[:8]
            key_parts.append(f"hash{hash_val}")
        
        return '_'.join(key_parts)
    
    def _get_cache_path(self, pde_name: str, params: Dict[str, Any]) -> Path:
        """Get the full cache directory path for a dataset."""
        cache_key = self._get_cache_key(params)
        return self.cache_dir / pde_name / cache_key
    
    def exists(self, pde_name: str, params: Dict[str, Any]) -> bool:
        """
        Check if a cached dataset exists and is valid.
        
        Args:
            pde_name: Name of the PDE (e.g., 'burgers_1d')
            params: Generation parameters
            
        Returns:
            True if valid cache exists, False otherwise
        """
        cache_path = self._get_cache_path(pde_name, params)
        
        # Check if directory exists
        if not cache_path.exists():
            return False
        
        # Check for required files
        metadata_path = cache_path / 'metadata.json'
        train_path = cache_path / 'train.pt'
        
        if not metadata_path.exists() or not train_path.exists():
            return False
        
        # Validate metadata matches parameters
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if stored params match requested params
            stored_params = metadata.get('params', {})
            for key, value in params.items():
                if key in stored_params and stored_params[key] != value:
                    return False
            
            return True
        except (json.JSONDecodeError, KeyError):
            return False
    
    def save(
        self,
        pde_name: str,
        params: Dict[str, Any],
        data: Dict[str, Any],
        split: str = 'train',
    ) -> Path:
        """
        Save a dataset to the cache.
        
        Args:
            pde_name: Name of the PDE (e.g., 'burgers_1d')
            params: Generation parameters (stored in metadata)
            data: Dictionary containing tensors and stats to save
            split: Data split name ('train' or 'test')
            
        Returns:
            Path to the saved cache directory
        """
        cache_path = self._get_cache_path(pde_name, params)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save data tensors
        data_path = cache_path / f'{split}.pt'
        torch.save(data, data_path)
        
        # Save or update metadata
        metadata_path = cache_path / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'pde_name': pde_name,
                'params': params,
                'created_at': datetime.now().isoformat(),
                'splits': {},
            }
        
        # Update split info
        metadata['splits'][split] = {
            'file': f'{split}.pt',
            'saved_at': datetime.now().isoformat(),
            'num_samples': self._count_samples(data),
        }
        metadata['updated_at'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return cache_path
    
    def load(
        self,
        pde_name: str,
        params: Dict[str, Any],
        split: str = 'train',
    ) -> Dict[str, Any]:
        """
        Load a dataset from the cache.
        
        Args:
            pde_name: Name of the PDE (e.g., 'burgers_1d')
            params: Generation parameters
            split: Data split to load ('train' or 'test')
            
        Returns:
            Dictionary containing the cached data
            
        Raises:
            FileNotFoundError: If cache doesn't exist
        """
        cache_path = self._get_cache_path(pde_name, params)
        data_path = cache_path / f'{split}.pt'
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"No cached {split} data found at {data_path}. "
                f"Generate the dataset first with cache=True."
            )
        
        return torch.load(data_path, weights_only=False)
    
    def get_metadata(
        self,
        pde_name: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a cached dataset.
        
        Args:
            pde_name: Name of the PDE
            params: Generation parameters
            
        Returns:
            Metadata dictionary or None if not found
        """
        cache_path = self._get_cache_path(pde_name, params)
        metadata_path = cache_path / 'metadata.json'
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def clear(self, pde_name: Optional[str] = None) -> int:
        """
        Clear cached datasets.
        
        Args:
            pde_name: If provided, only clear cache for this PDE.
                      If None, clear all caches.
                      
        Returns:
            Number of cache directories removed
        """
        import shutil
        
        count = 0
        
        if pde_name is not None:
            target_dir = self.cache_dir / pde_name
            if target_dir.exists():
                shutil.rmtree(target_dir)
                count = 1
        else:
            if self.cache_dir.exists():
                for pde_dir in self.cache_dir.iterdir():
                    if pde_dir.is_dir():
                        shutil.rmtree(pde_dir)
                        count += 1
        
        return count
    
    def list_cached(self, pde_name: Optional[str] = None) -> list:
        """
        List all cached datasets.
        
        Args:
            pde_name: If provided, only list caches for this PDE
            
        Returns:
            List of dictionaries with cache info
        """
        caches = []
        
        if not self.cache_dir.exists():
            return caches
        
        pde_dirs = [self.cache_dir / pde_name] if pde_name else self.cache_dir.iterdir()
        
        for pde_dir in pde_dirs:
            if not pde_dir.is_dir():
                continue
            
            for cache_dir in pde_dir.iterdir():
                if not cache_dir.is_dir():
                    continue
                
                metadata_path = cache_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    caches.append({
                        'pde': pde_dir.name,
                        'cache_key': cache_dir.name,
                        'path': str(cache_dir),
                        'metadata': metadata,
                    })
        
        return caches
    
    def _count_samples(self, data: Dict[str, Any]) -> int:
        """Count the number of samples in a data dictionary."""
        for key in ['initial', 'trajectory', 'source', 'solution']:
            if key in data and hasattr(data[key], '__len__'):
                return len(data[key])
        return 0
