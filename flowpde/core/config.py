"""
Configuration Management for FlowPDE

Provides centralized configuration handling with YAML support,
environment variables, and nested config access.
"""

import os
import yaml
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
import copy


class ConfigDict(dict):
    """
    Dictionary with dot notation access.
    
    Example:
        >>> cfg = ConfigDict({'model': {'type': 'unet', 'dim': 64}})
        >>> cfg.model.type  # 'unet'
        >>> cfg['model']['dim']  # 64
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dicts to ConfigDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
    
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any):
        self[key] = value
    
    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")
    
    def to_dict(self) -> Dict:
        """Convert to regular dict recursively."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update_nested(self, updates: Dict):
        """Update nested dictionary values."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in self and isinstance(self[key], (dict, ConfigDict)):
                if isinstance(self[key], ConfigDict):
                    self[key].update_nested(value)
                else:
                    self[key] = ConfigDict(self[key])
                    self[key].update_nested(value)
            else:
                self[key] = ConfigDict(value) if isinstance(value, dict) else value
    
    def merge(self, other: Dict):
        """Merge another dict into this one (deep merge)."""
        self.update_nested(other)
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Get value using dot-separated path.
        
        Example:
            >>> cfg.get_nested('model.unet.base_ch', default=64)
        """
        keys = key_path.split('.')
        value = self
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value


@dataclass
class BaseConfig:
    """
    Base configuration dataclass.
    
    Provides common functionality for all config classes.
    """
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, filepath: Union[str, Path]):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for models."""
    type: str = 'unet'
    input_dim: int = 64
    base_ch: int = 64
    time_dim: int = 1
    kwargs: Dict = field(default_factory=dict)


@dataclass
class SolverConfig(BaseConfig):
    """Configuration for ODE/SDE solvers."""
    type: str = 'torchdiffeq'
    method: str = 'dopri5'
    rtol: float = 1e-5
    atol: float = 1e-7
    adjoint: bool = False
    method_options: Dict = field(default_factory=dict)


@dataclass
class FlowConfig(BaseConfig):
    """Configuration for flow algorithms."""
    type: str = 'flow_matching'
    base_distribution: str = 'gaussian'
    path_type: str = 'linear'
    kwargs: Dict = field(default_factory=dict)


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    save_interval: int = 10
    print_interval: int = 1
    device: str = 'cuda'
    save_dir: str = 'results/training'
    optimizer_kwargs: Dict = field(default_factory=dict)
    scheduler_kwargs: Dict = field(default_factory=dict)


@dataclass
class InferenceConfig(BaseConfig):
    """Configuration for inference."""
    n_steps: int = 50
    batch_size: int = 32
    device: str = 'cuda'
    save_dir: str = 'results/inference'
    return_trajectory: bool = False
    solver: SolverConfig = field(default_factory=SolverConfig)


@dataclass
class DataConfig(BaseConfig):
    """Configuration for datasets."""
    type: str = 'poisson'
    train_path: str = ''
    test_path: str = ''
    val_split: float = 0.1
    num_workers: int = 4
    kwargs: Dict = field(default_factory=dict)


@dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration."""
    name: str = 'flow_matching_experiment'
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)


class Config:
    """
    Main configuration manager for FlowPDE.
    
    Supports:
    - Loading from YAML files
    - Environment variable overrides
    - Dot notation access
    - Config validation
    - Default values
    """
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional initial configuration dictionary
        """
        self._config = ConfigDict(config_dict or {})
        self._frozen = False
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML config file
            
        Returns:
            Config instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create configuration from dictionary."""
        return cls(config_dict)
    
    def to_yaml(self, filepath: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Path to save YAML file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self._config.to_dict(), f, default_flow_style=False)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self._config.to_dict()
    
    def merge(self, other: Union['Config', Dict]):
        """
        Merge another configuration into this one.
        
        Args:
            other: Another Config instance or dictionary
        """
        if self._frozen:
            raise RuntimeError("Cannot merge into frozen config")
        
        if isinstance(other, Config):
            other_dict = other.to_dict()
        else:
            other_dict = other
        
        self._config.merge(other_dict)
    
    def update(self, **kwargs):
        """Update configuration with keyword arguments."""
        if self._frozen:
            raise RuntimeError("Cannot update frozen config")
        
        self._config.update_nested(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key: Dot-separated key path (e.g., 'model.unet.dim')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get_nested(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key path.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        if self._frozen:
            raise RuntimeError("Cannot set value in frozen config")
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = ConfigDict()
            config = config[k]
        
        config[keys[-1]] = value
    
    def freeze(self):
        """Freeze configuration to prevent modifications."""
        self._frozen = True
    
    def unfreeze(self):
        """Unfreeze configuration to allow modifications."""
        self._frozen = False
    
    @property
    def is_frozen(self) -> bool:
        """Check if configuration is frozen."""
        return self._frozen
    
    def clone(self) -> 'Config':
        """Create a deep copy of the configuration."""
        return Config(copy.deepcopy(self._config.to_dict()))
    
    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            return super().__getattribute__(key)
        return self._config[key]
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        if self._frozen:
            raise RuntimeError("Cannot modify frozen config")
        self._config[key] = value
    
    def __repr__(self) -> str:
        frozen_str = " (frozen)" if self._frozen else ""
        return f"Config{frozen_str}({self._config})"
    
    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)


def load_config(
    filepath: Union[str, Path],
    overrides: Optional[Dict] = None,
    env_prefix: str = 'FLOWPDE_'
) -> Config:
    """
    Load configuration from file with optional overrides.
    
    Args:
        filepath: Path to YAML config file
        overrides: Optional dictionary of overrides
        env_prefix: Prefix for environment variable overrides
        
    Returns:
        Config instance
        
    Example:
        >>> config = load_config(
        ...     'configs/training/unet.yaml',
        ...     overrides={'training.epochs': 200}
        ... )
    """
    # Load base config
    config = Config.from_yaml(filepath)
    
    # Apply environment variable overrides
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            config_key = key[len(env_prefix):].lower().replace('_', '.')
            # Try to parse value as int, float, bool, or keep as string
            try:
                if value.lower() in ('true', 'false'):
                    parsed_value = value.lower() == 'true'
                elif '.' in value:
                    parsed_value = float(value)
                else:
                    parsed_value = int(value)
            except ValueError:
                parsed_value = value
            
            config.set(config_key, parsed_value)
    
    # Apply manual overrides
    if overrides:
        for key, value in overrides.items():
            config.set(key, value)
    
    return config


def create_default_config(config_type: str = 'experiment') -> Union[Config, BaseConfig]:
    """
    Create default configuration.
    
    Args:
        config_type: Type of config to create
            ('experiment', 'model', 'training', 'inference', etc.)
            
    Returns:
        Default configuration instance
    """
    config_classes = {
        'experiment': ExperimentConfig,
        'model': ModelConfig,
        'solver': SolverConfig,
        'flow': FlowConfig,
        'training': TrainingConfig,
        'inference': InferenceConfig,
        'data': DataConfig,
    }
    
    if config_type not in config_classes:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(config_classes.keys())}")
    
    return config_classes[config_type]()
