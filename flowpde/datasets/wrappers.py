"""
Dataset Wrappers for Flow Training
===================================

Wrappers that convert standard dataset formats to flow-compatible formats.

Flow trainers expect data in the format:
    {'f': condition, 'u': target}

But datasets might return different formats like:
    {'input': ..., 'target': ...}
    {'source': ..., 'coefficient': ..., 'solution': ...}

These wrappers handle the conversion transparently.
"""

from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Callable


class FlowDatasetWrapper(Dataset):
    """
    Wrapper to convert standard dataset format to flow-compatible format.
    
    Flow trainers expect:
        {'f': condition, 'u': target}
    
    This wrapper converts from common formats:
        - {'input': X, 'target': y} → {'f': X, 'u': y}
        - Custom mapping via input_key and target_key
    
    Args:
        dataset: The underlying dataset to wrap
        input_key: Key name for condition in original dataset (default: 'input')
        target_key: Key name for target in original dataset (default: 'target')
        transform: Optional transform to apply to samples
    
    Example:
        >>> from flowpde.datasets import PoissonForwardDataset, FlowDatasetWrapper
        >>> dataset = PoissonForwardDataset('train.pt')
        >>> flow_dataset = FlowDatasetWrapper(dataset)
        >>> sample = flow_dataset[0]
        >>> print(sample.keys())  # dict_keys(['f', 'u'])
    """
    
    def __init__(
        self,
        dataset: Dataset,
        input_key: str = 'input',
        target_key: str = 'target',
        transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.input_key = input_key
        self.target_key = target_key
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item in flow-compatible format.
        
        Returns:
            dict with keys:
                'f': condition tensor
                'u': target tensor
        """
        sample = self.dataset[idx]
        
        # Convert to flow format
        flow_sample = {
            'f': sample[self.input_key],
            'u': sample[self.target_key]
        }
        
        # Apply optional transform
        if self.transform is not None:
            flow_sample = self.transform(flow_sample)
        
        return flow_sample


class InverseFlowDatasetWrapper(Dataset):
    """
    Wrapper for inverse problems: observation → parameters.
    
    For inverse problems, the 'observation' is the condition and
    the parameters (source, coefficient, etc.) are the targets.
    
    Args:
        dataset: The underlying inverse problem dataset
        observation_key: Key for observation in original dataset (default: 'observation')
        target_keys: List of keys to concatenate as target (e.g., ['source', 'coefficient'])
        transform: Optional transform to apply to samples
    
    Example:
        >>> from flowpde.datasets import PoissonInverseDataset, InverseFlowDatasetWrapper
        >>> dataset = PoissonInverseDataset('train.pt')
        >>> flow_dataset = InverseFlowDatasetWrapper(
        ...     dataset,
        ...     target_keys=['source', 'coefficient']
        ... )
        >>> sample = flow_dataset[0]
        >>> sample['f'].shape  # observation (condition)
        >>> sample['u'].shape  # concatenated [source, coefficient] (target)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        observation_key: str = 'observation',
        target_keys: Optional[list] = None,
        transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.observation_key = observation_key
        self.target_keys = target_keys or ['source', 'coefficient']
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item in flow-compatible format for inverse problems.
        
        Returns:
            dict with keys:
                'f': observation (condition)
                'u': concatenated target parameters
        """
        import torch
        
        sample = self.dataset[idx]
        
        # Observation is the condition
        condition = sample[self.observation_key]
        
        # Concatenate target fields if multiple
        if len(self.target_keys) == 1:
            target = sample[self.target_keys[0]]
        else:
            targets = [sample[key] for key in self.target_keys]
            target = torch.cat(targets, dim=0)
        
        flow_sample = {
            'f': condition,
            'u': target
        }
        
        if self.transform is not None:
            flow_sample = self.transform(flow_sample)
        
        return flow_sample


__all__ = [
    'FlowDatasetWrapper',
    'InverseFlowDatasetWrapper',
]
