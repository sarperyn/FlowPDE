"""
Poisson Dataset Classes
========================

PyTorch Dataset classes for Poisson equation problems.

Forward Problem: (source, coefficient) → solution
Inverse Problem: observation → (source, coefficient)
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional


class PoissonForwardDataset(Dataset):
    """
    Dataset for Poisson forward problem: (f, a) → u
    
    Solves: -∇·(a∇u) = f with Dirichlet BC
    
    Args:
        data_path: Path to .pt file containing dataset
        normalize: Whether to normalize the data
        return_dict: If True, return dict; if False, return tuple
        
    Returns:
        If return_dict=True:  {'input': (f, a), 'target': u}
        If return_dict=False: ((f, a), u)
    """
    
    def __init__(self, data_path: str, normalize: bool = True, return_dict: bool = True):
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.return_dict = return_dict
        
        # Load data
        data = torch.load(self.data_path)
        self.source = data['source']           # (N, 1, H, W)
        self.coefficient = data['coefficient'] # (N, 1, H, W)
        self.solution = data['solution']       # (N, 1, H, W)
        
        # Compute normalization statistics
        if self.normalize:
            self.stats = self._compute_stats()
            self._normalize()
    
    def _compute_stats(self) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std for each field."""
        return {
            'source': (self.source.mean().item(), self.source.std().item()),
            'coefficient': (self.coefficient.mean().item(), self.coefficient.std().item()),
            'solution': (self.solution.mean().item(), self.solution.std().item()),
        }
    
    def _normalize(self):
        """Normalize all fields to zero mean and unit variance."""
        s_mean, s_std = self.stats['source']
        c_mean, c_std = self.stats['coefficient']
        u_mean, u_std = self.stats['solution']
        
        self.source = (self.source - s_mean) / (s_std + 1e-8)
        self.coefficient = (self.coefficient - c_mean) / (c_std + 1e-8)
        self.solution = (self.solution - u_mean) / (u_std + 1e-8)
    
    def __len__(self) -> int:
        return len(self.source)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Concatenate source and coefficient as input
        input_data = torch.cat([self.source[idx], self.coefficient[idx]], dim=0)  # (2, H, W)
        target_data = self.solution[idx]  # (1, H, W)
        
        if self.return_dict:
            return {'input': input_data, 'target': target_data}
        else:
            return input_data, target_data


class PoissonInverseDataset(Dataset):
    """
    Dataset for Poisson inverse problem: u → (f, a)
    
    Given observed solution, infer source term and coefficient.
    
    Args:
        data_path: Path to .pt file containing dataset
        normalize: Whether to normalize the data
        return_dict: If True, return dict; if False, return tuple
        
    Returns:
        If return_dict=True:  {'input': u_obs, 'target': (f, a)}
        If return_dict=False: (u_obs, (f, a))
    """
    
    def __init__(self, data_path: str, normalize: bool = True, return_dict: bool = True):
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.return_dict = return_dict
        
        # Load data
        data = torch.load(self.data_path)
        self.observation = data['observation']  # (N, 1, H, W)
        self.source = data['source']            # (N, 1, H, W)
        self.coefficient = data['coefficient']  # (N, 1, H, W)
        
        # Compute normalization statistics
        if self.normalize:
            self.stats = self._compute_stats()
            self._normalize()
    
    def _compute_stats(self) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std for each field."""
        return {
            'observation': (self.observation.mean().item(), self.observation.std().item()),
            'source': (self.source.mean().item(), self.source.std().item()),
            'coefficient': (self.coefficient.mean().item(), self.coefficient.std().item()),
        }
    
    def _normalize(self):
        """Normalize all fields to zero mean and unit variance."""
        o_mean, o_std = self.stats['observation']
        s_mean, s_std = self.stats['source']
        c_mean, c_std = self.stats['coefficient']
        
        self.observation = (self.observation - o_mean) / (o_std + 1e-8)
        self.source = (self.source - s_mean) / (s_std + 1e-8)
        self.coefficient = (self.coefficient - c_mean) / (c_std + 1e-8)
    
    def __len__(self) -> int:
        return len(self.observation)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Observation is input, parameters are target
        input_data = self.observation[idx]  # (1, H, W)
        target_data = torch.cat([self.source[idx], self.coefficient[idx]], dim=0)  # (2, H, W)
        
        if self.return_dict:
            return {'input': input_data, 'target': target_data}
        else:
            return input_data, target_data