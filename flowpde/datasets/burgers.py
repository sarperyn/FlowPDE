"""
Burgers Dataset Classes
========================

PyTorch Dataset classes for Burgers equation problems.

Forward Problem: u₀ → u(t)
Inverse Problem: u(T) → u₀
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional


class BurgersForwardDataset(Dataset):
    """
    Dataset for Burgers forward problem: u₀ → u(t)
    
    Solves: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    
    Args:
        data_path: Path to .pt file containing dataset
        normalize: Whether to normalize the data
        return_dict: If True, return dict; if False, return tuple
        use_trajectory: If True, target is full trajectory; if False, only final state
        
    Returns:
        If return_dict=True:  {'input': u₀, 'target': u(t) or trajectory}
        If return_dict=False: (u₀, u(t) or trajectory)
    """
    
    def __init__(self, data_path: str, normalize: bool = True, 
                 return_dict: bool = True, use_trajectory: bool = False):
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.return_dict = return_dict
        self.use_trajectory = use_trajectory
        
        # Load data
        data = torch.load(self.data_path)
        self.initial = data['initial']        # (N, 1, resolution)
        self.final = data['final']            # (N, 1, resolution)
        self.trajectory = data['trajectory']  # (N, n_snapshots, resolution)
        self.time = data['time']              # (n_snapshots,)
        self.viscosity = data['viscosity']    # scalar
        
        # Compute normalization statistics
        if self.normalize:
            self.stats = self._compute_stats()
            self._normalize()
    
    def _compute_stats(self) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std for each field."""
        stats = {
            'initial': (self.initial.mean().item(), self.initial.std().item()),
            'final': (self.final.mean().item(), self.final.std().item()),
        }
        if self.use_trajectory:
            stats['trajectory'] = (self.trajectory.mean().item(), self.trajectory.std().item())
        return stats
    
    def _normalize(self):
        """Normalize all fields to zero mean and unit variance."""
        i_mean, i_std = self.stats['initial']
        f_mean, f_std = self.stats['final']
        
        self.initial = (self.initial - i_mean) / (i_std + 1e-8)
        self.final = (self.final - f_mean) / (f_std + 1e-8)
        
        if self.use_trajectory:
            t_mean, t_std = self.stats['trajectory']
            self.trajectory = (self.trajectory - t_mean) / (t_std + 1e-8)
    
    def __len__(self) -> int:
        return len(self.initial)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_data = self.initial[idx]  # (1, resolution)
        
        if self.use_trajectory:
            target_data = self.trajectory[idx]  # (n_snapshots, resolution)
        else:
            target_data = self.final[idx]  # (1, resolution)
        
        if self.return_dict:
            return {'input': input_data, 'target': target_data}
        else:
            return input_data, target_data


class BurgersInverseDataset(Dataset):
    """
    Dataset for Burgers inverse problem: u(T) → u₀
    
    Given observed final state, infer initial condition.
    
    Args:
        data_path: Path to .pt file containing dataset
        normalize: Whether to normalize the data
        return_dict: If True, return dict; if False, return tuple
        
    Returns:
        If return_dict=True:  {'input': u(T), 'target': u₀}
        If return_dict=False: (u(T), u₀)
    """
    
    def __init__(self, data_path: str, normalize: bool = True, return_dict: bool = True):
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.return_dict = return_dict
        
        # Load data
        data = torch.load(self.data_path)
        self.observation = data['observation']  # (N, 1, resolution)
        self.initial = data['initial']          # (N, 1, resolution)
        self.time = data['time']                # scalar
        self.viscosity = data['viscosity']      # scalar
        
        # Compute normalization statistics
        if self.normalize:
            self.stats = self._compute_stats()
            self._normalize()
    
    def _compute_stats(self) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std for each field."""
        return {
            'observation': (self.observation.mean().item(), self.observation.std().item()),
            'initial': (self.initial.mean().item(), self.initial.std().item()),
        }
    
    def _normalize(self):
        """Normalize all fields to zero mean and unit variance."""
        o_mean, o_std = self.stats['observation']
        i_mean, i_std = self.stats['initial']
        
        self.observation = (self.observation - o_mean) / (o_std + 1e-8)
        self.initial = (self.initial - i_mean) / (i_std + 1e-8)
    
    def __len__(self) -> int:
        return len(self.observation)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Observation at final time is input, initial condition is target
        input_data = self.observation[idx]  # (1, resolution)
        target_data = self.initial[idx]     # (1, resolution)
        
        if self.return_dict:
            return {'input': input_data, 'target': target_data}
        else:
            return input_data, target_data
