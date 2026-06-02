"""
Base Dataset for Exponax-Generated PDE Data
============================================

Provides a PyTorch Dataset that wraps data generated directly from
Exponax solvers and IC generators.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict


@dataclass
class GenerationConfig:
    """
    Configuration for PDE data generation via Exponax.

    Attributes:
        num_spatial_dims: Number of spatial dimensions (1, 2, or 3)
        num_points: Number of grid points per spatial dimension
        domain_extent: Physical size of the periodic domain
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        torch_device: Target PyTorch device for converted tensors
        obs_noise_std: Optional additive Gaussian noise applied to the
            inverse-problem observation field.  Set to 0.0 to disable.
        obs_mask_fraction: Fraction of spatial grid points that are *observed*
            in inverse-problem datasets.  Each sample independently draws a
            random Bernoulli mask with this probability; unobserved locations
            are zeroed out in the observation field.  The binary mask
            (1 = observed, 0 = unobserved) is stored as ``'obs_mask'`` in the
            dataset and returned by ``__getitem__``.  1.0 (default) = full
            observations; 0.1 = only 10 % of points visible.
            Has no effect when ``problem='forward'`` or when left at 1.0.
    """
    num_spatial_dims: int = 2
    num_points: int = 64
    domain_extent: float = 10.0
    num_samples: int = 10000
    seed: int = 42
    torch_device: str = 'cpu'
    obs_noise_std: float = 0.0
    obs_mask_fraction: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PDEDataset(Dataset):
    """
    PyTorch Dataset wrapping Exponax-generated PDE data.

    Returns samples as {'input': ..., 'target': ...} where the
    semantics of *input* and *target* depend on the PDE type and the
    chosen problem direction:

    * **Poisson** (static): input=source, target=solution (forward)
      or input=solution, target=source (inverse).
    * **Burgers** (time-dependent): input=initial condition,
      target=final state (forward) or vice-versa (inverse).

    When partial observations are enabled (``obs_mask_fraction < 1.0``),
    ``__getitem__`` appends the observation mask to the conditioning input
    and additionally returns ``'obs_mask'``: a float tensor of shape
    ``(1, *spatial)`` with 1 at observed locations and 0 elsewhere.

    The dataset also stores normalization statistics and generation
    config for reference.
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        problem: Literal['forward', 'inverse'] = 'forward',
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            data: Dictionary with at least two tensor entries whose
                  keys identify the PDE fields (e.g. 'source' / 'solution'
                  for Poisson, 'initial' / 'final' for Burgers).
            problem: 'forward' maps natural data -> solution;
                     'inverse' reverses the mapping.
            metadata: Optional dict with 'stats', 'config', etc.
        """
        if problem not in {'forward', 'inverse'}:
            raise ValueError("problem must be 'forward' or 'inverse'")

        self.data = data
        self.problem = problem
        self.metadata = metadata or {}
        self.stats = self.metadata.get('stats', {})
        self.config = self.metadata.get('config', {})

        self._setup_keys()


    def _setup_keys(self):
        """Determine input / target keys from the available data."""
        #Static PDEs (Poisson)
        if 'source' in self.data and 'solution' in self.data:
            if self.problem == 'forward':
                self.input_key = 'source'
                self.target_key = 'solution'
            else:
                self.input_key = 'solution'
                self.target_key = 'source'

        #Time-dependent PDEs (Burgers)
        elif 'initial' in self.data and 'final' in self.data:
            if self.problem == 'forward':
                self.input_key = 'initial'
                self.target_key = 'final'
            else:
                self.input_key = 'final'
                self.target_key = 'initial'

        else:
            raise ValueError(
                f"Unrecognized data format. Keys: {list(self.data.keys())}. "
                "Expected ('source', 'solution') or ('initial', 'final')."
            )

    #Dataset interface
    def __len__(self) -> int:
        return len(self.data[self.input_key])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inp = self.data[self.input_key][idx]
        obs_mask = self.data.get('obs_mask')
        if obs_mask is not None:
            inp = torch.cat([inp, obs_mask[idx]], dim=0)

        sample = {
            'input': inp,
            'target': self.data[self.target_key][idx],
        }
        if obs_mask is not None:
            sample['obs_mask'] = obs_mask[idx]
        return sample

    # Getters for metadata
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics for all fields."""
        return self.stats

    def get_config(self) -> Dict[str, Any]:
        """Get generation configuration."""
        return self.config

    def get_raw_data(self) -> Dict[str, Any]:
        """Get the raw underlying data dictionary."""
        return self.data
