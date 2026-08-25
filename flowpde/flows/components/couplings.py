"""
Coupling Strategies for Flow Matching.

Defines how noise samples (x_0) and data samples (x_1) are paired
during training. Different couplings can affect training efficiency.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch
from torch import Tensor


class Coupling(ABC):
    """
    Abstract base class for noise-data coupling strategies.
    
    A coupling defines how to pair samples from the base distribution
    (noise, x_0) with samples from the data distribution (x_1).
    
    Standard flow matching uses independent coupling, but optimal
    transport (OT) couplings can improve training.
    """
    
    @abstractmethod
    def couple(
        self, 
        x_0: Tensor, 
        x_1: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply coupling strategy to pair noise and data.
        
        Args:
            x_0: Noise samples, shape (B, *)
            x_1: Data samples, shape (B, *)
        
        Returns:
            (x_0_coupled, x_1_coupled): Paired samples
        """
        pass
    
    def __call__(
        self, 
        x_0: Tensor, 
        x_1: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Alias for couple()."""
        return self.couple(x_0, x_1)


class IndependentCoupling(Coupling):
    """
    Independent coupling: no reordering.
    
    Each noise sample x_0[i] is paired with data sample x_1[i]
    as they appear in the batch. This is the standard approach.
    
    Simple and fast, but may not be optimal for transport.
    """
    
    def couple(
        self, 
        x_0: Tensor, 
        x_1: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Identity coupling: return inputs unchanged."""
        return x_0, x_1


class MiniBatchOTCoupling(Coupling):
    """
    Mini-batch Optimal Transport coupling.
    
    Reorders the noise samples within each mini-batch to minimize
    total transport cost (squared Euclidean distance).
    
    This can lead to more efficient training by creating shorter
    transport paths on average.
    
    Note: Requires scipy for linear assignment. Falls back to
    independent coupling if not available.
    
    Args:
        cost_fn: Cost function ('euclidean', 'cosine'). Default: 'euclidean'
    """
    
    def __init__(self, cost_fn: str = 'euclidean'):
        self.cost_fn = cost_fn
        self._scipy_available = None
    
    def _check_scipy(self):
        """Check if scipy is available."""
        if self._scipy_available is None:
            try:
                from scipy.optimize import linear_sum_assignment
                self._scipy_available = True
            except ImportError:
                self._scipy_available = False
        return self._scipy_available
    
    def couple(
        self, 
        x_0: Tensor, 
        x_1: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute OT coupling via linear assignment.
        
        Finds permutation of x_0 that minimizes total transport cost.
        """
        if not self._check_scipy():
            # Fall back to independent coupling
            return x_0, x_1
        
        from scipy.optimize import linear_sum_assignment
        
        batch_size = x_0.shape[0]
        
        # Flatten for distance computation
        x_0_flat = x_0.view(batch_size, -1)
        x_1_flat = x_1.view(batch_size, -1)
        
        # Compute cost matrix
        if self.cost_fn == 'euclidean':
            # Squared Euclidean distance
            cost = torch.cdist(x_0_flat, x_1_flat, p=2).pow(2)
        elif self.cost_fn == 'cosine':
            # Cosine distance
            x_0_norm = x_0_flat / (x_0_flat.norm(dim=1, keepdim=True) + 1e-8)
            x_1_norm = x_1_flat / (x_1_flat.norm(dim=1, keepdim=True) + 1e-8)
            cost = 1 - torch.mm(x_0_norm, x_1_norm.t())
        else:
            raise ValueError(f"Unknown cost function: {self.cost_fn}")
        
        # Solve linear assignment (on CPU)
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        
        # Reorder x_0 to match optimal assignment
        # col_ind[i] is the index in x_1 that x_0[i] should map to
        # We need to reorder x_0 so that x_0[col_ind] maps to x_1
        inverse_perm = torch.argsort(torch.tensor(col_ind, device=x_0.device))
        x_0_coupled = x_0[inverse_perm]
        
        return x_0_coupled, x_1


# =============================================================================
# Factory Function
# =============================================================================

_COUPLING_REGISTRY = {
    'independent': IndependentCoupling,
    'none': IndependentCoupling,  # Alias
    'minibatch_ot': MiniBatchOTCoupling,
    'ot': MiniBatchOTCoupling,  # Alias
}


def get_coupling(
    coupling: Union[str, Coupling], 
    **kwargs: Any
) -> Coupling:
    """
    Get a coupling strategy by name or return if already an instance.
    
    Args:
        coupling: Coupling name ('independent', 'minibatch_ot') or instance
        **kwargs: Additional arguments for coupling constructor
    
    Returns:
        Coupling instance
    
    Examples:
        >>> coupling = get_coupling('independent')
        >>> coupling = get_coupling('minibatch_ot', cost_fn='cosine')
    """
    if isinstance(coupling, Coupling):
        return coupling
    
    if coupling not in _COUPLING_REGISTRY:
        raise ValueError(
            f"Unknown coupling: '{coupling}'. "
            f"Available: {list(_COUPLING_REGISTRY.keys())}"
        )
    
    return _COUPLING_REGISTRY[coupling](**kwargs)
