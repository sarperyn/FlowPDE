"""
Poisson Forward Problem Generator
==================================

Forward Problem: (source_term, coefficient) → solution
    Given: f(x,y) and a(x,y)
    Find: u(x,y) such that -∇·(a∇u) = f with Dirichlet BC u=0 on boundary

Usage:
    python generate_forward.py --n_train 5000 --n_test 1000 --resolution 64
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
import argparse
from tqdm import tqdm
from scipy.fft import dct, idct
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class GaussianRandomField:
    """Generate smooth random fields using Gaussian Random Field."""
    
    def __init__(self, alpha: float = 2.0, tau: float = 3.0, size: int = 64):
        """
        Args:
            alpha: Smoothness parameter (higher = smoother)
            tau: Length scale parameter
            size: Grid resolution
        """
        self.alpha = alpha
        self.tau = tau
        self.size = size
        
        # Precompute eigenvalue coefficients
        k1, k2 = np.meshgrid(np.arange(size), np.arange(size))
        self.coef = tau**(alpha - 1) * (np.pi**2 * (k1**2 + k2**2) + tau**2)**(-alpha/2)
    
    def sample(self, smooth_boundary: bool = True) -> np.ndarray:
        """Generate one GRF sample."""
        xi = np.random.randn(self.size, self.size)
        L = self.size * self.coef * xi
        L[0, 0] = 0  # Zero mean
        
        # Inverse DCT
        U = idct(idct(L, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        if smooth_boundary:
            # Smoothly bring edges to zero
            x = np.linspace(-1, 1, self.size)
            y = np.linspace(-1, 1, self.size)
            X, Y = np.meshgrid(x, y)
            window = 0.75 * (1 + np.cos(np.pi * X)) * 0.75 * (1 + np.cos(np.pi * Y))
            U = U * window
        
        return U


class PoissonSolver:
    """Solve -∇·(a∇u) = f with Dirichlet BC."""
    
    def __init__(self, size: int, domain: Tuple[float, float] = (0, 1)):
        self.size = size
        self.h = (domain[1] - domain[0]) / (size + 1)  # Grid spacing
        self.n_interior = size * size
        
    def solve(self, f: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Solve the Poisson equation.
        
        Args:
            f: Source term (size x size)
            a: Diffusion coefficient (size x size)
            
        Returns:
            u: Solution (size x size)
        """
        # Build sparse matrix for -∇·(a∇u)
        # Using finite differences on interior points
        A_sparse = self._build_matrix(a)
        
        # Flatten source term (only interior points)
        f_vec = f.flatten()
        
        # Solve sparse system
        u_vec = spla.spsolve(A_sparse, f_vec)
        
        # Reshape to 2D
        u = u_vec.reshape(self.size, self.size)
        
        return u
    
    def _build_matrix(self, a: np.ndarray) -> sp.csr_matrix:
        """Build finite difference matrix."""
        n = self.size
        h2 = self.h ** 2
        
        # We'll use 5-point stencil
        # For -∇·(a∇u) ≈ -(a_{i+1/2,j}(u_{i+1,j}-u_{i,j}) - a_{i-1/2,j}(u_{i,j}-u_{i-1,j}))/h²
        #                  -(a_{i,j+1/2}(u_{i,j+1}-u_{i,j}) - a_{i,j-1/2}(u_{i,j}-u_{i,j-1}))/h²
        
        data = []
        row_ind = []
        col_ind = []
        
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                
                # Center coefficient
                a_center = 0.0
                
                # East neighbor
                if j < n - 1:
                    a_ij = (a[i, j] + a[i, j + 1]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx + 1)
                    a_center += a_ij / h2
                else:
                    # Boundary (u = 0)
                    a_ij = a[i, j]
                    a_center += a_ij / h2
                
                # West neighbor
                if j > 0:
                    a_ij = (a[i, j] + a[i, j - 1]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx - 1)
                    a_center += a_ij / h2
                else:
                    a_ij = a[i, j]
                    a_center += a_ij / h2
                
                # North neighbor
                if i < n - 1:
                    a_ij = (a[i, j] + a[i + 1, j]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx + n)
                    a_center += a_ij / h2
                else:
                    a_ij = a[i, j]
                    a_center += a_ij / h2
                
                # South neighbor
                if i > 0:
                    a_ij = (a[i, j] + a[i - 1, j]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx - n)
                    a_center += a_ij / h2
                else:
                    a_ij = a[i, j]
                    a_center += a_ij / h2
                
                # Diagonal (center point)
                data.append(a_center)
                row_ind.append(idx)
                col_ind.append(idx)
        
        A = sp.coo_matrix((data, (row_ind, col_ind)), shape=(n * n, n * n))
        return A.tocsr()


def generate_dataset(n_samples: int, resolution: int, 
                     grf_alpha: float = 2.0, grf_tau: float = 3.0) -> Dict[str, torch.Tensor]:
    """
    Generate forward problem dataset.
    
    Returns:
        dict with keys:
            'source': Source term f (n_samples, 1, H, W)
            'coefficient': Diffusion coefficient a (n_samples, 1, H, W)
            'solution': Solution u (n_samples, 1, H, W)
    """
    grf = GaussianRandomField(alpha=grf_alpha, tau=grf_tau, size=resolution)
    solver = PoissonSolver(size=resolution)
    
    sources = []
    coefficients = []
    solutions = []
    
    print(f"Generating {n_samples} forward problem samples...")
    for _ in tqdm(range(n_samples)):
        # Generate random source term
        f = grf.sample(smooth_boundary=True)
        
        # Generate random coefficient (positive, bounded away from zero)
        a_raw = grf.sample(smooth_boundary=False)
        a = 1.0 + 0.5 * a_raw  # Ensure a > 0.5
        
        # Solve PDE
        u = solver.solve(f, a)
        
        sources.append(f)
        coefficients.append(a)
        solutions.append(u)
    
    # Convert to tensors with shape (N, 1, H, W)
    dataset = {
        'source': torch.tensor(np.array(sources), dtype=torch.float32).unsqueeze(1),
        'coefficient': torch.tensor(np.array(coefficients), dtype=torch.float32).unsqueeze(1),
        'solution': torch.tensor(np.array(solutions), dtype=torch.float32).unsqueeze(1),
    }
    
    return dataset


def visualize_samples(dataset: Dict[str, torch.Tensor], n_vis: int = 4, save_path: str = None):
    """Visualize forward problem: (f, a) → u"""
    fig, axes = plt.subplots(n_vis, 3, figsize=(12, 4 * n_vis))
    
    for i in range(n_vis):
        f = dataset['source'][i, 0].numpy()
        a = dataset['coefficient'][i, 0].numpy()
        u = dataset['solution'][i, 0].numpy()
        
        # Source term
        im0 = axes[i, 0].imshow(f, cmap='RdBu_r', aspect='auto')
        axes[i, 0].set_title(f'Source Term f (sample {i+1})')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Coefficient
        im1 = axes[i, 1].imshow(a, cmap='viridis', aspect='auto')
        axes[i, 1].set_title(f'Coefficient a (sample {i+1})')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Solution
        im2 = axes[i, 2].imshow(u, cmap='RdBu_r', aspect='auto')
        axes[i, 2].set_title(f'Solution u (sample {i+1})')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate Poisson forward problem dataset')
    parser.add_argument('--n_train', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--resolution', type=int, default=64, help='Grid resolution')
    parser.add_argument('--grf_alpha', type=float, default=2.0, help='GRF smoothness')
    parser.add_argument('--grf_tau', type=float, default=3.0, help='GRF length scale')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    args = parser.parse_args()
    
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / 'datasets' / 'poisson' / 'forward'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training set
    train_data = generate_dataset(
        n_samples=args.n_train,
        resolution=args.resolution,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Generate test set
    test_data = generate_dataset(
        n_samples=args.n_test,
        resolution=args.resolution,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Save datasets
    train_path = output_dir / 'train.pt'
    test_path = output_dir / 'test.pt'
    
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    
    print(f"\n✓ Training data saved to {train_path}")
    print(f"  Shape: source {train_data['source'].shape}, "
          f"coefficient {train_data['coefficient'].shape}, "
          f"solution {train_data['solution'].shape}")
    
    print(f"\n✓ Test data saved to {test_path}")
    print(f"  Shape: source {test_data['source'].shape}, "
          f"coefficient {test_data['coefficient'].shape}, "
          f"solution {test_data['solution'].shape}")
    
    # Visualize
    if args.visualize:
        vis_path = output_dir / 'visualization.png'
        visualize_samples(train_data, n_vis=4, save_path=str(vis_path))


if __name__ == '__main__':
    main()
