"""
Poisson Inverse Problem Generator
==================================

Inverse Problem: solution → (source_term, coefficient)
    Given: Observed solution u(x,y) (possibly with noise)
    Infer: Source term f(x,y) and coefficient a(x,y)

This is the dataset for learning the inverse mapping using normalizing flows.

Usage:
    python generate_inverse.py --n_train 5000 --n_test 1000 --resolution 64 --noise_level 0.01
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
        self.alpha = alpha
        self.tau = tau
        self.size = size
        
        k1, k2 = np.meshgrid(np.arange(size), np.arange(size))
        self.coef = tau**(alpha - 1) * (np.pi**2 * (k1**2 + k2**2) + tau**2)**(-alpha/2)
    
    def sample(self, smooth_boundary: bool = True) -> np.ndarray:
        xi = np.random.randn(self.size, self.size)
        L = self.size * self.coef * xi
        L[0, 0] = 0
        
        U = idct(idct(L, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        if smooth_boundary:
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
        self.h = (domain[1] - domain[0]) / (size + 1)
        self.n_interior = size * size
        
    def solve(self, f: np.ndarray, a: np.ndarray) -> np.ndarray:
        A_sparse = self._build_matrix(a)
        f_vec = f.flatten()
        u_vec = spla.spsolve(A_sparse, f_vec)
        u = u_vec.reshape(self.size, self.size)
        return u
    
    def _build_matrix(self, a: np.ndarray) -> sp.csr_matrix:
        n = self.size
        h2 = self.h ** 2
        
        data = []
        row_ind = []
        col_ind = []
        
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                a_center = 0.0
                
                if j < n - 1:
                    a_ij = (a[i, j] + a[i, j + 1]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx + 1)
                    a_center += a_ij / h2
                else:
                    a_ij = a[i, j]
                    a_center += a_ij / h2
                
                if j > 0:
                    a_ij = (a[i, j] + a[i, j - 1]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx - 1)
                    a_center += a_ij / h2
                else:
                    a_ij = a[i, j]
                    a_center += a_ij / h2
                
                if i < n - 1:
                    a_ij = (a[i, j] + a[i + 1, j]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx + n)
                    a_center += a_ij / h2
                else:
                    a_ij = a[i, j]
                    a_center += a_ij / h2
                
                if i > 0:
                    a_ij = (a[i, j] + a[i - 1, j]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx - n)
                    a_center += a_ij / h2
                else:
                    a_ij = a[i, j]
                    a_center += a_ij / h2
                
                data.append(a_center)
                row_ind.append(idx)
                col_ind.append(idx)
        
        A = sp.coo_matrix((data, (row_ind, col_ind)), shape=(n * n, n * n))
        return A.tocsr()


def generate_dataset(n_samples: int, resolution: int, noise_level: float = 0.0,
                     coeff_noise_level: float = 0.0,
                     grf_alpha: float = 2.0, grf_tau: float = 3.0) -> Dict[str, torch.Tensor]:
    """
    Generate inverse problem dataset.
    
    Args:
        noise_level: Standard deviation of Gaussian noise added to observations
        coeff_noise_level: Standard deviation of Gaussian noise added to coefficients
    
    Returns:
        dict with keys:
            'observation': Observed solution u (possibly noisy) (n_samples, 1, H, W)
            'source': True source term f (n_samples, 1, H, W)
            'coefficient': True coefficient a (n_samples, 1, H, W)
            'observation_noise': Noise added to observations (n_samples, 1, H, W)
            'coefficient_noise': Noise added to coefficients (n_samples, 1, H, W)
    """
    grf = GaussianRandomField(alpha=grf_alpha, tau=grf_tau, size=resolution)
    solver = PoissonSolver(size=resolution)
    
    observations = []
    sources = []
    coefficients = []
    obs_noise_list = []
    coeff_noise_list = []
    
    print(f"Generating {n_samples} inverse problem samples (obs noise: {noise_level}, coeff noise: {coeff_noise_level})...")
    for _ in tqdm(range(n_samples)):
        # Generate random parameters
        f = grf.sample(smooth_boundary=True)
        a_raw = grf.sample(smooth_boundary=False)
        a_clean = 1.0 + 0.5 * a_raw
        
        # Add coefficient noise
        if coeff_noise_level > 0:
            coeff_noise = np.random.randn(*a_clean.shape) * coeff_noise_level * np.std(a_clean)
            a = np.maximum(a_clean + coeff_noise, 0.1)  # Keep coefficient positive
        else:
            coeff_noise = np.zeros_like(a_clean)
            a = a_clean
        
        # Solve forward problem to get "true" solution
        u_true = solver.solve(f, a)
        
        # Add observation noise
        if noise_level > 0:
            obs_noise = np.random.randn(*u_true.shape) * noise_level * np.std(u_true)
            u_obs = u_true + obs_noise
        else:
            obs_noise = np.zeros_like(u_true)
            u_obs = u_true
        
        observations.append(u_obs)
        sources.append(f)
        coefficients.append(a_clean)  # Store clean coefficient
        obs_noise_list.append(obs_noise)
        coeff_noise_list.append(coeff_noise)
    
    # For inverse problem: input is observation, output is parameters
    dataset = {
        'observation': torch.tensor(np.array(observations), dtype=torch.float32).unsqueeze(1),
        'source': torch.tensor(np.array(sources), dtype=torch.float32).unsqueeze(1),
        'coefficient': torch.tensor(np.array(coefficients), dtype=torch.float32).unsqueeze(1),
        'observation_noise': torch.tensor(np.array(obs_noise_list), dtype=torch.float32).unsqueeze(1),
        'coefficient_noise': torch.tensor(np.array(coeff_noise_list), dtype=torch.float32).unsqueeze(1),
    }
    
    return dataset


def visualize_samples(dataset: Dict[str, torch.Tensor], n_vis: int = 4, save_path: str = None):
    """Visualize inverse problem: u_obs → (f, a)"""
    fig, axes = plt.subplots(n_vis, 3, figsize=(12, 4 * n_vis))
    
    for i in range(n_vis):
        u_obs = dataset['observation'][i, 0].numpy()
        f = dataset['source'][i, 0].numpy()
        a = dataset['coefficient'][i, 0].numpy()
        
        # Observation (input)
        im0 = axes[i, 0].imshow(u_obs, cmap='RdBu_r', aspect='auto')
        axes[i, 0].set_title(f'Observation u (INPUT, sample {i+1})')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Source term (target to infer)
        im1 = axes[i, 1].imshow(f, cmap='RdBu_r', aspect='auto')
        axes[i, 1].set_title(f'Source f (TARGET, sample {i+1})')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Coefficient (target to infer)
        im2 = axes[i, 2].imshow(a, cmap='viridis', aspect='auto')
        axes[i, 2].set_title(f'Coefficient a (TARGET, sample {i+1})')
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
    parser = argparse.ArgumentParser(description='Generate Poisson inverse problem dataset')
    parser.add_argument('--n_train', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--coeff_noise_level', type=float, default=0.0, help='Coefficient noise level')
    parser.add_argument('--resolution', type=int, default=64, help='Grid resolution')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Observation noise level')
    parser.add_argument('--grf_alpha', type=float, default=2.0, help='GRF smoothness')
    parser.add_argument('--grf_tau', type=float, default=3.0, help='GRF length scale')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    args = parser.parse_args()
    
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / 'datasets' / 'poisson' / 'inverse'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training set
    train_data = generate_dataset(
        n_samples=args.n_train,
        resolution=args.resolution,
        noise_level=args.noise_level,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Generate test set
    test_data = generate_dataset(
        n_samples=args.n_test,
        resolution=args.resolution,
        noise_level=args.noise_level,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Save datasets with resolution in filename
    train_path = output_dir / f'train_{args.resolution}.pt'
    test_path = output_dir / f'test_{args.resolution}.pt'
    
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    
    print(f"\n✓ Training data saved to {train_path}")
    print(f"  Shape: observation {train_data['observation'].shape}, "
          f"source {train_data['source'].shape}, "
          f"coefficient {train_data['coefficient'].shape}")
    
    print(f"\n✓ Test data saved to {test_path}")
    print(f"  Shape: observation {test_data['observation'].shape}, "
          f"source {test_data['source'].shape}, "
          f"coefficient {test_data['coefficient'].shape}")
    
    # Visualize
    if args.visualize:
        vis_path = output_dir / f'visualization_{args.resolution}.png'
        visualize_samples(train_data, n_vis=4, save_path=str(vis_path))


if __name__ == '__main__':
    main()
