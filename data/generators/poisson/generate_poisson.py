"""
Unified Poisson Problem Dataset Generator
==========================================

Generate datasets for both forward and inverse problems:
    Forward:  (f, a) → u
    Inverse:  u → (f, a)

Usage:
    # Forward problem
    python generate_poisson.py --problem_type forward --n_train 5000 --n_test 1000 --resolution 64
    
    # Inverse problem with noisy observations
    python generate_poisson.py --problem_type inverse --n_train 5000 --n_test 1000 --resolution 64 --noise_level 0.01
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Literal
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
        A_sparse = self._build_matrix(a)
        f_vec = f.flatten()
        u_vec = spla.spsolve(A_sparse, f_vec)
        u = u_vec.reshape(self.size, self.size)
        return u
    
    def _build_matrix(self, a: np.ndarray) -> sp.csr_matrix:
        """Build finite difference matrix using 5-point stencil."""
        n = self.size
        h2 = self.h ** 2
        
        data = []
        row_ind = []
        col_ind = []
        
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                a_center = 0.0
                
                # East neighbor
                if j < n - 1:
                    a_ij = (a[i, j] + a[i, j + 1]) / 2
                    data.append(-a_ij / h2)
                    row_ind.append(idx)
                    col_ind.append(idx + 1)
                    a_center += a_ij / h2
                else:
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


def generate_dataset(
    n_samples: int,
    resolution: int,
    problem_type: Literal['forward', 'inverse'],
    source_noise_level: float = 0.0,
    coeff_noise_level: float = 0.0,
    obs_noise_level: float = 0.0,
    grf_alpha: float = 2.0,
    grf_tau: float = 3.0
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset for forward or inverse problem.
    
    Args:
        problem_type: 'forward' or 'inverse'
        source_noise_level: Std of noise added to source term (forward only)
        coeff_noise_level: Std of noise added to coefficient
        obs_noise_level: Std of noise added to observations (inverse only)
        
    Returns:
        For forward problem:
            'source': f (input)
            'coefficient': a (input)
            'solution': u (output)
            'source_noise', 'coefficient_noise'
            
        For inverse problem:
            'observation': u (input, possibly noisy)
            'source': f (output/target)
            'coefficient': a (output/target)
            'observation_noise', 'coefficient_noise'
    """
    grf = GaussianRandomField(alpha=grf_alpha, tau=grf_tau, size=resolution)
    solver = PoissonSolver(size=resolution)
    
    # Storage arrays
    sources = []
    coefficients = []
    solutions = []
    source_noise_list = []
    coeff_noise_list = []
    obs_noise_list = []
    
    # Determine which noise applies
    if problem_type == 'forward':
        apply_source_noise = source_noise_level > 0
        apply_obs_noise = False
        desc = f"forward (source_noise={source_noise_level}, coeff_noise={coeff_noise_level})"
    else:  # inverse
        apply_source_noise = False
        apply_obs_noise = obs_noise_level > 0
        desc = f"inverse (obs_noise={obs_noise_level}, coeff_noise={coeff_noise_level})"
    
    print(f"Generating {n_samples} {desc} samples...")
    for _ in tqdm(range(n_samples)):
        # Generate clean source term
        f_clean = grf.sample(smooth_boundary=True)
        
        # Add source noise (for forward problem)
        if apply_source_noise:
            source_noise = np.random.randn(*f_clean.shape) * source_noise_level * np.std(f_clean)
            f = f_clean + source_noise
        else:
            source_noise = np.zeros_like(f_clean)
            f = f_clean
        
        # Generate clean coefficient
        a_raw = grf.sample(smooth_boundary=False)
        a_clean = 1.0 + 0.5 * a_raw  # Ensure a > 0.5
        
        # Add coefficient noise
        if coeff_noise_level > 0:
            coeff_noise = np.random.randn(*a_clean.shape) * coeff_noise_level * np.std(a_clean)
            a = np.maximum(a_clean + coeff_noise, 0.1)  # Keep positive
        else:
            coeff_noise = np.zeros_like(a_clean)
            a = a_clean
        
        # Solve PDE
        u_true = solver.solve(f, a)
        
        # Add observation noise (for inverse problem)
        if apply_obs_noise:
            obs_noise = np.random.randn(*u_true.shape) * obs_noise_level * np.std(u_true)
            u = u_true + obs_noise
        else:
            obs_noise = np.zeros_like(u_true)
            u = u_true
        
        sources.append(f)
        coefficients.append(a_clean)  # Always store clean coefficient
        solutions.append(u)
        source_noise_list.append(source_noise)
        coeff_noise_list.append(coeff_noise)
        obs_noise_list.append(obs_noise)
    
    # Convert to tensors (N, 1, H, W)
    if problem_type == 'forward':
        dataset = {
            'source': torch.tensor(np.array(sources), dtype=torch.float32).unsqueeze(1),
            'coefficient': torch.tensor(np.array(coefficients), dtype=torch.float32).unsqueeze(1),
            'solution': torch.tensor(np.array(solutions), dtype=torch.float32).unsqueeze(1),
            'source_noise': torch.tensor(np.array(source_noise_list), dtype=torch.float32).unsqueeze(1),
            'coefficient_noise': torch.tensor(np.array(coeff_noise_list), dtype=torch.float32).unsqueeze(1),
        }
    else:  # inverse
        dataset = {
            'observation': torch.tensor(np.array(solutions), dtype=torch.float32).unsqueeze(1),
            'source': torch.tensor(np.array(sources), dtype=torch.float32).unsqueeze(1),
            'coefficient': torch.tensor(np.array(coefficients), dtype=torch.float32).unsqueeze(1),
            'observation_noise': torch.tensor(np.array(obs_noise_list), dtype=torch.float32).unsqueeze(1),
            'coefficient_noise': torch.tensor(np.array(coeff_noise_list), dtype=torch.float32).unsqueeze(1),
        }
    
    return dataset


def visualize_samples(
    dataset: Dict[str, torch.Tensor],
    problem_type: Literal['forward', 'inverse'],
    n_vis: int = 4,
    save_path: str = None
):
    """Visualize dataset samples."""
    fig, axes = plt.subplots(n_vis, 3, figsize=(12, 4 * n_vis))
    
    for i in range(n_vis):
        if problem_type == 'forward':
            # Forward: (f, a) → u
            f = dataset['source'][i, 0].numpy()
            a = dataset['coefficient'][i, 0].numpy()
            u = dataset['solution'][i, 0].numpy()
            
            titles = [
                f'Source f (INPUT, sample {i+1})',
                f'Coefficient a (INPUT, sample {i+1})',
                f'Solution u (OUTPUT, sample {i+1})'
            ]
            data = [f, a, u]
            cmaps = ['RdBu_r', 'viridis', 'RdBu_r']
            
        else:  # inverse
            # Inverse: u → (f, a)
            u = dataset['observation'][i, 0].numpy()
            f = dataset['source'][i, 0].numpy()
            a = dataset['coefficient'][i, 0].numpy()
            
            titles = [
                f'Observation u (INPUT, sample {i+1})',
                f'Source f (TARGET, sample {i+1})',
                f'Coefficient a (TARGET, sample {i+1})'
            ]
            data = [u, f, a]
            cmaps = ['RdBu_r', 'RdBu_r', 'viridis']
        
        for j, (d, title, cmap) in enumerate(zip(data, titles, cmaps)):
            im = axes[i, j].imshow(d, cmap=cmap, aspect='auto')
            axes[i, j].set_title(title)
            axes[i, j].axis('off')
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate Poisson problem dataset')
    
    # Problem type (REQUIRED)
    parser.add_argument('--problem_type', type=str, required=True, 
                        choices=['forward', 'inverse'],
                        help='Type of problem: forward or inverse')
    
    # Dataset size
    parser.add_argument('--n_train', type=int, default=5000, 
                        help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=1000, 
                        help='Number of test samples')
    parser.add_argument('--resolution', type=int, default=64, 
                        help='Grid resolution')
    
    # Noise levels
    parser.add_argument('--source_noise_level', type=float, default=0.0, 
                        help='Source noise level (forward problem only)')
    parser.add_argument('--coeff_noise_level', type=float, default=0.0, 
                        help='Coefficient noise level')
    parser.add_argument('--obs_noise_level', type=float, default=0.0, 
                        help='Observation noise level (inverse problem only)')
    
    # GRF parameters
    parser.add_argument('--grf_alpha', type=float, default=2.0, 
                        help='GRF smoothness parameter')
    parser.add_argument('--grf_tau', type=float, default=3.0, 
                        help='GRF length scale')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualization')
    
    args = parser.parse_args()
    
    # Output directory based on problem type
    output_dir = Path(__file__).parent.parent.parent / 'datasets' / 'poisson' / args.problem_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training set
    train_data = generate_dataset(
        n_samples=args.n_train,
        resolution=args.resolution,
        problem_type=args.problem_type,
        source_noise_level=args.source_noise_level,
        coeff_noise_level=args.coeff_noise_level,
        obs_noise_level=args.obs_noise_level,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Generate test set
    test_data = generate_dataset(
        n_samples=args.n_test,
        resolution=args.resolution,
        problem_type=args.problem_type,
        source_noise_level=args.source_noise_level,
        coeff_noise_level=args.coeff_noise_level,
        obs_noise_level=args.obs_noise_level,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Save datasets
    train_path = output_dir / f'train_{args.resolution}.pt'
    test_path = output_dir / f'test_{args.resolution}.pt'
    
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Problem Type: {args.problem_type.upper()}")
    print(f"{'='*60}")
    
    print(f"\n✓ Training data saved to {train_path}")
    print(f"  Keys: {list(train_data.keys())}")
    for key, val in train_data.items():
        print(f"  - {key}: {val.shape}")
    
    print(f"\n✓ Test data saved to {test_path}")
    print(f"  Keys: {list(test_data.keys())}")
    for key, val in test_data.items():
        print(f"  - {key}: {val.shape}")
    
    # Summary statistics
    if args.problem_type == 'forward':
        print(f"\n📊 Dataset Statistics:")
        print(f"  Source range: [{train_data['source'].min():.3f}, {train_data['source'].max():.3f}]")
        print(f"  Coefficient range: [{train_data['coefficient'].min():.3f}, {train_data['coefficient'].max():.3f}]")
        print(f"  Solution range: [{train_data['solution'].min():.3f}, {train_data['solution'].max():.3f}]")
    else:
        print(f"\n📊 Dataset Statistics:")
        print(f"  Observation range: [{train_data['observation'].min():.3f}, {train_data['observation'].max():.3f}]")
        print(f"  Source range: [{train_data['source'].min():.3f}, {train_data['source'].max():.3f}]")
        print(f"  Coefficient range: [{train_data['coefficient'].min():.3f}, {train_data['coefficient'].max():.3f}]")
    
    # Visualize
    if args.visualize:
        vis_path = output_dir / f'visualization_{args.resolution}.png'
        visualize_samples(train_data, args.problem_type, n_vis=4, save_path=str(vis_path))


if __name__ == '__main__':
    main()
