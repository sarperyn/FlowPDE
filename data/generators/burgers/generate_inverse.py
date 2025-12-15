"""
Burgers Inverse Problem Generator
==================================

Inverse Problem: final_state → initial_condition
    Given: Observed solution u(x, T) at final time (possibly with noise)
    Infer: Initial condition u(x, 0)

This is a classic inverse problem: "What initial condition led to this observation?"

Usage:
    python generate_inverse.py --n_train 5000 --n_test 1000 --resolution 256 --noise_level 0.01
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import argparse
from tqdm import tqdm


class GaussianRandomField1D:
    """Generate 1D Gaussian Random Fields."""
    
    def __init__(self, alpha: float = 2.0, tau: float = 5.0, size: int = 256):
        self.alpha = alpha
        self.tau = tau
        self.size = size
        
        k = np.fft.fftfreq(size, 1/size) * 2 * np.pi
        self.coef = tau**(alpha - 1) * (k**2 + tau**2)**(-alpha/2)
        self.coef[0] = 0
    
    def sample(self) -> np.ndarray:
        xi = np.random.randn(self.size) + 1j * np.random.randn(self.size)
        xi[0] = 0
        
        if self.size % 2 == 0:
            xi[self.size // 2] = xi[self.size // 2].real
        
        for i in range(1, (self.size + 1) // 2):
            xi[self.size - i] = np.conj(xi[i])
        
        u_hat = self.coef * xi * self.size
        u = np.fft.ifft(u_hat).real
        
        return u


class BurgersSolver:
    """Solve 1D Burgers equation using pseudo-spectral method."""
    
    def __init__(self, viscosity: float = 0.01, size: int = 256):
        self.nu = viscosity
        self.size = size
        self.k = np.fft.fftfreq(size, 1/size) * 2 * np.pi
        self.dealias = np.abs(self.k) <= (2/3) * np.max(np.abs(self.k))
    
    def solve(self, u0: np.ndarray, T: float, dt: float = 1e-3) -> np.ndarray:
        """Solve to final time T."""
        u = u0.copy()
        n_steps = int(T / dt)
        
        for _ in range(n_steps):
            k1 = self._rhs(u)
            k2 = self._rhs(u + dt * k1 / 2)
            k3 = self._rhs(u + dt * k2 / 2)
            k4 = self._rhs(u + dt * k3)
            
            u = u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return u
    
    def _rhs(self, u: np.ndarray) -> np.ndarray:
        u_hat = np.fft.fft(u)
        ux_hat = 1j * self.k * u_hat
        uxx_hat = -self.k**2 * u_hat
        
        ux = np.fft.ifft(ux_hat).real
        uxx = np.fft.ifft(uxx_hat).real
        
        return -u * ux + self.nu * uxx


def generate_dataset(n_samples: int, resolution: int, T_final: float = 1.0,
                     viscosity: float = 0.01, noise_level: float = 0.0,
                     grf_alpha: float = 2.0, grf_tau: float = 5.0) -> Dict[str, torch.Tensor]:
    """
    Generate inverse problem dataset.
    
    Args:
        noise_level: Standard deviation of noise added to final observation
    
    Returns:
        dict with keys:
            'observation': Observed final state u(T) (n_samples, 1, resolution)
            'initial': True initial condition u(0) (n_samples, 1, resolution)
    """
    grf = GaussianRandomField1D(alpha=grf_alpha, tau=grf_tau, size=resolution)
    solver = BurgersSolver(viscosity=viscosity, size=resolution)
    
    observations = []
    initials = []
    
    print(f"Generating {n_samples} inverse problem samples (noise level: {noise_level})...")
    for _ in tqdm(range(n_samples)):
        # Generate random initial condition (this is what we want to infer)
        u0 = grf.sample()
        
        # Solve forward to get final state
        u_final = solver.solve(u0, T=T_final)
        
        # Add observation noise
        if noise_level > 0:
            noise = np.random.randn(*u_final.shape) * noise_level * np.std(u_final)
            u_obs = u_final + noise
        else:
            u_obs = u_final
        
        observations.append(u_obs)
        initials.append(u0)
    
    # For inverse problem: observation is input, initial condition is target
    dataset = {
        'observation': torch.tensor(np.array(observations), dtype=torch.float32).unsqueeze(1),
        'initial': torch.tensor(np.array(initials), dtype=torch.float32).unsqueeze(1),
        'time': T_final,
        'viscosity': viscosity,
    }
    
    return dataset


def visualize_samples(dataset: Dict[str, torch.Tensor], n_vis: int = 4, save_path: str = None):
    """Visualize inverse problem: u(T) → u(0)"""
    fig, axes = plt.subplots(n_vis, 1, figsize=(12, 3 * n_vis))
    
    if n_vis == 1:
        axes = [axes]
    
    x = np.linspace(0, 2*np.pi, dataset['observation'].shape[-1])
    T = dataset['time']
    
    for i in range(n_vis):
        u_obs = dataset['observation'][i, 0].numpy()
        u0 = dataset['initial'][i, 0].numpy()
        
        axes[i].plot(x, u_obs, 'r-', linewidth=2, label=f'Observation u(x, T={T:.2f}) [INPUT]')
        axes[i].plot(x, u0, 'b--', linewidth=2, label='Initial u(x, 0) [TARGET TO INFER]')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('u')
        axes[i].set_title(f'Sample {i+1}: Inverse Problem - Given u(T), infer u(0)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add annotation
        axes[i].text(0.5, 0.95, '← Flow learns to go backward in time', 
                    transform=axes[i].transAxes, fontsize=10, 
                    verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate Burgers inverse problem dataset')
    parser.add_argument('--n_train', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--resolution', type=int, default=256, help='Spatial resolution')
    parser.add_argument('--T_final', type=float, default=1.0, help='Final observation time')
    parser.add_argument('--viscosity', type=float, default=0.01, help='Viscosity parameter')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Observation noise level')
    parser.add_argument('--grf_alpha', type=float, default=2.0, help='GRF smoothness')
    parser.add_argument('--grf_tau', type=float, default=5.0, help='GRF length scale')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    args = parser.parse_args()
    
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / 'datasets' / 'burgers' / 'inverse'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training set
    train_data = generate_dataset(
        n_samples=args.n_train,
        resolution=args.resolution,
        T_final=args.T_final,
        viscosity=args.viscosity,
        noise_level=args.noise_level,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Generate test set
    test_data = generate_dataset(
        n_samples=args.n_test,
        resolution=args.resolution,
        T_final=args.T_final,
        viscosity=args.viscosity,
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
          f"initial {train_data['initial'].shape}")
    
    print(f"\n✓ Test data saved to {test_path}")
    print(f"  Shape: observation {test_data['observation'].shape}, "
          f"initial {test_data['initial'].shape}")
    
    # Visualize
    if args.visualize:
        vis_path = output_dir / f'visualization_{args.resolution}.png'
        visualize_samples(train_data, n_vis=4, save_path=str(vis_path))


if __name__ == '__main__':
    main()
