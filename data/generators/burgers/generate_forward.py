"""
Burgers Forward Problem Generator
==================================

Forward Problem: initial_condition → solution_trajectory
    Given: u(x, t=0)
    Find: u(x, t) for t ∈ [0, T]
    
Solves: ∂u/∂t + u*∂u/∂x = ν*∂²u/∂x²

Usage:
    python generate_forward.py --n_train 5000 --n_test 1000 --resolution 256 --T_final 1.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import argparse
from tqdm import tqdm
from scipy.fft import fft, ifft


class GaussianRandomField1D:
    """Generate 1D Gaussian Random Fields."""
    
    def __init__(self, alpha: float = 2.0, tau: float = 5.0, size: int = 256):
        """
        Args:
            alpha: Smoothness parameter
            tau: Length scale
            size: Number of grid points
        """
        self.alpha = alpha
        self.tau = tau
        self.size = size
        
        # Wavenumbers
        k = np.fft.fftfreq(size, 1/size) * 2 * np.pi
        self.coef = tau**(alpha - 1) * (k**2 + tau**2)**(-alpha/2)
        self.coef[0] = 0  # Zero mean
    
    def sample(self) -> np.ndarray:
        """Generate one GRF sample."""
        xi = np.random.randn(self.size) + 1j * np.random.randn(self.size)
        xi[0] = 0  # Zero mean
        
        # Enforce Hermitian symmetry for real output
        if self.size % 2 == 0:
            xi[self.size // 2] = xi[self.size // 2].real
        
        for i in range(1, (self.size + 1) // 2):
            xi[self.size - i] = np.conj(xi[i])
        
        # Apply spectral coefficients
        u_hat = self.coef * xi * self.size
        
        # Transform to physical space
        u = np.fft.ifft(u_hat).real
        
        return u


class BurgersSolver:
    """
    Solve 1D Burgers equation using pseudo-spectral method.
    
    ∂u/∂t + u*∂u/∂x = ν*∂²u/∂x² on [0, 2π] with periodic BC
    """
    
    def __init__(self, viscosity: float = 0.01, size: int = 256):
        """
        Args:
            viscosity: Kinematic viscosity ν
            size: Number of spatial grid points
        """
        self.nu = viscosity
        self.size = size
        
        # Wavenumbers for derivatives
        self.k = np.fft.fftfreq(size, 1/size) * 2 * np.pi
        
        # Dealias mask (2/3 rule)
        self.dealias = np.abs(self.k) <= (2/3) * np.max(np.abs(self.k))
    
    def solve(self, u0: np.ndarray, T: float, dt: float = 1e-3) -> np.ndarray:
        """
        Solve Burgers equation.
        
        Args:
            u0: Initial condition (size,)
            T: Final time
            dt: Time step
            
        Returns:
            u: Solution at final time (size,)
        """
        u = u0.copy()
        t = 0.0
        
        # Time integration using RK4
        n_steps = int(T / dt)
        
        for _ in range(n_steps):
            k1 = self._rhs(u)
            k2 = self._rhs(u + dt * k1 / 2)
            k3 = self._rhs(u + dt * k2 / 2)
            k4 = self._rhs(u + dt * k3)
            
            u = u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            t += dt
        
        return u
    
    def solve_trajectory(self, u0: np.ndarray, T: float, n_snapshots: int = 50, 
                        dt: float = 1e-3) -> np.ndarray:
        """
        Solve and return trajectory at multiple time points.
        
        Returns:
            trajectory: (n_snapshots, size) array
        """
        u = u0.copy()
        trajectory = np.zeros((n_snapshots, self.size))
        
        n_steps = int(T / dt)
        save_every = max(1, n_steps // n_snapshots)
        
        trajectory[0] = u0
        snap_idx = 1
        
        for step in range(n_steps):
            k1 = self._rhs(u)
            k2 = self._rhs(u + dt * k1 / 2)
            k3 = self._rhs(u + dt * k2 / 2)
            k4 = self._rhs(u + dt * k3)
            
            u = u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            if (step + 1) % save_every == 0 and snap_idx < n_snapshots:
                trajectory[snap_idx] = u
                snap_idx += 1
        
        # Ensure last snapshot is the final state
        trajectory[-1] = u
        
        return trajectory
    
    def _rhs(self, u: np.ndarray) -> np.ndarray:
        """Compute RHS of Burgers equation."""
        # Transform to Fourier space
        u_hat = np.fft.fft(u)
        
        # Compute derivatives in Fourier space
        ux_hat = 1j * self.k * u_hat
        uxx_hat = -self.k**2 * u_hat
        
        # Transform back to physical space
        ux = np.fft.ifft(ux_hat).real
        uxx = np.fft.ifft(uxx_hat).real
        
        # Nonlinear term
        nonlinear = -u * ux
        
        # Viscous term
        viscous = self.nu * uxx
        
        return nonlinear + viscous


def generate_dataset(n_samples: int, resolution: int, T_final: float = 1.0,
                     viscosity: float = 0.01, n_snapshots: int = 50,
                     grf_alpha: float = 2.0, grf_tau: float = 5.0) -> Dict[str, torch.Tensor]:
    """
    Generate forward problem dataset.
    
    Returns:
        dict with keys:
            'initial': Initial condition u0 (n_samples, 1, resolution)
            'final': Solution at T (n_samples, 1, resolution)
            'trajectory': Full trajectory (n_samples, n_snapshots, resolution)
    """
    grf = GaussianRandomField1D(alpha=grf_alpha, tau=grf_tau, size=resolution)
    solver = BurgersSolver(viscosity=viscosity, size=resolution)
    
    initials = []
    finals = []
    trajectories = []
    
    print(f"Generating {n_samples} forward problem samples...")
    for _ in tqdm(range(n_samples)):
        # Generate random initial condition
        u0 = grf.sample()
        
        # Solve for trajectory
        trajectory = solver.solve_trajectory(u0, T=T_final, n_snapshots=n_snapshots)
        
        initials.append(u0)
        finals.append(trajectory[-1])
        trajectories.append(trajectory)
    
    dataset = {
        'initial': torch.tensor(np.array(initials), dtype=torch.float32).unsqueeze(1),
        'final': torch.tensor(np.array(finals), dtype=torch.float32).unsqueeze(1),
        'trajectory': torch.tensor(np.array(trajectories), dtype=torch.float32),
        'time': torch.linspace(0, T_final, n_snapshots),
        'viscosity': viscosity,
    }
    
    return dataset


def visualize_samples(dataset: Dict[str, torch.Tensor], n_vis: int = 4, save_path: str = None):
    """Visualize forward problem: u0 → u(t)"""
    fig, axes = plt.subplots(n_vis, 2, figsize=(14, 3 * n_vis))
    
    x = np.linspace(0, 2*np.pi, dataset['initial'].shape[-1])
    time = dataset['time'].numpy()
    
    for i in range(n_vis):
        u0 = dataset['initial'][i, 0].numpy()
        u_final = dataset['final'][i, 0].numpy()
        trajectory = dataset['trajectory'][i].numpy()
        
        # Initial condition
        axes[i, 0].plot(x, u0, 'b-', linewidth=2, label='Initial u(x, t=0)')
        axes[i, 0].plot(x, u_final, 'r-', linewidth=2, label=f'Final u(x, t={time[-1]:.2f})')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('u')
        axes[i, 0].set_title(f'Sample {i+1}: Initial → Final')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Space-time plot
        im = axes[i, 1].imshow(trajectory.T, aspect='auto', cmap='RdBu_r',
                               extent=[time[0], time[-1], x[0], x[-1]], origin='lower')
        axes[i, 1].set_xlabel('Time t')
        axes[i, 1].set_ylabel('Space x')
        axes[i, 1].set_title(f'Sample {i+1}: Evolution u(x,t)')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate Burgers forward problem dataset')
    parser.add_argument('--n_train', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--resolution', type=int, default=256, help='Spatial resolution')
    parser.add_argument('--T_final', type=float, default=1.0, help='Final time')
    parser.add_argument('--viscosity', type=float, default=0.01, help='Viscosity parameter')
    parser.add_argument('--n_snapshots', type=int, default=50, help='Number of time snapshots')
    parser.add_argument('--grf_alpha', type=float, default=2.0, help='GRF smoothness')
    parser.add_argument('--grf_tau', type=float, default=5.0, help='GRF length scale')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    args = parser.parse_args()
    
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / 'datasets' / 'burgers' / 'forward'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training set
    train_data = generate_dataset(
        n_samples=args.n_train,
        resolution=args.resolution,
        T_final=args.T_final,
        viscosity=args.viscosity,
        n_snapshots=args.n_snapshots,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Generate test set
    test_data = generate_dataset(
        n_samples=args.n_test,
        resolution=args.resolution,
        T_final=args.T_final,
        viscosity=args.viscosity,
        n_snapshots=args.n_snapshots,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Save datasets
    train_path = output_dir / 'train.pt'
    test_path = output_dir / 'test.pt'
    
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    
    print(f"\n✓ Training data saved to {train_path}")
    print(f"  Shape: initial {train_data['initial'].shape}, "
          f"final {train_data['final'].shape}, "
          f"trajectory {train_data['trajectory'].shape}")
    
    print(f"\n✓ Test data saved to {test_path}")
    print(f"  Shape: initial {test_data['initial'].shape}, "
          f"final {test_data['final'].shape}, "
          f"trajectory {test_data['trajectory'].shape}")
    
    # Visualize
    if args.visualize:
        vis_path = output_dir / 'visualization.png'
        visualize_samples(train_data, n_vis=4, save_path=str(vis_path))


if __name__ == '__main__':
    main()
