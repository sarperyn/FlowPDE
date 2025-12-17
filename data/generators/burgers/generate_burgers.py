"""
Unified Burgers Equation Dataset Generator
===========================================

Generate datasets for both forward and inverse problems:
    Forward:  u(x, 0) → u(x, t) for t ∈ [0, T]
    Inverse:  u(x, T) → u(x, 0)  [backward in time!]

Solves: ∂u/∂t + u*∂u/∂x = ν*∂²u/∂x² on [0, 2π] with periodic BC

Usage:
    # Forward problem with trajectory
    python generate_burgers.py --problem_type forward --n_train 5000 --n_test 1000 --resolution 256 --T_final 1.0
    
    # Inverse problem with noisy observations
    python generate_burgers.py --problem_type inverse --n_train 5000 --n_test 1000 --resolution 256 --obs_noise_level 0.01
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Literal
import argparse
from tqdm import tqdm


class GaussianRandomField1D:
    """Generate 1D Gaussian Random Fields."""
    
    def __init__(self, alpha: float = 2.0, tau: float = 5.0, size: int = 256):
        """
        Args:
            alpha: Smoothness parameter (higher = smoother)
            tau: Length scale parameter
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
        
        # Dealias mask (2/3 rule) - not strictly used but good practice
        self.dealias = np.abs(self.k) <= (2/3) * np.max(np.abs(self.k))
    
    def solve(self, u0: np.ndarray, T: float, dt: float = 1e-3) -> np.ndarray:
        """
        Solve Burgers equation to final time T.
        
        Args:
            u0: Initial condition (size,)
            T: Final time
            dt: Time step
            
        Returns:
            u: Solution at final time (size,)
        """
        u = u0.copy()
        n_steps = int(T / dt)
        
        # Time integration using RK4
        for _ in range(n_steps):
            k1 = self._rhs(u)
            k2 = self._rhs(u + dt * k1 / 2)
            k3 = self._rhs(u + dt * k2 / 2)
            k4 = self._rhs(u + dt * k3)
            
            u = u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return u
    
    def solve_trajectory(self, u0: np.ndarray, T: float, n_snapshots: int = 50, 
                        dt: float = 1e-3) -> np.ndarray:
        """
        Solve and return trajectory at multiple time points.
        
        Args:
            u0: Initial condition
            T: Final time
            n_snapshots: Number of snapshots to save
            dt: Time step
            
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
        """Compute RHS of Burgers equation: -u*∂u/∂x + ν*∂²u/∂x²"""
        # Transform to Fourier space
        u_hat = np.fft.fft(u)
        
        # Compute derivatives in Fourier space
        ux_hat = 1j * self.k * u_hat
        uxx_hat = -self.k**2 * u_hat
        
        # Transform back to physical space
        ux = np.fft.ifft(ux_hat).real
        uxx = np.fft.ifft(uxx_hat).real
        
        # Nonlinear term + Viscous term
        return -u * ux + self.nu * uxx


def generate_dataset(
    n_samples: int,
    resolution: int,
    problem_type: Literal['forward', 'inverse'],
    T_final: float = 1.0,
    viscosity: float = 0.01,
    n_snapshots: int = 50,
    initial_noise_level: float = 0.0,
    obs_noise_level: float = 0.0,
    grf_alpha: float = 2.0,
    grf_tau: float = 5.0
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset for forward or inverse Burgers problem.
    
    Args:
        problem_type: 'forward' or 'inverse'
        n_snapshots: Number of time snapshots (forward only)
        initial_noise_level: Std of noise added to initial condition (forward only)
        obs_noise_level: Std of noise added to observations (inverse only)
        
    Returns:
        For forward problem:
            'initial': u(x, 0) (input)
            'final': u(x, T) (output)
            'trajectory': u(x, t) for all t (output, n_snapshots × resolution)
            'initial_noise', 'time', 'viscosity'
            
        For inverse problem:
            'observation': u(x, T) (input, possibly noisy)
            'initial': u(x, 0) (output/target)
            'observation_noise', 'time', 'viscosity'
    """
    grf = GaussianRandomField1D(alpha=grf_alpha, tau=grf_tau, size=resolution)
    solver = BurgersSolver(viscosity=viscosity, size=resolution)
    
    # Storage
    initials = []
    finals = []
    trajectories = [] if problem_type == 'forward' else None
    initial_noise_list = []
    obs_noise_list = []
    
    # Determine noise settings
    apply_initial_noise = (problem_type == 'forward' and initial_noise_level > 0)
    apply_obs_noise = (problem_type == 'inverse' and obs_noise_level > 0)
    
    desc = f"{problem_type} (ν={viscosity}, T={T_final})"
    if problem_type == 'forward':
        desc += f", init_noise={initial_noise_level}"
    else:
        desc += f", obs_noise={obs_noise_level}"
    
    print(f"Generating {n_samples} {desc} samples...")
    for _ in tqdm(range(n_samples)):
        # Generate random initial condition
        u0_clean = grf.sample()
        
        # Add initial condition noise (for forward problem)
        if apply_initial_noise:
            initial_noise = np.random.randn(*u0_clean.shape) * initial_noise_level * np.std(u0_clean)
            u0 = u0_clean + initial_noise
        else:
            initial_noise = np.zeros_like(u0_clean)
            u0 = u0_clean
        
        # Solve forward problem
        if problem_type == 'forward':
            # Get full trajectory
            trajectory = solver.solve_trajectory(u0, T=T_final, n_snapshots=n_snapshots)
            u_final = trajectory[-1]
            trajectories.append(trajectory)
        else:
            # Just solve to final time
            u_final = solver.solve(u0, T=T_final)
        
        # Add observation noise (for inverse problem)
        if apply_obs_noise:
            obs_noise = np.random.randn(*u_final.shape) * obs_noise_level * np.std(u_final)
            u_obs = u_final + obs_noise
        else:
            obs_noise = np.zeros_like(u_final)
            u_obs = u_final
        
        initials.append(u0)
        finals.append(u_obs)
        initial_noise_list.append(initial_noise)
        obs_noise_list.append(obs_noise)
    
    # Build dataset dictionary
    if problem_type == 'forward':
        dataset = {
            'initial': torch.tensor(np.array(initials), dtype=torch.float32).unsqueeze(1),
            'final': torch.tensor(np.array(finals), dtype=torch.float32).unsqueeze(1),
            'trajectory': torch.tensor(np.array(trajectories), dtype=torch.float32),
            'initial_noise': torch.tensor(np.array(initial_noise_list), dtype=torch.float32).unsqueeze(1),
            'time': torch.linspace(0, T_final, n_snapshots),
            'viscosity': viscosity,
        }
    else:  # inverse
        dataset = {
            'observation': torch.tensor(np.array(finals), dtype=torch.float32).unsqueeze(1),
            'initial': torch.tensor(np.array(initials), dtype=torch.float32).unsqueeze(1),
            'observation_noise': torch.tensor(np.array(obs_noise_list), dtype=torch.float32).unsqueeze(1),
            'time': T_final,
            'viscosity': viscosity,
        }
    
    return dataset


def visualize_samples(
    dataset: Dict[str, torch.Tensor],
    problem_type: Literal['forward', 'inverse'],
    n_vis: int = 4,
    save_path: str = None
):
    """Visualize dataset samples."""
    x = np.linspace(0, 2*np.pi, dataset['initial' if problem_type == 'inverse' else 'initial'].shape[-1])
    
    if problem_type == 'forward':
        # Forward: Show initial → final with trajectory
        fig, axes = plt.subplots(n_vis, 2, figsize=(14, 3 * n_vis))
        time = dataset['time'].numpy()
        
        for i in range(n_vis):
            u0 = dataset['initial'][i, 0].numpy()
            u_final = dataset['final'][i, 0].numpy()
            trajectory = dataset['trajectory'][i].numpy()
            
            # Left: Initial vs Final
            axes[i, 0].plot(x, u0, 'b-', linewidth=2, label='Initial u(x, t=0)')
            axes[i, 0].plot(x, u_final, 'r-', linewidth=2, label=f'Final u(x, t={time[-1]:.2f})')
            axes[i, 0].set_xlabel('x')
            axes[i, 0].set_ylabel('u')
            axes[i, 0].set_title(f'Sample {i+1}: Initial → Final')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Right: Space-time plot
            im = axes[i, 1].imshow(trajectory.T, aspect='auto', cmap='RdBu_r',
                                   extent=[time[0], time[-1], x[0], x[-1]], origin='lower')
            axes[i, 1].set_xlabel('Time t')
            axes[i, 1].set_ylabel('Space x')
            axes[i, 1].set_title(f'Sample {i+1}: Evolution u(x,t)')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
    
    else:  # inverse
        # Inverse: Show observation → initial (backward in time)
        fig, axes = plt.subplots(n_vis, 1, figsize=(12, 3 * n_vis))
        if n_vis == 1:
            axes = [axes]
        
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
    parser = argparse.ArgumentParser(description='Generate Burgers equation dataset')
    
    # Problem type (REQUIRED)
    parser.add_argument('--problem_type', type=str, required=True,
                        choices=['forward', 'inverse'],
                        help='Type of problem: forward or inverse')
    
    # Dataset size
    parser.add_argument('--n_train', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Spatial resolution')
    
    # Physical parameters
    parser.add_argument('--T_final', type=float, default=1.0,
                        help='Final time')
    parser.add_argument('--viscosity', type=float, default=0.01,
                        help='Viscosity parameter ν')
    
    # Forward problem specific
    parser.add_argument('--n_snapshots', type=int, default=50,
                        help='Number of time snapshots (forward only)')
    parser.add_argument('--initial_noise_level', type=float, default=0.0,
                        help='Initial condition noise level (forward only)')
    
    # Inverse problem specific
    parser.add_argument('--obs_noise_level', type=float, default=0.0,
                        help='Observation noise level (inverse only)')
    
    # GRF parameters
    parser.add_argument('--grf_alpha', type=float, default=2.0,
                        help='GRF smoothness parameter')
    parser.add_argument('--grf_tau', type=float, default=5.0,
                        help='GRF length scale')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    
    args = parser.parse_args()
    
    # Output directory based on problem type
    output_dir = Path(__file__).parent.parent.parent / 'datasets' / 'burgers' / args.problem_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training set
    train_data = generate_dataset(
        n_samples=args.n_train,
        resolution=args.resolution,
        problem_type=args.problem_type,
        T_final=args.T_final,
        viscosity=args.viscosity,
        n_snapshots=args.n_snapshots,
        initial_noise_level=args.initial_noise_level,
        obs_noise_level=args.obs_noise_level,
        grf_alpha=args.grf_alpha,
        grf_tau=args.grf_tau
    )
    
    # Generate test set
    test_data = generate_dataset(
        n_samples=args.n_test,
        resolution=args.resolution,
        problem_type=args.problem_type,
        T_final=args.T_final,
        viscosity=args.viscosity,
        n_snapshots=args.n_snapshots,
        initial_noise_level=args.initial_noise_level,
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
    print(f"Viscosity: {args.viscosity}, Final Time: {args.T_final}")
    print(f"{'='*60}")
    
    print(f"\n✓ Training data saved to {train_path}")
    print(f"  Keys: {list(train_data.keys())}")
    for key, val in train_data.items():
        if isinstance(val, torch.Tensor):
            print(f"  - {key}: {val.shape}")
        else:
            print(f"  - {key}: {val}")
    
    print(f"\n✓ Test data saved to {test_path}")
    print(f"  Keys: {list(test_data.keys())}")
    for key, val in test_data.items():
        if isinstance(val, torch.Tensor):
            print(f"  - {key}: {val.shape}")
        else:
            print(f"  - {key}: {val}")
    
    # Summary statistics
    print(f"\n📊 Dataset Statistics:")
    if args.problem_type == 'forward':
        print(f"  Initial range: [{train_data['initial'].min():.3f}, {train_data['initial'].max():.3f}]")
        print(f"  Final range: [{train_data['final'].min():.3f}, {train_data['final'].max():.3f}]")
        print(f"  Trajectory shape: {train_data['trajectory'].shape}")
    else:
        print(f"  Observation range: [{train_data['observation'].min():.3f}, {train_data['observation'].max():.3f}]")
        print(f"  Initial range: [{train_data['initial'].min():.3f}, {train_data['initial'].max():.3f}]")
    
    # Visualize
    if args.visualize:
        vis_path = output_dir / f'visualization_{args.resolution}.png'
        visualize_samples(train_data, args.problem_type, n_vis=4, save_path=str(vis_path))


if __name__ == '__main__':
    main()
