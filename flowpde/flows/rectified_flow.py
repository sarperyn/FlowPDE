"""
Rectified Flow for learning straight transport paths.

Rectified flow iteratively straightens the transport paths between distributions,
leading to faster sampling and better generation quality.
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Optional, Tuple

from flowpde.core.base_flow import BaseFlow


class RectifiedFlow(BaseFlow):
    """
    Rectified Flow for learning straight transport paths.
    
    Rectified flow learns to transport between distributions along straight lines
    through an iterative straightening procedure. This leads to:
    - Faster sampling (fewer steps needed)
    - Better generation quality
    - More stable training
    
    The key idea is to minimize:
        L = E[||v(x_t, t) - (x_1 - x_0)||^2]
    
    where x_t = (1-t)x_0 + t*x_1 and the target velocity is constant.
    
    Args:
        model: Neural network predicting velocity v(x, condition, t)
        reflow_iterations: Number of reflow iterations for straightening
        time_sampling: Distribution for sampling time ('uniform' or 'logit_normal')
        
    References:
        - Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer
          Data with Rectified Flow", ICLR 2023
    """
    
    def __init__(
        self,
        model: nn.Module,
        reflow_iterations: int = 1,
        time_sampling: str = 'uniform'
    ):
        super().__init__(model)
        self.reflow_iterations = reflow_iterations
        self.time_sampling = time_sampling
        self.current_iteration = 0
        
        if time_sampling not in ['uniform', 'logit_normal']:
            raise ValueError(f"Unknown time sampling: {time_sampling}")
    
    def sample_base_distribution(
        self,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> Tensor:
        """Sample from base distribution (standard Gaussian)."""
        return torch.randn(*shape, device=device)
    
    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Sample time values according to specified distribution.
        
        Args:
            batch_size: Number of samples
            device: Device for computation
        
        Returns:
            Time values in [0, 1]
        """
        if self.time_sampling == 'uniform':
            return torch.rand(batch_size, 1, device=device)
        elif self.time_sampling == 'logit_normal':
            # Logit-normal distribution concentrates samples near 0 and 1
            z = torch.randn(batch_size, 1, device=device)
            t = torch.sigmoid(z)
            return t
        else:
            raise ValueError(f"Unknown time sampling: {self.time_sampling}")
    
    def compute_straight_path(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute interpolation along straight path.
        
        Args:
            x_0: Source samples
            x_1: Target samples  
            t: Time values in [0, 1]
        
        Returns:
            x_t: Interpolated samples
            v_t: Target velocity (constant along path)
        """
        # Expand t to match dimensions
        t_expanded = t.view(-1, *([1] * (x_0.dim() - 1)))
        
        # Linear interpolation
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # Constant velocity along straight line
        v_t = x_1 - x_0
        
        return x_t, v_t
    
    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        """
        Compute rectified flow loss.
        
        The loss encourages the model to predict constant velocities
        along straight paths from noise to data.
        
        Args:
            batch: Dictionary with 'u' (target) and 'f' (condition)
        
        Returns:
            MSE loss between predicted and target velocities
        """
        # Extract data
        x_1 = batch["u"].flatten(start_dim=1).to(self.model_device)
        condition = batch["f"].flatten(start_dim=1).to(self.model_device)
        batch_size = x_1.shape[0]
        
        # Sample base distribution
        x_0 = self.sample_base_distribution(x_1.shape, self.model_device)
        
        # Sample time
        t = self.sample_time(batch_size, self.model_device)
        
        # Compute straight path interpolation
        x_t, v_target = self.compute_straight_path(x_0, x_1, t)
        
        # Predict velocity
        v_pred = self.model(x_t, condition, t)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
    
    def reflow(
        self,
        data_loader,
        optimizer,
        epochs: int = 10,
        verbose: bool = True
    ):
        """
        Perform reflow iteration to straighten paths.
        
        After initial training, reflow uses the current model to generate
        new training pairs (x_0', x_1') and retrains, leading to straighter paths.
        
        Args:
            data_loader: Training data loader
            optimizer: Optimizer for retraining
            epochs: Number of epochs for reflow
            verbose: Whether to print progress
        """
        if self.current_iteration >= self.reflow_iterations:
            if verbose:
                print(f"Already completed {self.reflow_iterations} reflow iterations")
            return
        
        self.current_iteration += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Reflow Iteration {self.current_iteration}/{self.reflow_iterations}")
            print(f"{'='*60}")
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch in data_loader:
                # Move to device
                batch = {k: v.to(self.model_device) for k, v in batch.items()}
                
                # Generate new training pairs using current model
                with torch.no_grad():
                    condition = batch["f"].flatten(start_dim=1)
                    batch_size = condition.shape[0]
                    dim = condition.shape[1]
                    
                    # Sample new x_0 from base
                    x_0_new = self.sample_base_distribution(
                        (batch_size, dim),
                        self.model_device
                    )
                    
                    # Generate x_1 using current model
                    x_1_new = self.sample(
                        condition=condition,
                        x_init=x_0_new,
                        n_steps=50
                    )
                
                # Create new batch with generated pairs
                new_batch = {
                    "u": x_1_new,
                    "f": batch["f"]
                }
                
                # Train on new pairs
                optimizer.zero_grad()
                loss = self.compute_loss(new_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def sample(
        self,
        condition: Tensor,
        n_steps: int = 50,
        solver: str = 'dopri5',
        x_init: Optional[Tensor] = None,
        **solver_kwargs
    ) -> Tensor:
        """
        Sample from the learned distribution.
        
        Rectified flow often requires fewer steps than other methods
        due to straighter transport paths.
        
        Args:
            condition: Conditioning tensor
            n_steps: Number of integration steps
            solver: ODE solver name
            x_init: Optional initial noise
            **solver_kwargs: Additional solver arguments
        
        Returns:
            Generated samples
        """
        
        condition = condition.flatten(start_dim=1).to(self.model_device)
        batch_size = condition.shape[0]
        dim = condition.shape[1]
        
        # Sample from base distribution
        if x_init is None:
            x_0 = self.sample_base_distribution((batch_size, dim), self.model_device)
        else:
            x_0 = x_init.flatten(start_dim=1).to(self.model_device)
        
        # Create ODE solver
        from flowpde.solvers import ODEFlowSolver
        ode_solver = ODEFlowSolver(model=self.model, method=solver, **solver_kwargs)
        
        # Integrate from t=0 to t=1
        samples = ode_solver.sample(
            condition=condition,
            x_init=x_0,
            n_steps=n_steps
        )
        
        return samples
    
    def log_prob(
        self,
        x: Tensor,
        condition: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Compute log probability.
        
        Note: Rectified flow doesn't directly compute log probabilities.
        Use CNF for density estimation.
        """
        raise NotImplementedError(
            "Rectified flow doesn't compute log probabilities. "
            "Use ContinuousNormalizingFlow for density estimation."
        )
    
    def forward_transform(
        self,
        x: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Forward transformation: data -> latent (backward ODE).
        
        Note: This requires integrating the ODE backward in time.
        """
        raise NotImplementedError(
            "Forward transform requires backward ODE integration. "
            "Use CNF for bidirectional transformations."
        )
    
    def inverse_transform(
        self,
        z: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Inverse transformation: latent -> data (forward ODE).
        
        This is equivalent to the sample() method.
        """
        return self.sample(condition=condition, x_init=z, **kwargs)
    
    def get_config(self) -> Dict:
        """Return configuration dictionary."""
        return {
            'flow_type': 'rectified_flow',
            'reflow_iterations': self.reflow_iterations,
            'current_iteration': self.current_iteration,
            'time_sampling': self.time_sampling,
            'model_type': self.model.__class__.__name__
        }
    
    def estimate_straightness(
        self,
        data_loader,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Estimate how straight the learned transport paths are.
        
        Straighter paths should have more uniform velocities along trajectories.
        
        Args:
            data_loader: Data loader for evaluation
            n_samples: Number of samples to evaluate
        
        Returns:
            Dictionary with straightness metrics
        """
        self.model.eval()
        
        velocity_stds = []
        path_lengths = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(velocity_stds) >= n_samples:
                    break
                
                batch = {k: v.to(self.model_device) for k, v in batch.items()}
                condition = batch["f"].flatten(start_dim=1)
                x_1 = batch["u"].flatten(start_dim=1)
                batch_size = min(x_1.shape[0], n_samples - len(velocity_stds))
                
                condition = condition[:batch_size]
                x_1 = x_1[:batch_size]
                
                # Sample initial noise
                x_0 = self.sample_base_distribution(x_1.shape, self.model_device)
                
                # Measure velocities at multiple time points
                time_points = torch.linspace(0, 1, 10, device=self.model_device)
                velocities = []
                
                for t_val in time_points:
                    t = t_val.expand(batch_size, 1)
                    x_t = (1 - t_val) * x_0 + t_val * x_1
                    v_t = self.model(x_t, condition, t)
                    velocities.append(v_t)
                
                velocities = torch.stack(velocities, dim=1)  # (batch, time, dim)
                
                # Standard deviation across time (lower = straighter)
                vel_std = velocities.std(dim=1).mean(dim=1)
                velocity_stds.extend(vel_std.cpu().tolist())
                
                # Path length (straight line = ||x_1 - x_0||)
                straight_dist = (x_1 - x_0).norm(dim=1)
                path_lengths.extend(straight_dist.cpu().tolist())
        
        return {
            'mean_velocity_std': torch.tensor(velocity_stds).mean().item(),
            'mean_path_length': torch.tensor(path_lengths).mean().item(),
            'straightness_score': 1.0 / (1.0 + torch.tensor(velocity_stds).mean().item())
        }
