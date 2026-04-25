"""
Unified Flow Trainer.

This module provides a single trainer class that works with any FlowMatching
configuration (standard, rectified, OT-CFM). It also includes support for:

- Standard training
- Reflow (iterative path straightening)
- Multiple loss functions (MSE, Huber)
- Gradient clipping
- Mixed precision training
"""

import copy
import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

from flowpde.flows.flow_matching import FlowMatching
from flowpde.trainers.trainer import Trainer


class FlowTrainer(Trainer):
    """
    Unified trainer for all Flow Matching variants.
    
    This trainer:
    1. Works with any FlowMatching instance (standard, rectified, OT-CFM)
    2. Provides optional reflow for iterative path straightening
    3. Supports gradient clipping, mixed precision, etc.
    
    Reflow (Rectified Flow):
    -----------------------
    Reflow iteratively trains new models on the generated paths of previous
    models, leading to straighter transport paths. This is key for distillation
    and few-step generation.
    
    After training, call:
    ```python
    trainer.reflow(train_data, num_iterations=2)
    ```
    
    Args:
        flow: FlowMatching instance (configured with path, time_sampler, coupling)
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        device: Training device ('cuda' or 'cpu')
        gradient_clip: Max gradient norm (None to disable)
        use_amp: Use automatic mixed precision (default: False)
        target_key: Batch key for target tensors (default: flow.target_key)
        condition_key: Batch key for condition tensors (default: flow.condition_key)
    
    Example:
        ```python
        from flowpde.flows import FlowMatching
        from flowpde.trainers import FlowTrainer
        
        # Standard Flow Matching
        flow = FlowMatching(model, path='linear', time_sampler='uniform')
        trainer = FlowTrainer(flow, optimizer, scheduler)
        trainer.train(data_loader, epochs=100, save_dir='results/')
        
        # Rectified Flow with reflow
        flow = FlowMatching(model, path='linear', time_sampler='logit_normal')
        trainer = FlowTrainer(flow, optimizer, scheduler)
        trainer.train(data_loader, epochs=100, save_dir='results/')
        trainer.reflow(data_loader, num_iterations=2, epochs_per_iteration=50)
        ```
    """
    
    def __init__(
        self,
        flow: FlowMatching,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: str = "cuda",
        gradient_clip: Optional[float] = None,
        use_amp: bool = False,
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
    ):
        # Call parent init with the flow's model
        super().__init__(
            model=flow.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        
        # Store the flow object
        self.flow = flow
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.target_key = target_key or flow.target_key
        self.condition_key = condition_key or flow.condition_key
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Reflow iteration tracking
        self.reflow_iteration = 0
        self._reflow_history: list = []
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute flow matching loss using the configured flow.
        
        Delegates to FlowMatching.compute_loss() which handles:
        - Path interpolation
        - Time sampling
        - Coupling
        - MSE between predicted and target velocities
        
        Args:
            batch: Dictionary with target and condition tensors
        
        Returns:
            Scalar loss tensor
        """
        return self.flow.compute_loss(
            batch,
            target_key=self.target_key,
            condition_key=self.condition_key,
        )
    
    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Single training step with optional gradient clipping and AMP.
        
        Args:
            batch: Training batch
        
        Returns:
            Dictionary with 'loss' and optional metrics
        """
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast():
                loss = self.compute_loss(batch)
            
            self.scaler.scale(loss).backward()
            
            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.compute_loss(batch)
            loss.backward()
            
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def reflow(
        self,
        data_loader: Iterable,
        num_iterations: int = 1,
        epochs_per_iteration: int = 50,
        n_steps: int = 50,
        save_dir: Optional[str] = None,
        print_stats_interval: int = 10,
    ) -> None:
        """
        Perform reflow iterations to straighten transport paths.
        
        Reflow (Liu et al., 2023) iteratively:
        1. Uses current model to generate (x_0, x_1) pairs
        2. Trains a new model on these pairs
        
        This results in straighter paths, enabling faster (fewer-step) sampling.
        
        Args:
            data_loader: Training data loader
            num_iterations: Number of reflow iterations
            epochs_per_iteration: Training epochs per iteration
            n_steps: ODE steps for generating new pairs
            save_dir: Directory to save intermediate models
            print_stats_interval: How often to print stats
        
        After calling reflow(), use the model as normal for sampling.
        """
        print(f"\n{'='*60}")
        print(f"Starting Reflow: {num_iterations} iteration(s)")
        print(f"{'='*60}\n")
        
        for iteration in range(num_iterations):
            self.reflow_iteration += 1
            print(f"\n--- Reflow Iteration {self.reflow_iteration} ---\n")
            
            # Generate new data pairs using current model
            reflowed_loader = self._generate_reflowed_data(
                data_loader, n_steps=n_steps
            )
            
            # Optionally reset model weights
            # In practice, we usually continue training the same model
            # but one could also reinitialize here
            
            # Reset optimizer state
            self._reset_optimizer_state()
            
            # Train on reflowed data
            iter_save_dir = None
            if save_dir:
                iter_save_dir = os.path.join(save_dir, f"reflow_{self.reflow_iteration}")
                os.makedirs(iter_save_dir, exist_ok=True)
            
            # Train for one reflow iteration
            self.train(
                data_loader=reflowed_loader,
                epochs=epochs_per_iteration,
                print_stats_interval=print_stats_interval,
                save_dir=iter_save_dir or save_dir,
            )
            
            # Track reflow history
            self._reflow_history.append({
                'iteration': self.reflow_iteration,
                'best_loss': self.best_loss,
            })
        
        print(f"\n{'='*60}")
        print(f"Reflow complete. Total iterations: {self.reflow_iteration}")
        print(f"{'='*60}\n")
    
    def _generate_reflowed_data(
        self,
        data_loader: Iterable,
        n_steps: int = 50,
    ) -> Iterable:
        """
        Generate reflowed data pairs.
        
        For each $(f, u)$ pair in the original data:
        1. Sample $z ~ N(0, I)$
        2. Run ODE: $z \to x_1' = \text{model.sample}(\text{condition}=f, x_{\text{init}}=z)$
        3. Store $(z, x_1')$ as new training pair
        
        This creates pairs where the path $z \to x_1'$ is determined by
        the current model, which should be straighter than the original.
        """
        reflowed_data = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                u, f = self.flow._extract_target_condition(
                    batch,
                    target_key=self.target_key,
                    condition_key=self.condition_key,
                )
                
                batch_size = u.shape[0]
                dim = u.shape[1]
                
                # Sample noise
                z = torch.randn(batch_size, dim, device=self.device)
                
                # Generate new targets using current model
                x_1_new = self.flow.sample(
                    condition=f,
                    x_init=z,
                    n_steps=n_steps,
                )
                
                # Store reflowed pair
                # For reflow training, x_0 = z, x_1 = generated sample
                reflowed_data.append({
                    self.target_key: x_1_new.cpu(),
                    self.condition_key: f.cpu(),
                    '_z': z.cpu(),  # Original noise (metadata)
                })
        
        self.model.train()
        
        # Return as a simple list (acts like DataLoader)
        return reflowed_data
    
    def _reset_optimizer_state(self) -> None:
        """Reset optimizer momentum/state for reflow iterations."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                state = self.optimizer.state.get(p, {})
                state.clear()
    
    @property
    def reflow_history(self) -> list:
        """Return history of reflow iterations."""
        return self._reflow_history
    
    def estimate_path_straightness(
        self,
        data_loader: Iterable,
        n_batches: int = 5,
    ) -> Dict[str, float]:
        """
        Estimate how straight the learned transport paths are.
        
        Can be called before and after reflow to measure improvement.
        
        Args:
            data_loader: Validation data
            n_batches: Number of batches to evaluate
        
        Returns:
            Dictionary with straightness metrics
        """
        total_straightness = 0.0
        total_velocity_std = 0.0
        n_samples = 0
        
        for i, batch in enumerate(data_loader):
            if i >= n_batches:
                break
            
            metrics = self.flow.estimate_straightness(
                batch,
                target_key=self.target_key,
                condition_key=self.condition_key,
            )
            total_straightness += metrics['straightness_score']
            total_velocity_std += metrics['velocity_std']
            n_samples += 1
        
        return {
            'avg_straightness': total_straightness / n_samples,
            'avg_velocity_std': total_velocity_std / n_samples,
        }
