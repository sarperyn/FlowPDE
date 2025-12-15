"""
Base Trainer Abstract Class for Normalizing Flows

Defines the interface for all flow-based trainers in FlowPDE and provides common
functionality for training loops, checkpointing, and metrics tracking.

FlowPDE is a normalizing flow library for solving forward and inverse problems
in parametric PDEs. All trainers are designed for flow-based training algorithms
(Flow Matching, CNF, Rectified Flow), not specific datasets or problem types.

Trainers should be organized by training schema (algorithm), not by dataset or PDE type.
"""

import torch
import time
import os

from torch import nn, optim

from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any

from flowpde.utils import print_stats, save_model, plot_curve


class Trainer(ABC):
    """
    Abstract base trainer for all normalizing flow training algorithms.
    
    This class provides:
    - Generic training loop with epoch management
    - Checkpointing every N epochs
    - Learning rate scheduling
    - Training curve visualization
    - Device management
    
    Subclasses should implement:
    - compute_loss(): How to compute loss for the specific flow algorithm
    - step(): Single training step (optimizer update)
    
    The trainer is dataset-agnostic and works with any forward or inverse problem.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, device="cuda"):
        
        self.model        = model.to(device)
        self.optimizer    = optimizer
        self.scheduler  = scheduler
        self.device       = device
        self.best_loss    = float('inf')  # Track best loss for model saving


    # Generic training loop for all models
    def train(self,
              data_loader: Iterable,
              epochs: int,
              print_stats_interval: int,
              save_dir: str,
              save_interval: int = 30000,
              ):
        
        self.model.train()
        
        epoch_losses = []
        for epoch in range(epochs):

            
            start_time = time.perf_counter() # Start time for epoch

            # Train one epoch and get epoch loss (train_one_epoch may return (loss, metrics))
            result = self.train_one_epoch(data_loader = data_loader)
            if isinstance(result, tuple):
                epoch_loss, epoch_metrics = result
            else:
                epoch_loss = result
                epoch_metrics = {}

            epoch_losses.append(epoch_loss)
            
            
            self.scheduler.step() # Change LR
            elapsed_time = time.perf_counter() - start_time # Elapsed time for epoch


            # Print statistics about the training
            if epoch % print_stats_interval == 0:
                stats = {
                    "Epoch": f"{epoch+1:04d}/{epochs}",
                    "Train_Loss": epoch_loss,
                    "LR": self.optimizer.param_groups[0]['lr'],
                    "Time": elapsed_time,
                }
                # merge optional metrics
                stats.update(epoch_metrics)
                print_stats(**stats)

            # Save best model (overwrite previous best)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_model(save_dir=save_dir, 
                           epoch=epoch, 
                           model=self.model,
                           optimizer=self.optimizer,
                           scheduler=self.scheduler,
                           epoch_loss=epoch_loss,
                           filename="best_model.pt")
            
            # Save periodic checkpoint (overwrite to save space)
            if (epoch + 1) % save_interval == 0:
                save_model(save_dir=save_dir, 
                           epoch=epoch, 
                           model=self.model,
                           optimizer=self.optimizer,
                           scheduler=self.scheduler,
                           epoch_loss=epoch_loss,
                           filename="latest_checkpoint.pt")

        plot_curve(epoch_losses, title="Training loss curve", save_path=os.path.join(save_dir,"training_curve.png"))
        print(f"\nBest loss: {self.best_loss:.6f}")
        print(f"Best model saved to: {os.path.join(save_dir, 'best_model.pt')}")
        

    # General steps for all traninigs
    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        """Compute loss for a single batch. Subclasses must implement this."""
        return NotImplemented
    
    # Concrete implementation - same for all flow trainers
    def train_one_epoch(self, data_loader: Iterable) -> float:
        """
        Train for one epoch and return average loss.
        
        This is a concrete method that works for all flow trainers.
        Subclasses only need to implement compute_loss().
        
        Args:
            data_loader: DataLoader with training batches
            
        Returns:
            Average loss over the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            loss_dict = self.step(batch)
            total_loss += loss_dict['loss']
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
