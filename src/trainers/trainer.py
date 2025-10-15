import torch
import time
import os

from torch import nn, optim

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable





class FlowTrainer(ABC):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device="cuda"):
        
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.device    = device

    
    def train(self,
              data_loader: Iterable,
              lr_scheduler: optim.lr_scheduler,
              device: torch.device,
              epochs: int,
              print_stats_interval: int,
              save_interval: int,
              save_dir: str,
              visualize: bool = True
              ):
        
        self.model.train()
        
        epoch_losses = []
        for epoch in range(epochs):

            
            start_time = time.time() # Start timer

            # Train one epoch and get epoch loss
            epoch_loss = self.train_one_epoch(model      = self.model,
                                             data_loader = data_loader)
            
            
            lr_scheduler.step() # Change LR
            elapsed_time = time.time() - start_time # Compute elapsed time of one epoch


            # Print statistics about the training
            if epoch % print_stats_interval == 0:
                self.print_stats()

            # Save model and get visualizations if u want :)
            if epoch % save_interval == 0:
                self.save_model()

                if visualize:
                    self.visualize_training()


    
    def print_stats(self, *args):
        print(
            f"[Epoch {args.epoch+1:04d}/{args.epochs}] "
            f"Train Loss: {args.train_loss:.10f} | "
            f"LR: {args.current_lr:.2e} | "
            f"Time: {args.elapsed:.2f}s"
            )
        
    def save_model(self, *args):
        os.makedirs(args.save_dir, exits_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"model_{args.epoch+1}.pt")
        torch.save({
        "model_state": self.model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "scheduler_state": args.lr_scheduler.state_dict(),
        "train_loss": args.epoch_loss,    
        }, ckpt_path)

    def visualize_training(self, *args):
        return NotImplemented

    @abstractmethod
    def compute_loss(self, batch: torch.Tensor):
        return NotImplemented
    
    @abstractmethod
    def train_one_epoch(self, data_loader: Iterable):
        return NotImplemented
    
