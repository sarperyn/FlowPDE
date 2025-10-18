import torch
import time
import os

from torch import nn, optim

from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any

from utils.utils import print_stats, save_model
from src.visualization.utils import plot_curve




class Trainer(ABC):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler._LRScheduler, device="cuda"):
        
        self.model        = model.to(device)
        self.optimizer    = optimizer
        self.lr_scheduler = lr_scheduler
        self.device       = device


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

            
            start_time = time.time() # Start timer

            # Train one epoch and get epoch loss
            epoch_loss = self.train_one_epoch(data_loader = data_loader)
            epoch_losses.append(epoch_loss)
            
            
            self.lr_scheduler.step() # Change LR
            elapsed_time = time.time() - start_time # Compute elapsed time of one epoch


            # Print statistics about the training
            if epoch % print_stats_interval == 0:
                print_stats(
                    Epoch=f"{epoch+1:04d}/{epochs}",
                    Train_Loss=epoch_loss,
                    LR=self.optimizer.param_groups[0]['lr'],
                    Time=elapsed_time
                )

            # Save model and get visualizations if u want :)
            if epoch+1 % save_interval == 0:
                save_model(save_dir=save_dir, 
                           epoch=epoch, 
                           model=self.model,
                           optimizer=self.optimizer,
                           scheduler=self.lr_scheduler,
                           epoch_loss=epoch_loss)

        plot_curve(epoch_losses, title="Training loss curve", save_path=os.path.join(save_dir,"training_curve.png"))
        

    # General steps for all traninigs
    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        return NotImplemented
    
    @abstractmethod
    def train_one_epoch(self, data_loader: Iterable):
        return NotImplemented
    
