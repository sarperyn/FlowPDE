import torch
import time
import os

from torch import nn, optim

from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any

from utils.utils import print_stats, save_model




class Trainer(ABC):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device="cuda"):
        
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.device    = device

    
    def train(self,
              data_loader: Iterable,
              lr_scheduler: optim.lr_scheduler,
              epochs: int,
              print_stats_interval: int,
              save_interval: int,
              save_dir: str,
              visualize: bool = False
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
                print_stats()

            # Save model and get visualizations if u want :)
            if epoch % save_interval == 0:
                save_model()

                if visualize:
                    self.visualize_training()

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def visualize_training(self, *args):
        return NotImplemented

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        return NotImplemented
    
    @abstractmethod
    def train_one_epoch(self, data_loader: Iterable):
        return NotImplemented
    
