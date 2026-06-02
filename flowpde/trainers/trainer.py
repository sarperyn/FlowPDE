"""Generic trainer for FlowPDE objectives."""

import os
import time
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

from flowpde.utils import plot_curve, print_stats, save_model


class Trainer:
    """Train any objective exposing ``compute_loss(batch)`` and ``model``."""

    def __init__(
        self,
        objective: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        gradient_clip: Optional[float] = None,
        use_amp: bool = False,
    ):
        self.objective = objective.to(device)
        self.model = objective.model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.best_loss = float("inf")

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.objective.compute_loss(batch)

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(batch)
            self.scaler.scale(loss).backward()
            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.compute_loss(batch)
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

        return {"loss": loss.item()}

    def train_one_epoch(self, data_loader: Iterable) -> float:
        self.objective.train()
        total_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            loss_dict = self.step(batch)
            total_loss += loss_dict["loss"]
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self,
        data_loader: Iterable,
        epochs: int,
        print_stats_interval: int,
        save_dir: str,
        save_interval: int,
    ) -> None:
        self.objective.train()
        epoch_losses = []

        for epoch in range(epochs):
            start_time = time.perf_counter()
            epoch_loss = self.train_one_epoch(data_loader=data_loader)
            epoch_losses.append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % print_stats_interval == 0:
                stats = {
                    "Epoch": f"{epoch + 1:04d}/{epochs}",
                    "Train_Loss": epoch_loss,
                    "LR": self.optimizer.param_groups[0]["lr"],
                    "Time": time.perf_counter() - start_time,
                }
                print_stats(**stats)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_model(
                    save_dir=save_dir,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch_loss=epoch_loss,
                    filename="best_model.pt",
                )

            if (epoch + 1) % save_interval == 0:
                save_model(
                    save_dir=save_dir,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch_loss=epoch_loss,
                    filename="latest_checkpoint.pt",
                )

        plot_curve(
            epoch_losses,
            title="Training loss curve",
            save_path=os.path.join(save_dir, "training_curve.png"),
        )
        print(f"\nBest loss: {self.best_loss:.6f}")
        print(f"Best model saved to: {os.path.join(save_dir, 'best_model.pt')}")
