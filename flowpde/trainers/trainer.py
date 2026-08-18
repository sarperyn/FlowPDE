"""Generic trainer for FlowPDE objectives."""

import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

from flowpde.trainers.ema import EMA
from flowpde.utils import plot_curve, print_stats, save_model


class Trainer:
    """
    Train any objective exposing ``compute_loss(batch)`` and ``model``.

    Model selection defaults to the training loss, but for flow matching that
    loss has a large irreducible floor and correlates only weakly with sample
    quality.  Pass a ``validator`` (see
    :class:`~flowpde.trainers.evaluation.FlowEvaluator`) to select checkpoints
    on the error of ODE-sampled solutions instead — that is what the model is
    ultimately judged on.

    Args:
        objective: Object with ``compute_loss(batch)`` and a ``model``
            attribute.
        optimizer: Optimizer over the model parameters.
        scheduler: Optional LR scheduler, stepped once per epoch.
        device: Device to train on.
        gradient_clip: Max gradient norm, or ``None`` to disable.
        use_amp: Enable automatic mixed precision.
        ema_decay: Decay for an exponential moving average of the weights, or
            ``None`` to disable.  Validation and checkpointing then use the
            averaged weights, which is standard for this model family.
        validator: Zero-argument callable returning ``{metric: value}``.
            Called every ``val_interval`` epochs under EMA weights.
        val_interval: Epochs between validation passes.
        monitor: Validation metric to select checkpoints on.  Defaults to the
            first metric the validator returns.
        monitor_mode: ``'min'`` (default) or ``'max'``, whichever counts as an
            improvement for ``monitor``.
        checkpoint_extra: Extra entries stored in every checkpoint, e.g.
            ``{'normalizer_state': normalizer.state_dict()}`` so inference can
            reproduce the training-time preprocessing.
    """

    def __init__(
        self,
        objective: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        gradient_clip: Optional[float] = None,
        use_amp: bool = False,
        ema_decay: Optional[float] = None,
        validator: Optional[Callable[[], Dict[str, float]]] = None,
        val_interval: int = 1,
        monitor: Optional[str] = None,
        monitor_mode: str = "min",
        checkpoint_extra: Optional[Dict[str, Any]] = None,
    ):
        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")

        self.objective = objective.to(device)
        self.model = objective.model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        self.ema = (
            EMA(self.model, decay=ema_decay, device=torch.device(device))
            if ema_decay is not None
            else None
        )

        self.validator = validator
        self.val_interval = max(1, int(val_interval))
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.checkpoint_extra = dict(checkpoint_extra or {})

        self.best_loss = float("inf")
        self.best_metric = float("inf") if monitor_mode == "min" else -float("inf")
        self.history: Dict[str, list] = {"train_loss": [], "val": []}

    # Training

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

        if self.ema is not None:
            self.ema.update()

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

    # Validation

    def _ema_context(self):
        """Evaluate under averaged weights when EMA is enabled."""
        return self.ema.average_parameters() if self.ema is not None else nullcontext()

    def validate(self) -> Optional[Dict[str, float]]:
        """
        Run the validator under EMA weights, if one is configured.

        Returns:
            Metric dict, or ``None`` when no validator was given.
        """
        if self.validator is None:
            return None

        self.objective.eval()
        with torch.no_grad(), self._ema_context():
            metrics = self.validator()
        self.objective.train()
        return metrics

    def _is_improvement(self, value: float) -> bool:
        if self.monitor_mode == "min":
            return value < self.best_metric
        return value > self.best_metric

    # Checkpointing

    def _save(self, filename: str, epoch: int, epoch_loss: float) -> None:
        """Save a checkpoint, storing EMA weights as the primary model state."""
        extra = dict(self.checkpoint_extra)
        if self.ema is not None:
            extra["ema_state"] = self.ema.state_dict()

        with self._ema_context():
            # Under EMA the averaged weights are the deployable ones, so they
            # go into 'model_state'; raw weights are kept for resuming.
            save_model(
                save_dir=self.save_dir,
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch_loss=epoch_loss,
                filename=filename,
                extra=extra,
            )

    # Main loop

    def train(
        self,
        data_loader: Iterable,
        epochs: int,
        print_stats_interval: int,
        save_dir: str,
        save_interval: int,
    ) -> None:
        self.save_dir = save_dir
        self.objective.train()
        epoch_losses = []
        val_curve = []

        for epoch in range(epochs):
            start_time = time.perf_counter()
            epoch_loss = self.train_one_epoch(data_loader=data_loader)
            epoch_losses.append(epoch_loss)
            self.history["train_loss"].append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            metrics = None
            is_val_epoch = (epoch + 1) % self.val_interval == 0 or epoch == epochs - 1
            if self.validator is not None and is_val_epoch:
                metrics = self.validate()
                self.history["val"].append((epoch, metrics))
                if self.monitor is None:
                    self.monitor = next(iter(metrics))
                if self.monitor not in metrics:
                    raise KeyError(
                        f"monitor='{self.monitor}' is not among validator "
                        f"metrics {sorted(metrics)}"
                    )
                val_curve.append(metrics[self.monitor])

            if epoch % print_stats_interval == 0:
                stats = {
                    "Epoch": f"{epoch + 1:04d}/{epochs}",
                    "Train_Loss": epoch_loss,
                    "LR": self.optimizer.param_groups[0]["lr"],
                    "Time": time.perf_counter() - start_time,
                }
                if metrics:
                    stats.update({f"Val_{k}": v for k, v in metrics.items()})
                print_stats(**stats)

            # Model selection: validation metric when available, else loss.
            if metrics is not None:
                if self._is_improvement(metrics[self.monitor]):
                    self.best_metric = metrics[self.monitor]
                    self._save("best_model.pt", epoch, epoch_loss)
            elif self.validator is None and epoch_loss < self.best_loss:
                self._save("best_model.pt", epoch, epoch_loss)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

            if (epoch + 1) % save_interval == 0:
                self._save("latest_checkpoint.pt", epoch, epoch_loss)

        plot_curve(
            epoch_losses,
            title="Training loss curve",
            save_path=os.path.join(save_dir, "training_curve.png"),
        )
        if val_curve:
            plot_curve(
                val_curve,
                title=f"Validation {self.monitor}",
                save_path=os.path.join(save_dir, "validation_curve.png"),
            )

        print(f"\nBest train loss: {self.best_loss:.6f}")
        if self.validator is not None:
            print(f"Best val {self.monitor}: {self.best_metric:.6f}")
        print(f"Best model saved to: {os.path.join(save_dir, 'best_model.pt')}")
