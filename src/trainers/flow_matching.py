import torch
import torch.nn.functional as F
from torch import nn, Tensor

from src.trainers.trainer import Trainer

from typing import Iterable, Dict, Optional, Tuple, Union

class FlowMatchingTrainer(Trainer):
    "Flow matching training for CNFs"

    def train_one_epoch(self, data_loader: Iterable):

        batch_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()} # to device
            loss = self.step(batch)
            batch_loss += loss["loss"]

        epoch_loss = batch_loss / len(data_loader)
        return epoch_loss

    def compute_loss(self, batch: Dict[str, torch.Tensor], path: str = "linear"):

        # can be another function
        ################################################################
        x_1       = batch["u"].flatten(start_dim=1).to(self.device)
        x_0       = torch.randn(x_1.shape).to(self.device)
        condition = batch["f"].flatten(start_dim=1).to(self.device)
        t         = torch.rand((batch["f"].shape[0], 1), device=self.device)
        ################################################################

        if path == "linear":
            x_t, dx_t = self.linear_flow_matching(x_1=x_1, x_0=x_0, t=t)
        

        return F.mse_loss(self.model(x_t, condition, t), dx_t)
    
    def linear_flow_matching(self, x_1: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor):

        x_t  = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        return x_t, dx_t