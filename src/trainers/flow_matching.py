import torch

from trainer import Trainer




class FlowMatchingTrainer(Trainer):
    "Flow matching training for CNFs"

    def train_one_epoch(self, data_loader):

        train_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.step(batch)


    def compute_loss(self, batch):
        return NotImplemented
    