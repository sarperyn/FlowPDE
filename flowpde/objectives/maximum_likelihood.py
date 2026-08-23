"""Maximum likelihood objective for neural ODE flows."""

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from flowpde.flows import NeuralODEFlow


class MaximumLikelihoodObjective(nn.Module):
    r"""
    Negative log likelihood objective for ``NeuralODEFlow``.

    The objective uses the flow's exact log probability computation:

    $$\mathcal{L} = -\mathbb{E}[\log p_\theta(x \mid c)]$$

    By default, the negative log likelihood is divided by the flattened target
    dimension so training logs report NLL per dimension instead of total NLL.
    """

    def __init__(
        self,
        flow: NeuralODEFlow,
        regularization: float = 0.0,
        normalize_by_dim: bool = True,
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
    ):
        super().__init__()
        self.flow = flow
        self.model = flow.model
        self.regularization = regularization
        self.normalize_by_dim = normalize_by_dim
        self.target_key = target_key or flow.target_key
        self.condition_key = condition_key or flow.condition_key

    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        x, condition = self.flow._extract_target_condition(
            batch,
            target_key=target_key or self.target_key,
            condition_key=condition_key or self.condition_key,
        )
        target_dim = x.flatten(start_dim=1).shape[1]
        self.flow._target_dim = target_dim
        log_px = self.flow.log_prob(x, condition)
        loss = -log_px.mean()
        if self.normalize_by_dim:
            loss = loss / target_dim

        if self.regularization > 0.0:
            t = torch.rand(x.shape[0], 1, device=self.flow.model_device)
            velocity = self.model(x.detach(), condition, t)
            loss = loss + self.regularization * velocity.pow(2).mean()

        return loss

    def get_config(self) -> Dict[str, Any]:
        return {
            "objective": "maximum_likelihood",
            "flow": self.flow.get_config(),
            "regularization": self.regularization,
            "normalize_by_dim": self.normalize_by_dim,
            "target_key": self.target_key,
            "condition_key": self.condition_key,
        }
