import pytest
import torch
from torch import nn

from flowpde.flows import NeuralODEFlow
from flowpde.objectives import FlowMatchingObjective
from flowpde.trainers import Trainer


class DummyVelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, condition, t):
        del condition, t
        return self.scale * torch.ones_like(x)


def _batch(target_key="u", condition_key="f"):
    return {
        target_key: torch.randn(4, 3),
        condition_key: torch.randn(4, 2),
    }


def test_flow_matching_default_u_f_batch_still_works():
    flow = NeuralODEFlow(DummyVelocityModel())
    objective = FlowMatchingObjective(flow)

    loss = objective.compute_loss(_batch())

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_flow_matching_custom_inverse_problem_keys_work():
    flow = NeuralODEFlow(DummyVelocityModel())
    objective = FlowMatchingObjective(flow)
    batch = _batch(target_key="coefficient", condition_key="observation")

    loss = objective.compute_loss(
        batch,
        target_key="coefficient",
        condition_key="observation",
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_trainer_trains_configured_objective():
    model = DummyVelocityModel()
    flow = NeuralODEFlow(model)
    objective = FlowMatchingObjective(
        flow,
        target_key="coefficient",
        condition_key="observation",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = Trainer(
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
    )

    result = trainer.step(_batch(target_key="coefficient", condition_key="observation"))

    assert result["loss"] >= 0


def test_objective_uses_flow_keys_by_default():
    model = DummyVelocityModel()
    flow = NeuralODEFlow(
        model,
        target_key="coefficient",
        condition_key="observation",
    )
    objective = FlowMatchingObjective(flow)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = Trainer(
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
    )

    result = trainer.step(_batch(target_key="coefficient", condition_key="observation"))

    assert result["loss"] >= 0


def test_missing_configured_keys_raise_clear_error():
    flow = NeuralODEFlow(DummyVelocityModel())
    objective = FlowMatchingObjective(
        flow,
        target_key="coefficient",
        condition_key="observation",
    )
    batch = {"coefficient": torch.randn(4, 3)}

    with pytest.raises(KeyError, match="condition_key='observation'"):
        objective.compute_loss(batch)


def test_sampling_uses_target_dimension_when_condition_dimension_differs():
    flow = NeuralODEFlow(DummyVelocityModel())
    objective = FlowMatchingObjective(flow)
    batch = {
        "solution": torch.randn(4, 3),
        "parameters": torch.randn(4, 5),
    }
    objective.compute_loss(batch, target_key="solution", condition_key="parameters")

    samples = objective.sample(batch["parameters"], n_steps=2)

    assert samples.shape == (4, 3)


def test_sampling_accepts_target_shape_before_training():
    flow = NeuralODEFlow(DummyVelocityModel())
    objective = FlowMatchingObjective(flow)

    samples = objective.sample(torch.randn(4, 5), n_steps=2, target_shape=3)

    assert samples.shape == (4, 3)


def test_neural_ode_flow_is_the_public_continuous_flow_name():
    flow = NeuralODEFlow(DummyVelocityModel())

    assert flow.get_config()["flow_type"] == "neural_ode"


def test_flow_matching_objective_wraps_neural_ode_flow():
    flow = NeuralODEFlow(DummyVelocityModel())
    objective = FlowMatchingObjective(flow)

    assert objective.flow is flow
