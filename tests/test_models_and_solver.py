"""Shape-contract tests for the velocity backbones, the ODE solver, and the
flow-matching loss.

Every backbone must honour the same contract: ``forward(x, f, t) -> velocity``
with the velocity shaped like ``x``. Because the objective flattens fields
before calling the model, the flattened path is what gets exercised here.
"""

import pytest
import torch
from torch import nn

from flowpde.core.base_conditioner import ConcatConditioner, NullConditioner
from flowpde.flows import NeuralODEFlow
from flowpde.models.convnet import ConvNet
from flowpde.models.mlp import MLP
from flowpde.models.resnet import ResNet
from flowpde.models.unet import UNet
from flowpde.objectives import FlowMatchingObjective, create_flow_matching
from flowpde.solvers import ODEFlowSolver


# Backbones


def test_mlp_velocity_matches_input_shape():
    model = MLP(input_dim=16, condition_dim=16, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 16)
    f = torch.randn(4, 16)
    t = torch.rand(4, 1)
    assert model(x, f, t).shape == x.shape


def test_mlp_accepts_differing_condition_dim():
    model = MLP(input_dim=16, condition_dim=8, hidden_dim=32, num_layers=2)
    out = model(torch.randn(4, 16), torch.randn(4, 8), torch.rand(4, 1))
    assert out.shape == (4, 16)


@pytest.mark.parametrize("spatial_size", [16, 32])
def test_unet_2d_flattened_roundtrip(spatial_size):
    """The objective passes flattened tensors; UNet must reshape internally
    and return a flattened velocity of the same size."""
    model = UNet(
        spatial_dim=2, spatial_size=spatial_size, base_channels=8,
        solution_channels=1, condition_channels=1, use_attention=False,
        return_spatial=False,
    )
    flat = spatial_size * spatial_size
    out = model(torch.randn(2, flat), torch.randn(2, flat), torch.rand(2, 1))
    assert out.shape == (2, flat)


def test_unet_multichannel_condition():
    """Inverse problems append an observation-mask channel to the condition."""
    model = UNet(
        spatial_dim=2, spatial_size=16, base_channels=8,
        solution_channels=1, condition_channels=2, use_attention=False,
        return_spatial=False,
    )
    out = model(torch.randn(2, 256), torch.randn(2, 512), torch.rand(2, 1))
    assert out.shape == (2, 256)


def test_unet_return_spatial():
    model = UNet(
        spatial_dim=2, spatial_size=16, base_channels=8,
        solution_channels=1, condition_channels=1, use_attention=False,
        return_spatial=True,
    )
    out = model(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16), torch.rand(2, 1))
    assert out.shape == (2, 1, 16, 16)


def test_convnet_and_resnet_shape_contract():
    for model in (
        ConvNet(spatial_dim=1, spatial_size=32, hidden_channels=8, num_blocks=2,
                solution_channels=1, condition_channels=1, return_spatial=False),
        ResNet(spatial_dim=1, spatial_size=32, base_channels=8,
               blocks_per_stage=[1, 1], solution_channels=1,
               condition_channels=1, return_spatial=False),
    ):
        out = model(torch.randn(3, 32), torch.randn(3, 32), torch.rand(3, 1))
        assert out.shape == (3, 32), type(model).__name__


def test_null_conditioner_ignores_condition():
    """The unconditional ablation: output must not depend on f.

    If this fails for a conditioned model it is a bug; if a *conditioned*
    model's error barely changes under this ablation, the model is ignoring
    its condition."""
    model = MLP(input_dim=8, hidden_dim=16, num_layers=1, conditioner=NullConditioner())
    with torch.no_grad():
        model.output_proj.weight.normal_()
    x = torch.randn(4, 8)
    t = torch.rand(4, 1)
    a = model(x, torch.randn(4, 8), t)
    b = model(x, torch.randn(4, 8) * 100, t)
    assert torch.allclose(a, b, atol=1e-6)


def test_concat_conditioner_uses_condition():
    model = MLP(input_dim=8, condition_dim=8, hidden_dim=16, num_layers=1,
                conditioner=ConcatConditioner(dim=1))
    # Models zero-initialize their output projection so the velocity starts at
    # zero; perturb it so the test measures conditioning, not initialization.
    with torch.no_grad():
        model.output_proj.weight.normal_()

    x = torch.randn(4, 8)
    t = torch.rand(4, 1)
    a = model(x, torch.randn(4, 8), t)
    b = model(x, torch.randn(4, 8) * 100, t)
    assert not torch.allclose(a, b, atol=1e-6)


# ODE solver


class ExponentialDecay(nn.Module):
    """dx/dt = -x, so x(1) = x(0) * exp(-1)."""

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, f, t):
        return -x


@pytest.mark.parametrize("method,tolerance", [("euler", 2e-2), ("rk4", 1e-5), ("dopri5", 1e-5)])
def test_solver_integrates_known_ode(method, tolerance):
    solver = ODEFlowSolver(model=ExponentialDecay(), method=method)
    x_init = torch.ones(3, 4)
    result = solver.sample(condition=torch.zeros(3, 4), x_init=x_init, n_steps=200)
    expected = x_init * torch.exp(torch.tensor(-1.0))
    assert torch.allclose(result, expected, atol=tolerance)


def test_solver_returns_trajectory():
    solver = ODEFlowSolver(model=ExponentialDecay(), method="euler")
    samples, trajectory = solver.sample(
        condition=torch.zeros(2, 4), x_init=torch.ones(2, 4),
        n_steps=10, return_trajectory=True,
    )
    assert trajectory.shape == (11, 2, 4)
    assert torch.allclose(trajectory[-1], samples)
    assert torch.allclose(trajectory[0], torch.ones(2, 4))


def test_solver_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        ODEFlowSolver(model=ExponentialDecay(), method="not_a_solver")


# Objective


def make_objective(**kwargs):
    model = MLP(input_dim=8, condition_dim=8, hidden_dim=16, num_layers=1)
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    return FlowMatchingObjective(flow, target_key="target", condition_key="input", **kwargs)


def test_loss_is_scalar_and_differentiable():
    objective = make_objective()
    batch = {"target": torch.randn(4, 8), "input": torch.randn(4, 8)}
    loss = objective.compute_loss(batch)

    assert loss.shape == ()
    loss.backward()
    grads = [p.grad for p in objective.model.parameters() if p.grad is not None]
    assert grads, "loss must produce gradients"


def test_missing_batch_key_raises_informative_error():
    objective = make_objective()
    with pytest.raises(KeyError, match="missing required key"):
        objective.compute_loss({"input": torch.randn(4, 8)})


def test_sample_respects_target_shape_when_dims_differ():
    """Condition and target need not share a dimension (multi-channel
    conditions, trajectory targets)."""
    model = MLP(input_dim=8, condition_dim=16, hidden_dim=16, num_layers=1)
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    objective = FlowMatchingObjective(flow, target_key="target", condition_key="input")

    samples = objective.sample(
        condition=torch.randn(3, 16), n_steps=5, solver="euler", target_shape=8
    )
    assert samples.shape == (3, 8)


def test_sample_return_trajectory():
    objective = make_objective()
    samples, trajectory = objective.sample(
        condition=torch.randn(3, 8), n_steps=7, solver="euler",
        target_shape=8, return_trajectory=True,
    )
    assert samples.shape == (3, 8)
    assert trajectory.shape == (8, 3, 8)


@pytest.mark.parametrize("variant", ["standard", "rectified", "ot_cfm", "ot_cfm_coupled"])
def test_all_presets_train_one_step(variant):
    model = MLP(input_dim=8, condition_dim=8, hidden_dim=16, num_layers=1)
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    objective = create_flow_matching(flow, variant=variant,
                                     target_key="target", condition_key="input")
    loss = objective.compute_loss({"target": torch.randn(6, 8), "input": torch.randn(6, 8)})
    assert torch.isfinite(loss)


def test_unknown_preset_rejected():
    model = MLP(input_dim=8, condition_dim=8, hidden_dim=16, num_layers=1)
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    with pytest.raises(ValueError, match="Unknown variant"):
        create_flow_matching(flow, variant="nonexistent")
