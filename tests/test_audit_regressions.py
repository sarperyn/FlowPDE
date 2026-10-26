"""Analytic regressions for the SciML correctness audit."""



import math

import pytest
import torch
from torch import nn

from flowpde import MLP, FlowMatchingObjective, NeuralODEFlow
from flowpde.flows import (
    BatchSource,
    GaussianSource,
    MiniBatchOTCoupling,
    OTConditionalPath,
)
from flowpde.flows.components import UniformSampler
from flowpde.flows.neural_ode import NeuralODELogProbVectorField
from flowpde.objectives import MaximumLikelihoodObjective


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

    def forward(self, x, condition, t):
        return self.a * x + condition



class Translation(Affine):
    def forward(self, x, condition, t):
        return self.a.expand_as(x)



@pytest.mark.parametrize("estimator", ["exact", "hutchinson"])
def test_translation_likelihood_and_gradient(estimator):
    model = Translation()
    flow = NeuralODEFlow(model, trace_estimator=estimator, ode_method="rk4", ode_n_steps=8)
    x = torch.randn(2, 3, dtype=torch.float64)
    result = flow.log_prob(x, torch.zeros_like(x))
    expected = torch.distributions.Normal(model.a, 1).log_prob(x).sum(1)
    assert torch.allclose(result, expected, atol=1e-9)
    actual_grad = torch.autograd.grad(result.sum(), model.a)[0]
    expected_grad = torch.autograd.grad(expected.sum(), model.a)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=1e-9)
    with torch.no_grad():
        assert not flow.log_prob(x, torch.zeros_like(x)).requires_grad



def test_fixed_hutchinson_rhs_and_affine_likelihood():
    model = Affine()
    x = torch.ones(2, 3, dtype=torch.float64)
    field = NeuralODELogProbVectorField(model, torch.zeros_like(x), n_trace_samples=3)
    state = torch.cat([x, x.new_zeros(2, 1)], 1)
    first = field(x.new_tensor(0.5), state)
    assert torch.equal(first, field(x.new_tensor(0.5), state))
    flow = NeuralODEFlow(model, trace_estimator="exact", ode_method="rk4", ode_n_steps=16)
    c = torch.randn_like(x)
    actual = flow.log_prob(x, c)
    z = (x + c / model.a) * torch.exp(-model.a) - c / model.a
    expected = torch.distributions.Normal(0, 1).log_prob(z).sum(1) - 3 * model.a
    assert torch.allclose(actual, expected, atol=1e-8)
    assert torch.allclose(torch.autograd.grad(actual.sum(), model.a)[0],
                          torch.autograd.grad(expected.sum(), model.a)[0], atol=1e-7)



def test_uniform_density_in_pde_dimensions_and_prior_inheritance():
    flow = NeuralODEFlow(Affine(), base_distribution="uniform")
    objective = FlowMatchingObjective(flow)
    assert objective.source is flow.source
    z = flow.sample_base_distribution((2, 4096), torch.device("cpu"))
    assert z.dtype == torch.float64
    assert torch.all(z.abs() <= 1)
    assert torch.allclose(flow.base_log_prob(z), z.new_full((2,), -4096 * math.log(2)))
    assert torch.isneginf(flow.base_log_prob(z.new_full((1, 4096), 2))).all()



@pytest.mark.parametrize("method", ["rk4", "dopri5"])
def test_adjoint_condition_and_parameter_gradients_with_mode_restoration(method):
    class DropoutAffine(Affine):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.8)

        def forward(self, x, condition, t):
            return self.dropout(super().forward(x, condition, t))

    gradients = []
    for adjoint in (False, True):
        model = DropoutAffine().train()
        flow = NeuralODEFlow(model, use_adjoint=adjoint, ode_rtol=1e-9, ode_atol=1e-11)
        condition = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
        sample = flow.sample(condition, x_init=torch.zeros_like(condition), solver=method,
                             n_steps=24, no_grad=False)
        assert sample.dtype == torch.float64 and model.training and model.dropout.training
        sample.sum().backward()
        assert model.training and model.dropout.training
        gradients.append((condition.grad, model.a.grad))
    for direct, adjoint in zip(*gradients, strict=True):
        assert torch.allclose(direct, adjoint, atol=1e-7)



def test_conditional_ot_never_exchanges_noise_between_conditions():
    source, target = torch.randn(12, 3), torch.randn(12, 3)
    coupling = MiniBatchOTCoupling()
    condition = torch.arange(12).reshape(12, 1)
    result, _ = coupling.couple_conditioned(source, target, condition)
    assert torch.equal(result, source)
    condition = torch.arange(3).repeat_interleave(4).reshape(12, 1)
    result, _ = coupling.couple_conditioned(source, target, condition)
    for group in range(3):
        indices = condition[:, 0] == group
        assert torch.allclose(result[indices].sum(0), source[indices].sum(0), atol=1e-6)
        assert (result[indices] - target[indices]).square().sum() <= (source[indices] - target[indices]).square().sum() + 1e-5



@pytest.mark.parametrize("schedule", ["bridge", "smooth_bridge"])
def test_noise_schedule_derivative_near_endpoints(schedule):
    path = OTConditionalPath(0.5, schedule=schedule)
    t = torch.tensor([[1e-8], [1 - 1e-8]], dtype=torch.float64, requires_grad=True)
    x = torch.zeros(2, 1, dtype=torch.float64)
    noise = torch.ones_like(x)
    numerical = torch.autograd.grad(path.interpolate(x, x, t, noise).sum(), t)[0]
    assert torch.allclose(path.velocity(x, x, t, noise), numerical, atol=1e-7)



def test_bridge_requires_explicit_interior_training_and_constant_tube_prior():
    with pytest.raises(ValueError, match="bridge training requires"):
        FlowMatchingObjective(NeuralODEFlow(Affine()), path=OTConditionalPath(0.5, "bridge"))
    FlowMatchingObjective(NeuralODEFlow(Affine()), path=OTConditionalPath(0.5, "bridge"),
                          time_sampler=UniformSampler(0.01, 0.99))
    objective = FlowMatchingObjective(NeuralODEFlow(Affine()), path="ot", sigma=0.5,
                                      source=GaussianSource(2))
    assert objective.flow.source.std == pytest.approx(math.sqrt(4.25))



def test_spatial_velocity_loss_and_solver_preserve_batch_pairing():
    class Spatial(Affine):
        def forward(self, x, c, t):
            return (c + self.a * 0).reshape(x.shape[0], 1, -1)

    objective = FlowMatchingObjective(NeuralODEFlow(Spatial()), source=BatchSource())
    batch = {"u": torch.randn(3, 5), "f": torch.randn(3, 5), "x_0": torch.randn(3, 5)}
    expected = (batch["f"] - (batch["u"] - batch["x_0"])).square().mean()
    assert torch.allclose(objective.compute_loss(batch), expected.double())
    result = objective.sample(batch["f"], x_init=batch["x_0"], solver="euler", n_steps=4)
    assert torch.allclose(result, (batch["x_0"] + batch["f"]).double(), atol=1e-7)



def test_likelihood_objective_can_train_a_translation():
    model = Translation()
    objective = MaximumLikelihoodObjective(NeuralODEFlow(model, trace_estimator="exact", ode_method="rk4", ode_n_steps=4))
    loss = objective.compute_loss({"u": torch.ones(3, 2), "f": torch.zeros(3, 2)})
    loss.backward()
    assert model.a.grad == pytest.approx(-0.8)



def test_source_assignment_updates_the_flow_and_sampling_accepts_a_generator():
    objective = FlowMatchingObjective(NeuralODEFlow(MLP(4, hidden_dim=8, num_layers=1)))
    objective.source = GaussianSource(3)
    assert objective.flow.source is objective.source
    generator = torch.Generator().manual_seed(9)
    sample = objective.sample(torch.zeros(2, 4), target_shape=4, solver="euler",
                               n_steps=1, generator=generator)
    expected = torch.randn(2, 4, generator=torch.Generator().manual_seed(9)) * 3
    assert torch.equal(sample, expected)



def test_adaptive_hutchinson_likelihood_uses_one_probe_set():
    model = Affine()
    flow = NeuralODEFlow(model, n_trace_samples=3, ode_rtol=1e-9, ode_atol=1e-11)
    x = torch.ones(2, 3, dtype=torch.float64)
    torch.manual_seed(91)
    actual = flow.log_prob(x, torch.zeros_like(x))
    probes = torch.randn(3, 2, 3, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(91))
    z = x * torch.exp(-model.a)
    trace = model.a * probes.square().sum(-1).mean(0)
    expected = torch.distributions.Normal(0, 1).log_prob(z).sum(1) - trace
    assert torch.allclose(actual, expected, atol=1e-8)



def test_parameterless_zero_field_has_a_valid_density():
    class Zero(nn.Module):
        def forward(self, x, condition, t):
            return torch.zeros_like(x)

    flow = NeuralODEFlow(Zero(), trace_estimator="exact", ode_method="euler", ode_n_steps=2)
    x = torch.ones(2, 3)
    assert torch.allclose(flow.log_prob(x, torch.zeros_like(x)),
                          torch.distributions.Normal(0, 1).log_prob(x).sum(1))
