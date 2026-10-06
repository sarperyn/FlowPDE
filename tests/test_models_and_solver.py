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
from flowpde.solvers import ODEFlowSolver, compare_solvers, sample_with_ode_solver

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


# Module-level solver helpers. These are exported from ``flowpde.solvers`` but
# used by nothing inside the library, so nothing else would notice if their
# call into ODEFlowSolver drifted out of sync with its signature.


def test_sample_with_ode_solver_matches_analytic_decay():
    samples, trajectory = sample_with_ode_solver(
        model=ExponentialDecay(),
        condition=torch.zeros(3, 4),
        solver="rk4",
        n_steps=200,
        device="cpu",
        return_trajectory=True,
    )
    # x_init is drawn internally, so check the ratio rather than a fixed value.
    expected = trajectory[0] * torch.exp(torch.tensor(-1.0))
    assert samples.shape == (3, 4)
    assert torch.allclose(samples, expected, atol=1e-5)


def test_compare_solvers_reports_every_requested_solver():
    results = compare_solvers(
        model=ExponentialDecay(),
        condition=torch.zeros(2, 4),
        ground_truth=torch.zeros(2, 4),
        solvers=["euler", "rk4", "dopri5"],
        device="cpu",
        n_steps=50,
    )
    assert set(results) == {"euler", "rk4", "dopri5"}
    for name, info in results.items():
        assert info["samples"].shape == (2, 4), name
        assert info["time"] >= 0.0, name
        assert info["error"] is not None and info["error"] >= 0.0, name


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


# Regression tests for side effects (see Phase 1 review)


def test_sampling_does_not_leak_eval_mode():
    """Sampling must not permanently switch the caller's model to eval.

    ``ODEFlowSolver.sample`` sets eval mode for the integration; if it does
    not restore it, a mid-training call silently disables dropout and freezes
    BatchNorm statistics for the rest of the run.
    """
    model = MLP(input_dim=8, condition_dim=8, hidden_dim=16, num_layers=1, dropout=0.1)
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    objective = FlowMatchingObjective(flow, target_key="target", condition_key="input")

    objective.train()
    objective.sample(
        condition=torch.randn(4, 8), n_steps=4, solver="euler", target_shape=8
    )
    assert model.training, "sample() left the model in eval mode"

    objective.eval()
    objective.sample(
        condition=torch.randn(4, 8), n_steps=4, solver="euler", target_shape=8
    )
    assert not model.training, "sample() wrongly switched the model to train mode"


def test_sampling_no_grad_is_opt_out():
    """Default sampling detaches; no_grad=False keeps the graph."""
    model = MLP(input_dim=8, condition_dim=8, hidden_dim=16, num_layers=1)
    flow = NeuralODEFlow(model, target_key="target", condition_key="input")
    objective = FlowMatchingObjective(flow, target_key="target", condition_key="input")
    condition = torch.randn(4, 8)

    assert not objective.sample(
        condition=condition, n_steps=4, solver="euler", target_shape=8
    ).requires_grad
    assert objective.sample(
        condition=condition, n_steps=4, solver="euler", target_shape=8,
        no_grad=False,
    ).requires_grad


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: MLP(input_dim=64, condition_dim=64, hidden_dim=16, num_layers=2),
            id="mlp",
        ),
        pytest.param(
            lambda: ConvNet(
                spatial_dim=2, spatial_size=8, hidden_channels=8, num_blocks=2
            ),
            id="convnet",
        ),
        pytest.param(
            lambda: UNet(spatial_dim=2, spatial_size=8, base_channels=8),
            id="unet",
        ),
        pytest.param(
            lambda: ResNet(
                spatial_dim=2, spatial_size=8, base_channels=8, blocks_per_stage=[1, 1]
            ),
            id="resnet",
        ),
        pytest.param(
            lambda: ResNet(
                spatial_dim=2,
                spatial_size=8,
                base_channels=8,
                blocks_per_stage=[1, 1],
                downsample=True,
            ),
            id="resnet-downsample",
        ),
    ],
)
def test_velocity_head_is_zero_initialised(build):
    """Every backbone must predict v = 0 at initialisation.

    That is the stable start for flow matching, and it used to depend on the
    final layer happening to be *named* 'output_*' -- which the ResNet
    upsampling path is not.
    """
    model = build()
    x = torch.randn(2, 1, 8, 8)
    f = torch.randn(2, 1, 8, 8)
    t = torch.rand(2, 1)

    with torch.no_grad():
        velocity = model(x, f, t)

    assert torch.count_nonzero(velocity) == 0


def test_attention_output_projection_is_not_zero_initialised():
    """Only the velocity head is zeroed, not layers that merely share a name."""
    model = UNet(spatial_dim=2, spatial_size=8, base_channels=8, use_attention=True)

    assert torch.count_nonzero(model.output_conv.weight) == 0
    assert torch.count_nonzero(model.bottleneck_attn.attn.out_proj.weight) > 0


# Phase 2: one sampling path, configured defaults, explicit target dimension


class _Swirl(nn.Module):
    """Non-trivial velocity, so transport actually moves points."""

    def __init__(self, dim=4):
        super().__init__()
        self.net = nn.Linear(2 * dim + 1, dim)
        with torch.no_grad():
            self.net.weight.normal_(0, 0.4)
            self.net.bias.normal_(0, 0.2)

    def forward(self, x, f, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x, f, t.expand(x.shape[0], 1)], dim=1))


@pytest.mark.parametrize(
    "method,n_steps,tolerance",
    [("euler", 400, 1e-2), ("rk4", 100, 1e-5), ("dopri5", None, 1e-4)],
)
def test_forward_transform_inverts_sampling(method, n_steps, tolerance):
    """data -> latent -> data must return where it started.

    Backward integration (t: 1 -> 0) needs a positive step_size magnitude;
    a signed one makes torchdiffeq reject the fixed-step grid outright.
    """
    model = _Swirl()
    flow = NeuralODEFlow(model, ode_method=method, ode_n_steps=n_steps)
    x = torch.randn(3, 4)
    condition = torch.randn(3, 4)

    latent = flow.forward_transform(x, condition)
    recovered = flow.sample(condition=condition, x_init=latent)

    # The flow must actually transport, or the round-trip is vacuous.
    assert (latent - x).abs().max() > 0.5
    assert torch.allclose(recovered, x, atol=tolerance)


def test_constructor_solver_settings_govern_sampling():
    """`ode_method` used to be overridden by sample()'s own default."""
    model = MLP(input_dim=8, condition_dim=8, hidden_dim=16, num_layers=1)
    flow = NeuralODEFlow(model, ode_method="euler", ode_n_steps=7, ode_rtol=1e-3)

    built = flow._make_solver()
    assert built.method == "euler"
    assert built.rtol == 1e-3

    # An explicit argument still wins.
    assert flow._make_solver("rk4").method == "rk4"


def test_unknown_solver_kwargs_raise():
    """Silently dropping solver arguments hides typos in run configs."""
    model = MLP(input_dim=8, condition_dim=8, hidden_dim=16, num_layers=1)
    flow = NeuralODEFlow(model)
    objective = FlowMatchingObjective(flow)

    with pytest.raises(TypeError):
        objective.sample(
            condition=torch.randn(2, 8), target_shape=8, not_a_solver_option=1
        )


def test_target_dimension_survives_a_state_dict_round_trip():
    """A reloaded flow must still know what shape to sample.

    The dimension is learned during training; as a plain attribute it was
    lost on reload and sampling silently fell back to the *condition*
    dimension.
    """
    def build():
        model = MLP(input_dim=4, condition_dim=8, hidden_dim=16, num_layers=1)
        flow = NeuralODEFlow(model, target_key="u", condition_key="f")
        return FlowMatchingObjective(flow)

    trained = build()
    batch = {"u": torch.randn(3, 4), "f": torch.randn(3, 8)}
    trained.compute_loss(batch)

    reloaded = build()
    reloaded.load_state_dict(trained.state_dict())

    samples = reloaded.sample(condition=batch["f"], n_steps=2, solver="euler")
    assert samples.shape == (3, 4)


def test_unknown_target_dimension_raises_instead_of_guessing():
    """Better a clear error than silently sampling the condition's shape."""
    model = MLP(input_dim=4, condition_dim=8, hidden_dim=16, num_layers=1)
    objective = FlowMatchingObjective(
        NeuralODEFlow(model, target_key="u", condition_key="f")
    )

    with pytest.raises(RuntimeError, match="target dimension"):
        objective.sample(condition=torch.randn(3, 8), n_steps=2, solver="euler")

    # Either escape hatch works.
    assert objective.sample(
        condition=torch.randn(3, 8), n_steps=2, solver="euler", target_shape=4
    ).shape == (3, 4)
    assert objective.sample(
        condition=torch.randn(3, 8), n_steps=2, solver="euler",
        x_init=torch.randn(3, 4),
    ).shape == (3, 4)


def test_fixed_step_solver_requires_n_steps():
    """Without n_steps a fixed-step solve collapses to one giant step."""
    model = MLP(input_dim=4, condition_dim=4, hidden_dim=8, num_layers=1)
    solver = ODEFlowSolver(model=model, method="euler")

    with pytest.raises(ValueError, match="n_steps is required"):
        solver.sample(condition=torch.randn(2, 4), x_init=torch.randn(2, 4))


def test_base_distribution_sampling_and_density_agree():
    """`sample_base_distribution` and `base_log_prob` must describe one measure.

    BaseFlow used to draw U(0,1) while NeuralODEFlow drew U(-1,1) and scored
    the density of yet another support.
    """
    model = MLP(input_dim=4, condition_dim=4, hidden_dim=8, num_layers=1)

    gaussian = NeuralODEFlow(model, base_distribution="gaussian")
    z = gaussian.sample_base_distribution((4096, 4), torch.device("cpu"))
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(z).sum(dim=1)
    assert torch.allclose(gaussian.base_log_prob(z), expected, atol=1e-4)

    uniform = NeuralODEFlow(model, base_distribution="uniform")
    z = uniform.sample_base_distribution((4096, 4), torch.device("cpu"))
    assert z.min() >= -1.0 and z.max() <= 1.0
    assert uniform.base_log_prob(z).allclose(
        torch.full((4096,), -4 * torch.tensor(2.0).log().item()), atol=1e-4
    )

    # Outside the support the density is zero.
    assert uniform.base_log_prob(torch.full((1, 4), 5.0)).item() < -20


# Phase 3: portability and contract hygiene


def test_unet_output_head_matches_clamped_channel_width():
    """max_channels can clamp the first encoder level below base_channels.

    The output head was built for base_channels regardless, so the forward
    pass died on a channel mismatch.
    """
    model = UNet(spatial_dim=2, spatial_size=32, base_channels=64, max_channels=32)
    out = model(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32), torch.rand(2, 1))
    assert out.shape == (2, 32 * 32)


def test_film_starts_as_identity():
    """FiLM is (1 + scale) * x + shift, so an untrained block passes features through.

    With a bare `scale`, the generator's random zero-mean output rescales
    every channel before training has begun.
    """
    from flowpde.core import FiLMConditioner

    conditioner = FiLMConditioner(condition_dim=256, feature_dim=16, hidden_dim=64)
    with torch.no_grad():
        # A generator that has learned nothing outputs zeros.
        for layer in conditioner.film_generator:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()

    features = torch.randn(8, 16, 16, 16)
    condition = torch.randn(8, 1, 16, 16)
    assert torch.allclose(conditioner(features, condition), features)


def test_resnet_blocks_per_stage_default_is_not_shared():
    """A mutable default argument is shared by every instance ever built."""
    first = ResNet(spatial_dim=2, spatial_size=8, base_channels=8)
    assert len(first.stages) == 4

    custom = ResNet(
        spatial_dim=2, spatial_size=8, base_channels=8, blocks_per_stage=[1]
    )
    assert len(custom.stages) == 1

    # The default must be unaffected by the call that passed its own list.
    assert len(ResNet(spatial_dim=2, spatial_size=8, base_channels=8).stages) == 4


def test_minibatch_ot_coupling_reports_a_missing_scipy():
    """Falling back to independent coupling would make an OT ablation a no-op."""
    import builtins

    from flowpde.flows.components import MiniBatchOTCoupling

    coupling = MiniBatchOTCoupling()
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("scipy is not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_scipy
    try:
        with pytest.raises(ImportError, match="needs scipy"):
            coupling.couple(torch.randn(8, 4), torch.randn(8, 4))
    finally:
        builtins.__import__ = real_import

    # With scipy present it still produces a genuine assignment.
    x_1 = torch.randn(32, 6)
    x_0 = x_1[torch.randperm(32)] + 0.01 * torch.randn(32, 6)
    before = (x_0 - x_1).pow(2).sum(1).mean()
    coupled_0, coupled_1 = coupling.couple(x_0, x_1)
    assert (coupled_0 - coupled_1).pow(2).sum(1).mean() < before / 100
