"""Tests for flow-matching components: paths, time samplers, couplings."""

import pytest
import torch

from flowpde.flows.components import (
    BetaSampler,
    IndependentCoupling,
    LinearPath,
    LogitNormalSampler,
    MiniBatchOTCoupling,
    OTConditionalPath,
    UniformSampler,
    get_coupling,
    get_path,
    get_time_sampler,
)

# Paths


@pytest.mark.parametrize("path", [LinearPath(), OTConditionalPath(sigma=0.0)])
def test_path_velocity_matches_finite_difference(path):
    """The declared velocity must be the actual time derivative of the path.

    This is the single most important invariant: if velocity() and
    interpolate() disagree, the model regresses a target that does not
    correspond to the interpolation it is being shown, and training silently
    learns the wrong vector field.
    """
    x_0 = torch.randn(16, 8, dtype=torch.float64)
    x_1 = torch.randn(16, 8, dtype=torch.float64)
    t = torch.rand(16, 1, dtype=torch.float64) * 0.8 + 0.1
    h = 1e-6

    numerical = (path.interpolate(x_0, x_1, t + h) - path.interpolate(x_0, x_1, t - h)) / (2 * h)
    analytic = path.velocity(x_0, x_1, t)

    assert torch.allclose(numerical, analytic, atol=1e-5)


@pytest.mark.parametrize("path", [LinearPath(), OTConditionalPath(sigma=0.0)])
def test_path_hits_endpoints(path):
    """x_t must equal x_0 at t=0 and x_1 at t=1."""
    x_0 = torch.randn(8, 5)
    x_1 = torch.randn(8, 5)

    zeros = torch.zeros(8, 1)
    ones = torch.ones(8, 1)

    assert torch.allclose(path.interpolate(x_0, x_1, zeros), x_0, atol=1e-6)
    assert torch.allclose(path.interpolate(x_0, x_1, ones), x_1, atol=1e-6)


@pytest.mark.parametrize("schedule", ["constant", "bridge"])
def test_ot_path_velocity_is_exact_for_sigma_above_zero(schedule):
    """The regression target must be the derivative of the shown sample path.

    Both schedules are checked with the *same* epsilon passed to interpolate()
    and velocity(): the constant schedule's target is the chord, the bridge
    schedule's target needs the d(sigma_t)/dt term as well.  Dropping it makes
    the model regress a biased velocity.
    """
    path = OTConditionalPath(sigma=0.5, schedule=schedule)
    x_0 = torch.randn(16, 8, dtype=torch.float64)
    x_1 = torch.randn(16, 8, dtype=torch.float64)
    noise = torch.randn(16, 8, dtype=torch.float64)
    t = torch.rand(16, 1, dtype=torch.float64) * 0.8 + 0.1
    h = 1e-6

    numerical = (
        path.interpolate(x_0, x_1, t + h, noise=noise)
        - path.interpolate(x_0, x_1, t - h, noise=noise)
    ) / (2 * h)
    analytic = path.velocity(x_0, x_1, t, noise=noise)

    assert torch.allclose(numerical, analytic, atol=1e-5)


@pytest.mark.parametrize("schedule", ["constant", "bridge"])
def test_ot_path_call_shares_one_epsilon(schedule):
    """__call__ must derive x_t and v_t from a single noise draw.

    Two draws would hand the model an input from one sample path and the
    derivative of a different one -- the bug this guards against.
    """
    path = OTConditionalPath(sigma=0.5, schedule=schedule)
    x_0 = torch.randn(32, 6, dtype=torch.float64)
    x_1 = torch.randn(32, 6, dtype=torch.float64)
    t = torch.rand(32, 1, dtype=torch.float64) * 0.8 + 0.1

    x_t, v_t = path(x_0, x_1, t)

    # Recover the epsilon that produced x_t, then check v_t is built from it.
    width = path._tube_width(t)
    epsilon = (x_t - (t * x_1 + (1 - t) * x_0)) / width
    expected = path.velocity(x_0, x_1, t, noise=epsilon)
    assert torch.allclose(v_t, expected, atol=1e-8)


def test_ot_bridge_schedule_pins_endpoints():
    """The bridge tube closes at t=0 and t=1; the constant tube does not."""
    x_0 = torch.randn(64, 6)
    x_1 = torch.randn(64, 6)
    zeros, ones = torch.zeros(64, 1), torch.ones(64, 1)

    bridge = OTConditionalPath(sigma=0.5, schedule="bridge")
    assert torch.allclose(bridge.interpolate(x_0, x_1, zeros), x_0, atol=1e-6)
    assert torch.allclose(bridge.interpolate(x_0, x_1, ones), x_1, atol=1e-6)

    # Mid-path it must differ from the noiseless interpolant.
    mid = bridge.interpolate(x_0, x_1, torch.full((64, 1), 0.5))
    assert not torch.allclose(mid, 0.5 * x_0 + 0.5 * x_1, atol=1e-3)

    # Tong et al.'s constant-width tube keeps noise at the endpoints, which is
    # exactly what lets its velocity target stay bounded.
    constant = OTConditionalPath(sigma=0.5, schedule="constant")
    assert not torch.allclose(constant.interpolate(x_0, x_1, zeros), x_0, atol=1e-3)


def test_ot_path_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="Unknown schedule"):
        OTConditionalPath(sigma=0.1, schedule="not_a_schedule")


def test_path_broadcasts_over_spatial_dims():
    """t of shape (B, 1) must broadcast against (B, C, H, W) fields."""
    path = LinearPath()
    x_0 = torch.randn(4, 1, 8, 8)
    x_1 = torch.randn(4, 1, 8, 8)
    t = torch.rand(4, 1)

    x_t, v_t = path(x_0, x_1, t)
    assert x_t.shape == x_0.shape
    assert v_t.shape == x_0.shape


def test_get_path_registry_and_passthrough():
    assert isinstance(get_path("linear"), LinearPath)
    assert isinstance(get_path("ot_conditional", sigma=0.1), OTConditionalPath)
    instance = LinearPath()
    assert get_path(instance) is instance
    with pytest.raises(ValueError, match="Unknown path"):
        get_path("does_not_exist")


# Time samplers


@pytest.mark.parametrize(
    "sampler",
    [UniformSampler(), LogitNormalSampler(), BetaSampler()],
)
def test_time_sampler_shape_and_range(sampler):
    """Samplers must return (B, 1) values strictly inside [0, 1]."""
    t = sampler(256, torch.device("cpu"))
    assert t.shape == (256, 1)
    assert torch.all(t >= 0.0) and torch.all(t <= 1.0)


def test_uniform_sampler_is_roughly_uniform():
    t = UniformSampler()(20000, torch.device("cpu"))
    assert abs(t.mean().item() - 0.5) < 0.02


def test_logit_normal_concentrates_mass_at_mid_times():
    """Logit-normal is the Rectified Flow default because it oversamples
    mid-path times, where the velocity field is hardest to learn."""
    t = LogitNormalSampler()(20000, torch.device("cpu"))
    mid_fraction = ((t > 0.25) & (t < 0.75)).float().mean().item()
    uniform_fraction = 0.5
    assert mid_fraction > uniform_fraction


def test_get_time_sampler_registry():
    assert isinstance(get_time_sampler("uniform"), UniformSampler)
    assert isinstance(get_time_sampler("logit_normal"), LogitNormalSampler)
    with pytest.raises(ValueError):
        get_time_sampler("nope")


# Couplings


def test_independent_coupling_is_identity():
    x_0, x_1 = torch.randn(8, 4), torch.randn(8, 4)
    out_0, out_1 = IndependentCoupling()(x_0, x_1)
    assert torch.equal(out_0, x_0)
    assert torch.equal(out_1, x_1)


def test_minibatch_ot_coupling_permutes_and_shortens_transport():
    """OT coupling must return a permutation of the same noise samples, and
    the total squared transport cost must not increase."""
    x_0 = torch.randn(32, 6)
    x_1 = torch.randn(32, 6)

    coupled_0, coupled_1 = MiniBatchOTCoupling()(x_0, x_1)

    assert coupled_0.shape == x_0.shape
    assert torch.equal(coupled_1, x_1), "data samples must not be reordered"

    # Same multiset of noise samples, just reordered.
    assert torch.allclose(coupled_0.sum(dim=0), x_0.sum(dim=0), atol=1e-5)

    cost_before = (x_1 - x_0).pow(2).sum()
    cost_after = (coupled_1 - coupled_0).pow(2).sum()
    assert cost_after <= cost_before + 1e-4


def test_get_coupling_registry():
    assert isinstance(get_coupling("independent"), IndependentCoupling)
    assert isinstance(get_coupling("minibatch_ot"), MiniBatchOTCoupling)
