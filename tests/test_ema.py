"""Tests for the exponential moving average of model weights."""

import pytest
import torch
from torch import nn

from flowpde.trainers import EMA


def make_model(value: float = 0.0) -> nn.Module:
    model = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def test_shadow_follows_ema_recursion():
    """theta_ema <- d * theta_ema + (1 - d) * theta, with warmup disabled."""
    model = make_model(0.0)
    ema = EMA(model, decay=0.9, warmup=False)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update()

    # 0.9 * 0 + 0.1 * 1
    assert torch.allclose(ema.shadow["weight"], torch.full((2, 3), 0.1), atol=1e-6)

    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update()

    # 0.9 * 0.1 + 0.1 * 2
    assert torch.allclose(ema.shadow["weight"], torch.full((2, 3), 0.29), atol=1e-6)


def test_warmup_ramps_decay_from_zero():
    """Without warmup the shadow stays anchored to its arbitrary init."""
    model = make_model(0.0)
    ema = EMA(model, decay=0.999, warmup=True)

    assert ema.current_decay == pytest.approx(1 / 10)
    ema.update()
    assert ema.current_decay == pytest.approx(2 / 11)

    # After one update the shadow has already moved most of the way, whereas
    # decay=0.999 without warmup would have moved 0.1%.
    model2 = make_model(0.0)
    ema_no_warmup = EMA(model2, decay=0.999, warmup=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model2.weight.fill_(1.0)
    ema.update()
    ema_no_warmup.update()
    assert ema.shadow["weight"].mean() > ema_no_warmup.shadow["weight"].mean()


def test_average_parameters_swaps_and_restores():
    model = make_model(1.0)
    ema = EMA(model, decay=0.5, warmup=False)
    with torch.no_grad():
        model.weight.fill_(3.0)

    live = model.weight.detach().clone()

    with ema.average_parameters():
        assert torch.allclose(model.weight, ema.shadow["weight"])
        assert not torch.allclose(model.weight, live)

    assert torch.allclose(model.weight, live), "live weights must be restored"


def test_average_parameters_restores_on_exception():
    """A failing validation pass must not corrupt the training weights."""
    model = make_model(1.0)
    ema = EMA(model, decay=0.5, warmup=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    live = model.weight.detach().clone()

    with pytest.raises(RuntimeError, match="boom"):
        with ema.average_parameters():
            raise RuntimeError("boom")

    assert torch.allclose(model.weight, live)


def test_restore_without_store_raises():
    ema = EMA(make_model(), decay=0.9)
    with pytest.raises(RuntimeError, match="without a matching store"):
        ema.restore()


def test_copy_to_is_permanent():
    model = make_model(0.0)
    ema = EMA(model, decay=0.5, warmup=False)
    with torch.no_grad():
        model.weight.fill_(4.0)
    ema.update()

    ema.copy_to()
    assert torch.allclose(model.weight, ema.shadow["weight"])


def test_state_dict_roundtrip():
    model = make_model(0.0)
    ema = EMA(model, decay=0.9, warmup=False)
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update()
    ema.update()

    restored = EMA(make_model(0.0), decay=0.1).load_state_dict(ema.state_dict())
    assert restored.num_updates == ema.num_updates
    assert restored.decay == ema.decay
    assert torch.allclose(restored.shadow["weight"], ema.shadow["weight"])


def test_ema_smooths_oscillating_weights():
    """The point of EMA: average out the jitter of a noisy optimizer."""
    model = make_model(0.0)
    ema = EMA(model, decay=0.95, warmup=False)

    for step in range(400):
        with torch.no_grad():
            # True value 1.0, corrupted by a large alternating perturbation.
            model.weight.fill_(1.0 + (1.0 if step % 2 == 0 else -1.0))
        ema.update()

    raw_error = abs(model.weight.mean().item() - 1.0)
    ema_error = abs(ema.shadow["weight"].mean().item() - 1.0)
    assert ema_error < raw_error / 10


def test_rejects_invalid_decay():
    with pytest.raises(ValueError, match="decay must be in"):
        EMA(make_model(), decay=1.5)
