"""Tests for uncertainty-quantification metrics.

Calibration metrics are only trustworthy if they are themselves validated, so
these check against ensembles whose calibration is known by construction: draw
samples from ``N(target, sigma^2)`` and the ensemble is exactly calibrated;
shrink or inflate sigma and it becomes provably over- or under-confident.
"""

import pytest
import torch

from flowpde.utils import (
    UQMetrics,
    credible_interval_coverage,
    crps_ensemble,
    energy_score,
    error_spread_correlation,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
    variance_decomposition,
)


def calibrated_ensemble(num_members=200, batch=64, dim=8, sigma=1.0, seed=0):
    """An exactly calibrated ensemble.

    Calibration means the truth is *exchangeable* with the ensemble members —
    an independent draw from the same predictive distribution. Centring the
    ensemble on the truth instead would make the truth the median, and every
    credible interval would cover it trivially.
    """
    generator = torch.Generator().manual_seed(seed)
    center = torch.randn(batch, 1, dim, generator=generator)
    target = center + sigma * torch.randn(batch, 1, dim, generator=generator)
    samples = center + sigma * torch.randn(
        num_members, batch, 1, dim, generator=generator
    )
    return samples, target


def rescale_spread(samples, factor):
    """Inflate or shrink ensemble spread about its own mean."""
    mean = samples.mean(dim=0, keepdim=True)
    return mean + factor * (samples - mean)


# Coverage and reliability


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9])
def test_coverage_matches_nominal_for_calibrated_ensemble(level):
    samples, target = calibrated_ensemble(num_members=1000, batch=400)
    coverage = credible_interval_coverage(samples, target, level).item()
    assert abs(coverage - level) < 0.015, f"nominal {level}, empirical {coverage}"


def test_finite_ensemble_under_covers_at_high_levels():
    """A trap worth pinning: empirical quantiles from few members under-cover.

    A perfectly calibrated 50-member ensemble covers only ~96% with its
    nominal 99% interval. Reading that as over-confidence would be wrong —
    it is the estimator, not the model. Hold K fixed when comparing models.
    """
    few, target = calibrated_ensemble(num_members=50, batch=400)
    many, target_many = calibrated_ensemble(num_members=2000, batch=400, seed=1)

    coverage_few = credible_interval_coverage(few, target, 0.99).item()
    coverage_many = credible_interval_coverage(many, target_many, 0.99).item()

    assert coverage_few < 0.97, "small ensembles should visibly under-cover"
    assert coverage_many > coverage_few, "more members should reduce the bias"


def test_underdispersed_ensemble_is_overconfident():
    """Too-narrow spread must show coverage below nominal."""
    samples, target = calibrated_ensemble(sigma=1.0)
    narrow = rescale_spread(samples, 0.25)
    assert credible_interval_coverage(narrow, target, 0.9).item() < 0.6


def test_overdispersed_ensemble_exceeds_nominal():
    samples, target = calibrated_ensemble(sigma=1.0)
    wide = rescale_spread(samples, 4.0)
    assert credible_interval_coverage(wide, target, 0.9).item() > 0.98


def test_reliability_curve_is_near_diagonal_when_calibrated():
    samples, target = calibrated_ensemble()
    curve = reliability_curve(samples, target)
    assert len(curve["nominal"]) == len(curve["empirical"])
    assert curve["calibration_error"] < 0.03


def test_reliability_curve_detects_miscalibration():
    samples, target = calibrated_ensemble()
    narrow = rescale_spread(samples, 0.2)
    assert reliability_curve(narrow, target)["calibration_error"] > 0.2


def test_coverage_rejects_invalid_level():
    samples, target = calibrated_ensemble(num_members=8, batch=4)
    with pytest.raises(ValueError, match="level must be in"):
        credible_interval_coverage(samples, target, 1.5)


# Rank histogram


def test_rank_histogram_is_flat_when_calibrated():
    samples, target = calibrated_ensemble(num_members=20, batch=400)
    histogram = rank_histogram(samples, target)

    assert len(histogram) == 21
    expected = 1.0 / 21
    assert (histogram - expected).abs().max().item() < 0.02


def test_rank_histogram_is_u_shaped_when_underdispersed():
    """Truth falls outside a too-narrow ensemble, piling up in the end bins."""
    samples, target = calibrated_ensemble(num_members=20, batch=400)
    narrow = rescale_spread(samples, 0.2)
    histogram = rank_histogram(narrow, target)

    edges = histogram[0] + histogram[-1]
    middle = histogram[1:-1].sum()
    assert edges > middle


# Proper scoring rules


def test_crps_of_deterministic_forecast_equals_mae():
    """CRPS reduces exactly to MAE for a point-mass ensemble, which is what
    makes it comparable against a deterministic baseline."""
    generator = torch.Generator().manual_seed(0)
    target = torch.randn(16, 1, 8, generator=generator)
    prediction = torch.randn(16, 1, 8, generator=generator)

    samples = prediction.unsqueeze(0).repeat(12, 1, 1, 1)
    expected_mae = (prediction - target).abs().mean()

    assert crps_ensemble(samples, target).item() == pytest.approx(
        expected_mae.item(), abs=1e-6
    )


def test_crps_prefers_calibrated_over_miscalibrated():
    """A proper scoring rule cannot be improved by misreporting uncertainty."""
    samples, target = calibrated_ensemble(num_members=100, sigma=1.0)
    narrow = rescale_spread(samples, 0.2)
    wide = rescale_spread(samples, 5.0)

    calibrated = crps_ensemble(samples, target).item()
    assert calibrated < crps_ensemble(narrow, target).item()
    assert calibrated < crps_ensemble(wide, target).item()


def test_crps_matches_closed_form_for_gaussian():
    """For X ~ N(mu, sigma^2) and y = mu, CRPS = sigma * (sqrt(2) - 1) / sqrt(pi)."""
    torch.manual_seed(0)
    sigma = 2.0
    target = torch.zeros(1, 1, 1)
    samples = sigma * torch.randn(20000, 1, 1, 1)

    expected = sigma * (2**0.5 - 1) / (torch.pi**0.5)
    assert crps_ensemble(samples, target).item() == pytest.approx(expected, rel=0.02)


def test_energy_score_penalises_spatially_incoherent_fields():
    """The reason to prefer the energy score over pointwise CRPS.

    Both ensembles have identical pointwise marginals; only one preserves the
    spatial correlation of the truth. CRPS cannot tell them apart, the energy
    score can.
    """
    torch.manual_seed(0)
    num_members, batch, dim = 60, 32, 16

    base = torch.randn(batch, 1, dim)
    target = base.clone()

    # Coherent: one shared perturbation per field.
    shared = torch.randn(num_members, batch, 1, 1)
    coherent = base.unsqueeze(0) + shared.expand(-1, -1, -1, dim)

    # Incoherent: same marginal spread, independent per location.
    scrambled = coherent.clone()
    for location in range(dim):
        permutation = torch.randperm(num_members)
        scrambled[:, :, :, location] = coherent[permutation, :, :, location]

    crps_gap = abs(
        crps_ensemble(coherent, target).item() - crps_ensemble(scrambled, target).item()
    )
    energy_coherent = energy_score(coherent, target).item()
    energy_scrambled = energy_score(scrambled, target).item()

    assert crps_gap < 1e-6, "CRPS should be blind to the difference"
    assert energy_coherent < energy_scrambled, "energy score should not be"


def test_scoring_rules_require_multiple_members():
    target = torch.randn(4, 1, 8)
    single = target.unsqueeze(0)
    with pytest.raises(ValueError, match="at least 2"):
        crps_ensemble(single, target)
    with pytest.raises(ValueError, match="at least 2"):
        energy_score(single, target)


# Spread-skill


def test_spread_skill_ratio_is_one_when_calibrated():
    samples, target = calibrated_ensemble(num_members=50, batch=256)
    result = spread_skill_ratio(samples, target)
    assert result["adjusted_ratio"] == pytest.approx(1.0, abs=0.1)


def test_spread_skill_ratio_detects_overconfidence():
    samples, target = calibrated_ensemble(num_members=50, batch=256)
    narrow = rescale_spread(samples, 0.25)
    assert spread_skill_ratio(narrow, target)["adjusted_ratio"] < 0.5


# Decomposition


def test_variance_decomposition_recovers_known_split():
    """Construct known aleatoric and epistemic variances and recover them."""
    torch.manual_seed(0)
    num_models, num_members, batch, dim = 40, 60, 64, 4
    aleatoric_std, epistemic_std = 0.5, 2.0

    base = torch.randn(batch, 1, dim)
    model_samples = []
    for _ in range(num_models):
        model_bias = epistemic_std * torch.randn(batch, 1, dim)
        members = (base + model_bias).unsqueeze(0) + aleatoric_std * torch.randn(
            num_members, batch, 1, dim
        )
        model_samples.append(members)

    result = variance_decomposition(model_samples)
    assert result["aleatoric"] == pytest.approx(aleatoric_std**2, rel=0.15)
    assert result["epistemic"] == pytest.approx(epistemic_std**2, rel=0.15)
    assert result["total"] == pytest.approx(
        result["aleatoric"] + result["epistemic"], rel=1e-6
    )


def test_variance_decomposition_needs_multiple_models():
    samples, _ = calibrated_ensemble(num_members=8, batch=4)
    with pytest.raises(ValueError, match="at least 2 independently trained"):
        variance_decomposition([samples])


# Error-spread relationship


def test_error_spread_correlation_detects_informative_uncertainty():
    """Build samples whose spread is deliberately tied to their error."""
    torch.manual_seed(0)
    batch, dim, num_members = 128, 8, 40

    target = torch.zeros(batch, 1, dim)
    scale = torch.linspace(0.1, 3.0, batch).view(batch, 1, 1)
    samples = scale * torch.randn(num_members, batch, 1, dim)

    result = error_spread_correlation(samples, target)
    assert result["spearman"] > 0.7
    assert result["top_decile_error_ratio"] > 1.5


def test_error_spread_correlation_flat_when_uninformative():
    """Constant spread carries no per-sample information about error."""
    torch.manual_seed(0)
    batch, dim, num_members = 128, 8, 40
    center = torch.randn(batch, 1, dim)
    target = center + torch.randn(batch, 1, dim)
    samples = center.unsqueeze(0) + torch.randn(num_members, batch, 1, dim)

    result = error_spread_correlation(samples, target)
    assert abs(result["spearman"]) < 0.3


# Suite


def test_uq_metrics_suite_reports_expected_keys():
    samples, target = calibrated_ensemble(num_members=40, batch=32)
    results = UQMetrics(levels=[0.5, 0.9])(samples, target)

    for key in ("crps", "energy_score", "coverage_50", "coverage_90",
                "spread", "skill", "ratio", "adjusted_ratio", "spearman"):
        assert key in results, key
    assert results["coverage_90"] == pytest.approx(0.9, abs=0.05)


def test_accepts_list_of_tensors():
    samples, target = calibrated_ensemble(num_members=10, batch=16)
    as_list = [samples[i] for i in range(samples.shape[0])]
    assert crps_ensemble(as_list, target).item() == pytest.approx(
        crps_ensemble(samples, target).item(), abs=1e-6
    )
