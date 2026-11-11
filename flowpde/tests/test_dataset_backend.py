import pytest
import torch

from flowpde.datasets import (
    BurgersGenerator,
    DarcyGenerator,
    PoissonConfig,
    PoissonGenerator,
)


def _same_tensors(left, right, keys):
    return all(torch.allclose(left[key], right[key]) for key in keys)


def test_poisson_forward_inverse_and_reproducibility():
    gen = PoissonGenerator(
        num_spatial_dims=1,
        num_points=16,
        num_samples=3,
        source_num_terms=2,
        source_max_mode=2,
    )

    forward = gen.generate(seed=7)
    inverse = gen.generate(seed=7, problem="inverse")
    repeat = gen.generate(seed=7)

    assert len(forward) == 3
    assert forward[0]["input"].shape == (1, 16)
    assert forward[0]["target"].shape == (1, 16)
    assert torch.allclose(forward[0]["input"], forward.get_raw_data()["source"][0])
    assert torch.allclose(forward[0]["target"], forward.get_raw_data()["solution"][0])
    assert torch.allclose(inverse[0]["input"], inverse.get_raw_data()["solution"][0])
    assert torch.allclose(inverse[0]["target"], inverse.get_raw_data()["source"][0])
    assert _same_tensors(forward.get_raw_data(), repeat.get_raw_data(), ["source", "solution"])


def test_poisson_removed_advanced_source_options():
    cfg = PoissonConfig()

    assert not hasattr(cfg, "gaussian_bump_prob")
    assert not hasattr(cfg, "ic_cutoff_min")
    assert not hasattr(cfg, "ic_cutoff_max")

    with pytest.raises(TypeError):
        PoissonGenerator(gaussian_bump_prob=0.5)


def test_problem_validation():
    with pytest.raises(ValueError):
        PoissonGenerator(num_points=8, num_samples=1).generate(problem="bad")

    with pytest.raises(ValueError):
        BurgersGenerator(num_points=16, num_samples=1, num_steps=1).generate(problem="bad")

    with pytest.raises(ValueError):
        DarcyGenerator(num_spatial_dims=1, num_points=8, num_samples=1).generate(problem="bad")


def test_burgers_forward_inverse_and_optional_observation_mask():
    gen = BurgersGenerator(
        num_spatial_dims=1,
        num_points=16,
        num_samples=2,
        num_steps=2,
        ic_num_terms=2,
        ic_max_mode=3,
        diffusivity_min=1e-2,
        diffusivity_max=1e-2,
    )

    forward = gen.generate(seed=11)
    inverse = gen.generate(seed=11, problem="inverse")

    assert "obs_mask" not in forward[0]
    assert "obs_mask" not in inverse[0]
    assert torch.allclose(forward[0]["input"], forward.get_raw_data()["initial"][0])
    assert torch.allclose(inverse[0]["input"], inverse.get_raw_data()["final"][0])
    assert torch.allclose(inverse[0]["target"], inverse.get_raw_data()["initial"][0])

    masked_gen = BurgersGenerator(
        num_spatial_dims=1,
        num_points=16,
        num_samples=2,
        num_steps=1,
        ic_num_terms=2,
        ic_max_mode=3,
        obs_mask_fraction=0.5,
    )
    masked = masked_gen.generate(seed=13, problem="inverse")

    assert "obs_mask" in masked[0]
    assert masked[0]["input"].shape[0] == masked.get_raw_data()["final"].shape[1] + 1


def test_burgers_removed_advanced_ic_options():
    gen = BurgersGenerator()
    cfg = gen.config

    assert not hasattr(cfg, "ic_cutoff")
    assert not hasattr(cfg, "ic_max_one")
    assert not hasattr(cfg, "amplitude_min")
    assert not hasattr(cfg, "amplitude_max")

    with pytest.raises(TypeError):
        BurgersGenerator(ic_cutoff=3)


def test_observation_noise_is_seeded():
    gen = PoissonGenerator(
        num_spatial_dims=1,
        num_points=16,
        num_samples=2,
        obs_noise_std=0.1,
    )

    first = gen.generate(seed=5, problem="inverse")
    second = gen.generate(seed=5, problem="inverse")

    assert torch.allclose(first.get_raw_data()["solution"], second.get_raw_data()["solution"])


def test_darcy_forward_and_inverse_modes():
    gen = DarcyGenerator(
        num_spatial_dims=1,
        num_points=8,
        num_samples=2,
        f_cutoff=2,
        cg_steps=20,
    )

    forward = gen.generate(seed=17)
    assert forward[0]["input"].shape == (2, 8)
    assert forward[0]["target"].shape == (1, 8)
    assert torch.allclose(
        forward[0]["input"],
        torch.cat(
            [
                forward.get_raw_data()["kappa"][0],
                forward.get_raw_data()["source"][0],
            ],
            dim=0,
        ),
    )

    both = gen.generate(seed=17, problem="inverse", inverse_mode="both")
    coefficient = gen.generate(seed=17, problem="inverse", inverse_mode="coefficient")
    source = gen.generate(seed=17, problem="inverse", inverse_mode="source")

    assert both[0]["input"].shape == (1, 8)
    assert both[0]["target"].shape == (2, 8)
    assert coefficient[0]["input"].shape == (2, 8)
    assert coefficient[0]["target"].shape == (1, 8)
    assert source[0]["input"].shape == (2, 8)
    assert source[0]["target"].shape == (1, 8)
