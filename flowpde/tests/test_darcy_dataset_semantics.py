import pytest
import torch

darcy = pytest.importorskip("flowpde.datasets.exponax.darcy")
DarcyDataset = darcy.DarcyDataset


def _raw_darcy_data():
    return {
        "kappa": torch.full((2, 1, 4, 4), 2.0),
        "source": torch.full((2, 1, 4, 4), 3.0),
        "solution": torch.full((2, 1, 4, 4), 5.0),
    }


def _raw_darcy_data_with_mask():
    data = _raw_darcy_data()
    data["obs_mask"] = torch.tensor(
        [
            [[[1.0, 0.0, 1.0, 0.0]] * 4],
            [[[0.0, 1.0, 0.0, 1.0]] * 4],
        ]
    )
    return data


def test_darcy_forward_conditions_on_kappa_and_source_targets_solution():
    dataset = DarcyDataset(_raw_darcy_data(), problem="forward")

    sample = dataset[0]

    assert sample["input"].shape == (2, 4, 4)
    assert sample["target"].shape == (1, 4, 4)
    torch.testing.assert_close(sample["input"][0], torch.full((4, 4), 2.0))
    torch.testing.assert_close(sample["input"][1], torch.full((4, 4), 3.0))
    torch.testing.assert_close(sample["target"][0], torch.full((4, 4), 5.0))


def test_darcy_inverse_conditions_on_solution_targets_kappa_and_source():
    dataset = DarcyDataset(_raw_darcy_data(), problem="inverse", inverse_mode="both")

    sample = dataset[0]

    assert sample["input"].shape == (1, 4, 4)
    assert sample["target"].shape == (2, 4, 4)
    torch.testing.assert_close(sample["input"][0], torch.full((4, 4), 5.0))
    torch.testing.assert_close(sample["target"][0], torch.full((4, 4), 2.0))
    torch.testing.assert_close(sample["target"][1], torch.full((4, 4), 3.0))


def test_darcy_inverse_coefficient_mode_conditions_on_solution_and_source():
    dataset = DarcyDataset(
        _raw_darcy_data(),
        problem="inverse",
        inverse_mode="coefficient",
    )

    sample = dataset[0]

    assert sample["input"].shape == (2, 4, 4)
    assert sample["target"].shape == (1, 4, 4)
    torch.testing.assert_close(sample["input"][0], torch.full((4, 4), 5.0))
    torch.testing.assert_close(sample["input"][1], torch.full((4, 4), 3.0))
    torch.testing.assert_close(sample["target"][0], torch.full((4, 4), 2.0))


def test_darcy_inverse_source_mode_conditions_on_solution_and_kappa():
    dataset = DarcyDataset(
        _raw_darcy_data(),
        problem="inverse",
        inverse_mode="source",
    )

    sample = dataset[0]

    assert sample["input"].shape == (2, 4, 4)
    assert sample["target"].shape == (1, 4, 4)
    torch.testing.assert_close(sample["input"][0], torch.full((4, 4), 5.0))
    torch.testing.assert_close(sample["input"][1], torch.full((4, 4), 2.0))
    torch.testing.assert_close(sample["target"][0], torch.full((4, 4), 3.0))


def test_darcy_inverse_appends_observation_mask_to_condition():
    dataset = DarcyDataset(
        _raw_darcy_data_with_mask(),
        problem="inverse",
        inverse_mode="both",
    )

    sample = dataset[0]

    assert sample["input"].shape == (2, 4, 4)
    assert sample["target"].shape == (2, 4, 4)
    assert sample["obs_mask"].shape == (1, 4, 4)
    torch.testing.assert_close(sample["input"][0], torch.full((4, 4), 5.0))
    torch.testing.assert_close(sample["input"][1], sample["obs_mask"][0])


def test_darcy_inverse_coefficient_appends_mask_after_known_fields():
    dataset = DarcyDataset(
        _raw_darcy_data_with_mask(),
        problem="inverse",
        inverse_mode="coefficient",
    )

    sample = dataset[0]

    assert sample["input"].shape == (3, 4, 4)
    torch.testing.assert_close(sample["input"][0], torch.full((4, 4), 5.0))
    torch.testing.assert_close(sample["input"][1], torch.full((4, 4), 3.0))
    torch.testing.assert_close(sample["input"][2], sample["obs_mask"][0])
