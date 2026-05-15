import pytest
import torch

from flowpde.core.base_conditioner import FiLMConditioner
from flowpde.datasets.exponax.base import PDEDataset
from flowpde.models.mlp import MLP


def test_pde_dataset_appends_observation_mask_to_inverse_condition():
    data = {
        "source": torch.full((2, 1, 4), 2.0),
        "solution": torch.full((2, 1, 4), 5.0),
        "obs_mask": torch.tensor([[[1.0, 0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0, 1.0]]]),
    }
    dataset = PDEDataset(data, problem="inverse")

    sample = dataset[0]

    assert sample["input"].shape == (2, 4)
    assert sample["target"].shape == (1, 4)
    torch.testing.assert_close(sample["input"][0], torch.full((4,), 5.0))
    torch.testing.assert_close(sample["input"][1], sample["obs_mask"][0])


def test_mlp_rejects_input_level_film_conditioner():
    conditioner = FiLMConditioner(condition_dim=2, feature_dim=3)

    with pytest.raises(ValueError, match="FiLMConditioner"):
        MLP(input_dim=3, condition_dim=2, conditioner=conditioner)
