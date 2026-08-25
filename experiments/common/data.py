"""Dataset helpers for FlowPDE experiments."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from flowpde.datasets.exponax.burgers import BurgersConfig, BurgersGenerator
from flowpde.datasets.exponax.darcy import DarcyConfig, DarcyGenerator
from flowpde.datasets.normalization import FieldNormalizer


def build_darcy_splits(config: Dict[str, Any]):
    """Generate normalized Darcy train/val/test datasets."""
    generator_config = DarcyConfig(**config["generator"])
    generator = DarcyGenerator(config=generator_config)
    split_cfg = config["splits"]

    train = generator.generate(
        num_samples=split_cfg["train"],
        seed=split_cfg["train_seed"],
        problem="forward",
    )
    val = generator.generate(
        num_samples=split_cfg["val"],
        seed=split_cfg["val_seed"],
        problem="forward",
    )
    test = generator.generate(
        num_samples=split_cfg["test"],
        seed=split_cfg["test_seed"],
        problem="forward",
    )

    normalizer = FieldNormalizer.from_dataset(train)
    train.set_normalizer(normalizer)
    val.set_normalizer(normalizer)
    test.set_normalizer(normalizer)

    return train, val, test, normalizer


def build_burgers_splits(config: Dict[str, Any]):
    """Generate normalized Burgers train/val/test datasets."""
    generator_config = BurgersConfig(**config["generator"])
    generator = BurgersGenerator(config=generator_config)
    split_cfg = config["splits"]

    train = generator.generate(
        num_samples=split_cfg["train"],
        seed=split_cfg["train_seed"],
        problem="forward",
    )
    val = generator.generate(
        num_samples=split_cfg["val"],
        seed=split_cfg["val_seed"],
        problem="forward",
    )
    test = generator.generate(
        num_samples=split_cfg["test"],
        seed=split_cfg["test_seed"],
        problem="forward",
    )

    normalizer = FieldNormalizer.from_dataset(train)
    train.set_normalizer(normalizer)
    val.set_normalizer(normalizer)
    test.set_normalizer(normalizer)

    return train, val, test, normalizer


def build_loaders(
    train,
    val,
    test,
    loader_config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test dataloaders."""
    batch_size = loader_config["batch_size"]
    eval_batch_size = loader_config.get("eval_batch_size", batch_size)
    num_workers = loader_config.get("num_workers", 0)
    pin_memory = loader_config.get("pin_memory", False)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=loader_config.get("drop_last", False),
    )
    val_loader = DataLoader(
        val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
