"""
Field Normalization for PDE Datasets
=====================================

Flow matching transports a standard Gaussian base distribution to the data
distribution.  When the data has a very different scale from ``N(0, I)`` the
velocity targets ``x_1 - x_0`` inherit the raw data magnitude, which makes the
regression problem badly conditioned.  Standardizing each PDE field to roughly
zero mean and unit variance removes that mismatch.

Statistics must always be fitted on the **training split** and then reused
verbatim for validation/test data, otherwise the evaluation leaks information
about the held-out set::

    train_ds = generator.generate(num_samples=1000, seed=0)
    test_ds  = generator.generate(num_samples=200, seed=1)

    normalizer = FieldNormalizer.from_dataset(train_ds)
    train_ds.set_normalizer(normalizer)
    test_ds.set_normalizer(normalizer)      # same statistics, not refitted

Metrics should be reported in physical units, so predictions are mapped back
with :meth:`FieldNormalizer.denormalize` (or
:meth:`FieldNormalizer.denormalize_channels` for targets that concatenate
several fields) before computing errors.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor


class FieldNormalizer:
    """
    Per-field mean/std standardization keyed by PDE field name.

    Fields are addressed by their raw name (``'source'``, ``'solution'``,
    ``'kappa'``, ``'initial'``, ``'final'``) rather than by their role in the
    learning problem, so a single normalizer stays correct when the same data
    is used for both the forward and the inverse direction.

    Args:
        stats: Mapping ``{field_name: {'mean': float, 'std': float}}``.
        eps: Floor applied to standard deviations to avoid division by zero.

    Example:
        >>> normalizer = FieldNormalizer({'solution': {'mean': 2.0, 'std': 4.0}})
        >>> z = normalizer.normalize('solution', torch.tensor([6.0]))
        >>> z
        tensor([1.])
        >>> normalizer.denormalize('solution', z)
        tensor([6.])
    """

    def __init__(
        self,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        eps: float = 1e-8,
    ):
        self.eps = eps
        self.stats: Dict[str, Dict[str, float]] = {}
        if stats:
            for name, field_stats in stats.items():
                self.add_field(
                    name,
                    mean=float(field_stats["mean"]),
                    std=float(field_stats["std"]),
                )

    # Construction

    @classmethod
    def from_dataset(
        cls,
        dataset,
        fields: Optional[Iterable[str]] = None,
        eps: float = 1e-8,
    ) -> "FieldNormalizer":
        """
        Build a normalizer from statistics a dataset already carries.

        Generators compute per-field mean/std at construction time and store
        them in ``metadata['stats']``, so no second pass over the data is
        needed.

        Args:
            dataset: Dataset exposing ``get_stats()`` (``PDEDataset``,
                ``DarcyDataset``).
            fields: Restrict to these field names.  Defaults to every field
                that has statistics, excluding ``'obs_mask'``.
            eps: Floor applied to standard deviations.

        Returns:
            A fitted ``FieldNormalizer``.
        """
        if not hasattr(dataset, "get_stats"):
            raise TypeError(
                f"{type(dataset).__name__} does not expose get_stats(); "
                "pass statistics to FieldNormalizer(...) directly."
            )

        stats = dataset.get_stats()
        if not stats:
            raise ValueError(
                "Dataset carries no normalization statistics. It was probably "
                "not built through a generator's wrap_dataset()."
            )

        selected = list(fields) if fields is not None else [
            name for name in stats if name != "obs_mask"
        ]

        missing = [name for name in selected if name not in stats]
        if missing:
            raise KeyError(
                f"Dataset has no statistics for field(s): {missing}. "
                f"Available: {sorted(stats)}"
            )

        return cls({name: stats[name] for name in selected}, eps=eps)

    @classmethod
    def from_tensors(
        cls,
        tensors: Dict[str, Tensor],
        eps: float = 1e-8,
    ) -> "FieldNormalizer":
        """Fit directly from raw tensors, one entry per field."""
        stats = {
            name: {"mean": tensor.mean().item(), "std": tensor.std().item()}
            for name, tensor in tensors.items()
        }
        return cls(stats, eps=eps)

    def add_field(self, name: str, mean: float, std: float) -> None:
        """Register statistics for a single field."""
        self.stats[name] = {"mean": float(mean), "std": max(float(std), self.eps)}

    # Application

    @property
    def fields(self) -> List[str]:
        """Names of all fields this normalizer knows about."""
        return sorted(self.stats)

    def has_field(self, name: str) -> bool:
        return name in self.stats

    def normalize(self, name: str, tensor: Tensor) -> Tensor:
        """
        Standardize a field to zero mean and unit variance.

        Fields with no registered statistics pass through unchanged, so
        auxiliary channels such as ``obs_mask`` stay binary.
        """
        if name not in self.stats:
            return tensor
        field_stats = self.stats[name]
        return (tensor - field_stats["mean"]) / field_stats["std"]

    def denormalize(self, name: str, tensor: Tensor) -> Tensor:
        """Map a standardized field back to physical units."""
        if name not in self.stats:
            return tensor
        field_stats = self.stats[name]
        return tensor * field_stats["std"] + field_stats["mean"]

    def denormalize_channels(
        self,
        names: Sequence[str],
        tensor: Tensor,
        channel_dim: int = 1,
    ) -> Tensor:
        """
        Denormalize a tensor whose channels concatenate several fields.

        Used for targets such as Darcy's ``inverse_mode='both'``, where the
        target is ``cat([kappa, source])``.

        Args:
            names: Field name per channel group, in channel order.
            tensor: Tensor with ``len(names)`` equal-sized channel groups.
            channel_dim: Dimension holding the channels (default: 1).

        Returns:
            Tensor of the same shape, in physical units.
        """
        if len(names) == 1:
            return self.denormalize(names[0], tensor)

        num_channels = tensor.shape[channel_dim]
        if num_channels % len(names) != 0:
            raise ValueError(
                f"Cannot split {num_channels} channels evenly across "
                f"{len(names)} fields ({list(names)})."
            )

        chunks = torch.chunk(tensor, len(names), dim=channel_dim)
        restored = [
            self.denormalize(name, chunk) for name, chunk in zip(names, chunks)
        ]
        return torch.cat(restored, dim=channel_dim)

    # Serialization

    def state_dict(self) -> Dict[str, object]:
        """Serializable state, suitable for storing next to model weights."""
        return {"stats": {k: dict(v) for k, v in self.stats.items()}, "eps": self.eps}

    def load_state_dict(self, state: Dict[str, object]) -> "FieldNormalizer":
        """Restore statistics saved by :meth:`state_dict`."""
        self.eps = float(state.get("eps", 1e-8))
        self.stats = {}
        for name, field_stats in state["stats"].items():
            self.add_field(name, field_stats["mean"], field_stats["std"])
        return self

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "FieldNormalizer":
        """Rebuild a normalizer from a serialized state."""
        return cls().load_state_dict(state)

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{name}(mean={s['mean']:.4g}, std={s['std']:.4g})"
            for name, s in sorted(self.stats.items())
        )
        return f"FieldNormalizer({entries})"
