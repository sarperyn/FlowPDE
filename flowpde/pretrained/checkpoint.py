"""Self-describing pretrained checkpoints: save, download, load.

A pretrained file is a ``torch.save``-d dict holding only tensors and plain
values, so it loads with ``weights_only=True``:

- ``format`` / ``format_version``: identify the file and its layout.
- ``recipe``: how to rebuild the model (see `flowpde.pretrained.recipe`).
- ``model_state``: backbone weights (the EMA average when trained with EMA).
- ``target_shape``: per-sample shape of what the flow generates, e.g. ``[1, 64]``.
- ``normalizer_state``: the training-time `FieldNormalizer`, or ``None``.
- ``data``: how the training data was generated, so matching test data can
  be regenerated (generator name, its config, problem, splits, field names).
- ``metrics`` / ``info``: benchmark numbers and free-form provenance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

import flowpde
from flowpde.datasets.normalization import FieldNormalizer
from flowpde.objectives import FlowMatchingObjective

from .recipe import build_objective, validate_recipe

logger = logging.getLogger(__name__)

FORMAT = "flowpde-pretrained"
FORMAT_VERSION = 1


# Saving


def data_spec(
    generator: Any,
    problem: str = "forward",
    inverse_mode: Optional[str] = None,
    splits: Optional[Dict[str, Any]] = None,
    dataset: Any = None,
) -> Dict[str, Any]:
    """Describe how a dataset was generated, for storing in a pretrained file.

    Args:
        generator: The `PoissonGenerator` / `BurgersGenerator` /
            `DarcyGenerator` that produced the data.
        problem: ``'forward'`` or ``'inverse'``.
        inverse_mode: Darcy's ``inverse_mode`` argument, when used.
        splits: Sizes and seeds of the train/val/test splits.
        dataset: A generated dataset, to record its input and target fields.
    """
    spec: Dict[str, Any] = {
        "generator": type(generator).__name__,
        "config": dataclasses.asdict(generator.config),
        "problem": problem,
        "inverse_mode": inverse_mode,
        "splits": dict(splits or {}),
    }
    if dataset is not None:
        spec["input_fields"] = list(getattr(dataset, "input_fields", []))
        spec["target_fields"] = list(getattr(dataset, "target_fields", []))
    return spec


def save_pretrained(
    path: Union[str, os.PathLike],
    objective: FlowMatchingObjective,
    recipe: Dict[str, Any],
    target_shape: Sequence[int],
    normalizer: Optional[FieldNormalizer] = None,
    data: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``objective``'s backbone weights and everything needed to reuse them.

    Pass the objective with the weights you want to ship — under EMA training,
    load the averaged weights first (e.g. ``Trainer.load_checkpoint(...,
    resume_training=False)`` on ``best_model.pt``).

    Args:
        path: Output file, conventionally ``<name>.pt``.
        objective: Trained objective whose ``model`` weights are saved.
        recipe: The recipe ``objective`` was built from.
        target_shape: Per-sample shape of the generated field, e.g. ``(1, 64, 64)``.
        normalizer: Training-time normalizer, so outputs map back to physical units.
        data: Output of `data_spec`.
        metrics: Evaluation results to ship with the weights.
        info: Free-form provenance (training budget, hardware, git commit, ...).

    Returns:
        The written path.
    """
    validate_recipe(recipe)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in objective.model.state_dict().items()
    }
    checkpoint = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "flowpde_version": flowpde.__version__,
        "recipe": _plain(recipe),
        "model_state": state,
        "target_shape": [int(s) for s in target_shape],
        "normalizer_state": _plain(normalizer.state_dict()) if normalizer is not None else None,
        "data": _plain(data),
        "metrics": _plain(metrics or {}),
        "info": _plain(info or {}),
    }
    torch.save(checkpoint, path)
    return path


def _plain(value: Any) -> Any:
    """Convert to str/int/float/bool/None/list/dict so ``weights_only`` loading accepts it.

    Catches the easy mistakes: ``torch.__version__`` (a ``str`` subclass),
    tuples, numpy scalars and 0-d tensors.
    """
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    raise TypeError(
        f"Cannot store {type(value).__name__} in a pretrained file; use plain "
        "numbers, strings, lists and dicts."
    )


# Loading


@dataclass
class PretrainedModel:
    """A loaded pretrained model, ready to sample.

    Attributes:
        objective: The rebuilt `FlowMatchingObjective`, in eval mode.
        normalizer: Training-time normalizer, or ``None``.
        target_shape: Per-sample shape of generated fields.
        recipe: The recipe the model was built from.
        data: How the training data was generated (see `data_spec`).
        metrics: Evaluation results shipped with the weights.
        info: Provenance recorded at save time.
    """

    objective: FlowMatchingObjective
    normalizer: Optional[FieldNormalizer]
    target_shape: List[int]
    recipe: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def flow(self):
        return self.objective.flow

    @property
    def device(self) -> torch.device:
        return next(self.objective.model.parameters()).device

    def sample(
        self,
        condition: Tensor,
        n_steps: int = 50,
        solver: str = "euler",
        physical_units: bool = True,
        **solver_kwargs: Any,
    ) -> Tensor:
        """Draw one sample per condition, shaped ``(B, *target_shape)``.

        Args:
            condition: Normalized conditioning input ``(B, C, ...)``, as
                returned by a dataset with the model's normalizer set (see
                `make_dataset`).
            n_steps: Integration steps for fixed-step solvers.
            solver: ODE solver name, e.g. ``'euler'``, ``'rk4'``, ``'dopri5'``.
            physical_units: Map the sample back to physical units with the
                model's normalizer.  Ignored when there is no normalizer.
        """
        condition = condition.to(self.device)
        flat = self.objective.sample(
            condition, n_steps=n_steps, solver=solver, **solver_kwargs
        )
        samples = flat.view(flat.shape[0], *self.target_shape)
        fields = (self.data or {}).get("target_fields")
        if physical_units and self.normalizer is not None and fields:
            samples = self.normalizer.denormalize_channels(fields, samples)
        return samples

    def make_dataset(self, num_samples: int, seed: int):
        """Generate fresh data from the training distribution, normalized for this model.

        Requires the ``data`` extra (JAX and Exponax).  Use a seed different
        from the recorded ``data['splits']`` seeds to get unseen samples.
        """
        if not self.data:
            raise ValueError("This checkpoint does not record how its data was generated.")
        from flowpde import datasets

        generator = getattr(datasets, self.data["generator"])(**self.data["config"])
        kwargs = {"num_samples": num_samples, "seed": seed, "problem": self.data["problem"]}
        if self.data.get("inverse_mode"):
            kwargs["inverse_mode"] = self.data["inverse_mode"]
        dataset = generator.generate(**kwargs)
        if self.normalizer is not None:
            dataset.set_normalizer(self.normalizer)
        return dataset


def load_pretrained(
    name_or_path: Union[str, os.PathLike],
    device: Optional[Union[str, torch.device]] = None,
    sha256: Optional[str] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> PretrainedModel:
    """Load a pretrained model by registry name, local path or URL.

    Args:
        name_or_path: A name from `list_pretrained`, a local ``.pt`` file, or an
            ``https://`` / ``file://`` URL.
        device: Device to place the model on; defaults to CPU.
        sha256: Expected checksum for a URL.  Registry entries carry their own.
        cache_dir: Download cache; defaults to `default_cache_dir`.

    Returns:
        A `PretrainedModel` in eval mode.

    Example:
        >>> from flowpde.pretrained import load_pretrained
        >>> model = load_pretrained("path/to/burgers_forward.pt")
        >>> test = model.make_dataset(num_samples=16, seed=2024)
        >>> batch = next(iter(DataLoader(test, batch_size=4)))
        >>> u_T = model.sample(batch["input"])        # physical units
    """
    path = _resolve(str(name_or_path), sha256=sha256, cache_dir=cache_dir)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _check_format(checkpoint, path)

    objective = build_objective(checkpoint["recipe"])
    objective.model.load_state_dict(checkpoint["model_state"])
    target_shape = list(checkpoint["target_shape"])
    dim = 1
    for size in target_shape:
        dim *= size
    objective.flow.set_target_dim(dim)
    objective.to(torch.device(device) if device is not None else torch.device("cpu"))
    objective.eval()

    normalizer_state = checkpoint.get("normalizer_state")
    return PretrainedModel(
        objective=objective,
        normalizer=FieldNormalizer.from_state_dict(normalizer_state) if normalizer_state else None,
        target_shape=target_shape,
        recipe=checkpoint["recipe"],
        data=checkpoint.get("data"),
        metrics=checkpoint.get("metrics", {}),
        info=checkpoint.get("info", {}),
    )


def _check_format(checkpoint: Any, path: Path) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a FlowPDE pretrained file. Training checkpoints from "
            "Trainer are loaded with Trainer.load_checkpoint; convert one with "
            "flowpde.pretrained.save_pretrained."
        )
    version = checkpoint.get("format_version", 0)
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{path} uses pretrained format version {version}, newer than this "
            f"flowpde ({flowpde.__version__}) supports ({FORMAT_VERSION}). "
            "Upgrade with: pip install -U flowpde"
        )


# Registry and downloads


def registry() -> Dict[str, Dict[str, Any]]:
    """The published pretrained models shipped with this version of flowpde."""
    text = resources.files("flowpde.pretrained").joinpath("registry.json").read_text()
    return json.loads(text)


def list_pretrained() -> List[str]:
    """Names accepted by `load_pretrained`."""
    return sorted(registry())


def default_cache_dir() -> Path:
    """``$FLOWPDE_CACHE``, else ``~/.cache/flowpde/pretrained``."""
    root = os.environ.get("FLOWPDE_CACHE")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "flowpde" / "pretrained"


def _resolve(name: str, sha256: Optional[str], cache_dir) -> Path:
    entries = registry()
    if name in entries:
        entry = entries[name]
        return _download(entry["url"], entry["sha256"], cache_dir)
    if "://" in name:
        return _download(name, sha256, cache_dir)
    path = Path(name).expanduser()
    if path.is_file():
        if sha256 is not None:
            _verify(path, sha256)
        return path
    available = ", ".join(sorted(entries)) or "none published yet"
    raise FileNotFoundError(
        f"'{name}' is neither a registered pretrained model ({available}), "
        "a local file, nor a URL."
    )


def _download(url: str, sha256: Optional[str], cache_dir) -> Path:
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / Path(urllib.request.url2pathname(url.rsplit("/", 1)[-1]))

    if target.is_file():
        if sha256 is None or _sha256(target) == sha256:
            return target
        logger.warning("Cached %s fails its checksum; downloading again.", target.name)

    logger.info("Downloading %s to %s", url, target)
    fd, tmp = tempfile.mkstemp(dir=cache, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, out)
        if sha256 is not None:
            _verify(Path(tmp), sha256, source=url)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, sha256: str, source: Optional[str] = None) -> None:
    actual = _sha256(path)
    if actual != sha256:
        raise ValueError(
            f"Checksum mismatch for {source or path}: expected sha256 {sha256}, "
            f"got {actual}. The file is corrupt or has been replaced."
        )
