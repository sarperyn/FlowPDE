"""Pretrained FlowPDE models.

Load a published model by name, a local file, or a URL::

    from torch.utils.data import DataLoader
    from flowpde.pretrained import list_pretrained, load_pretrained

    print(list_pretrained())
    model = load_pretrained("burgers_forward_convnet")
    test = model.make_dataset(num_samples=16, seed=2024)   # needs flowpde[data]
    batch = next(iter(DataLoader(test, batch_size=4)))
    u_T = model.sample(batch["input"])                     # physical units

Files are downloaded once into `default_cache_dir` and checked against the
sha256 in the registry.  To produce pretrained files, train with the scripts in
the repository's ``benchmarks/`` folder, or call `save_pretrained` yourself.
"""

from .checkpoint import (
    FORMAT_VERSION,
    PretrainedModel,
    data_spec,
    default_cache_dir,
    list_pretrained,
    load_pretrained,
    registry,
    save_pretrained,
)
from .recipe import build_objective, validate_recipe

__all__ = [
    "FORMAT_VERSION",
    "PretrainedModel",
    "build_objective",
    "data_spec",
    "default_cache_dir",
    "list_pretrained",
    "load_pretrained",
    "registry",
    "save_pretrained",
    "validate_recipe",
]
