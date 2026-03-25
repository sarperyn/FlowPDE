"""
DEPRECATED — activation_functions.py
=====================================
``Swish`` is identical to ``torch.nn.SiLU``, which has been natively
supported in PyTorch since v1.7 with a CUDA-optimised kernel.

Use ``nn.SiLU()`` directly, or call
``flowpde.models.components.get_activation("silu")`` for the factory helper.

This module is kept only for backward compatibility and will be removed
in a future version.
"""

import warnings
from torch import nn


class Swish(nn.SiLU):
    """Deprecated alias for ``torch.nn.SiLU``.  Use ``nn.SiLU()`` instead."""

    def __init__(self):
        warnings.warn(
            "flowpde.utils.activation_functions.Swish is deprecated. "
            "Use torch.nn.SiLU() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
