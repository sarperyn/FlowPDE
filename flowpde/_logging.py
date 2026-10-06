"""Library-wide logging for FlowPDE.

Every module logs through a child of the ``flowpde`` logger.  Out of the box
that logger prints INFO messages (training progress, dataset summaries) to
stdout as bare lines, so scripts and notebooks see the same output that
``print`` used to give.  To change that:

    >>> import flowpde
    >>> flowpde.set_verbosity("WARNING")     # silence progress output
    >>> flowpde.disable_default_handler()    # route records to your own handlers

The default handler resolves ``sys.stdout`` at emit time rather than at
import, so redirected or captured stdout (pytest, notebooks) keeps working.
"""

from __future__ import annotations

import logging
import sys
from typing import Union

_LIBRARY_LOGGER = "flowpde"


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler bound to whatever ``sys.stdout`` is when a record is emitted."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        pass


_default_handler = _StdoutHandler()
_default_handler.setFormatter(logging.Formatter("%(message)s"))

_root = logging.getLogger(_LIBRARY_LOGGER)
_root.addHandler(_default_handler)
_root.setLevel(logging.INFO)
# The default handler already prints; propagating as well would print every
# line twice for anyone who has configured the root logger.
_root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a FlowPDE module (``__name__`` inside the package)."""
    return logging.getLogger(name)


def set_verbosity(level: Union[int, str]) -> None:
    """Set the level of all FlowPDE loggers, e.g. ``"WARNING"`` to silence progress output."""
    _root.setLevel(level.upper() if isinstance(level, str) else level)


def disable_default_handler() -> None:
    """Stop printing to stdout and propagate records to the root logger instead."""
    _root.removeHandler(_default_handler)
    _root.propagate = True


def enable_default_handler() -> None:
    """Restore the default stdout handler removed by `disable_default_handler`."""
    if _default_handler not in _root.handlers:
        _root.addHandler(_default_handler)
    _root.propagate = False
