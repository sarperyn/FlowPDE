"""Tests for library logging: default stdout output and how users silence or reroute it."""

import logging

import pytest

import flowpde
from flowpde.utils import print_stats


@pytest.fixture(autouse=True)
def restore_logging():
    yield
    flowpde.enable_default_handler()
    flowpde.set_verbosity("INFO")


def test_default_output_goes_to_stdout(capsys):
    print_stats(Epoch="0001/10", Train_Loss=0.5)
    assert capsys.readouterr().out == "Epoch: 0001/10 | Train_Loss: 0.5\n"


def test_set_verbosity_silences_progress(capsys):
    flowpde.set_verbosity("WARNING")
    print_stats(Train_Loss=0.5)
    assert capsys.readouterr().out == ""


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def root_handler():
    # A plain handler on the root logger: pytest's caplog also attaches to
    # non-propagating loggers, so it cannot tell whether records propagate.
    handler = _ListHandler()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield handler
    root.removeHandler(handler)
    root.setLevel(old_level)


def test_disable_default_handler_propagates_to_root(capsys, root_handler):
    flowpde.disable_default_handler()
    print_stats(Train_Loss=0.5)
    assert capsys.readouterr().out == ""
    assert root_handler.messages == ["Train_Loss: 0.5"]


def test_default_handler_does_not_duplicate_through_root(capsys, root_handler):
    print_stats(Train_Loss=0.5)
    assert capsys.readouterr().out == "Train_Loss: 0.5\n"
    assert root_handler.messages == []
