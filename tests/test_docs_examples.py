"""Run the Python examples in the README and getting-started docs.

Each page's ```python blocks are executed in order in one namespace, the way a
reader would paste them.  Sizes are shrunk first (samples, grid, epochs,
channels, steps) so this checks that the documented API still exists and fits
together, not that the documented run converges.  A block directly preceded by
an HTML comment starting ``<!-- docs-test: skip`` is left out, for examples
that need files a test cannot provide.
"""

import re
from pathlib import Path

import pytest

import flowpde

ROOT = Path(__file__).resolve().parents[1]

PAGES = [
    "README.md",
    "docs/index.md",
    "docs/getting_started/installation.md",
    "docs/getting_started/quickstart.md",
]

# (pattern, replacement) applied to every block before it runs.
SHRINK = [
    (r"num_samples=\d+", "num_samples=8"),
    (r"num_points=\d+", "num_points=16"),
    (r"spatial_size=\d+", "spatial_size=16"),
    (r"base_channels=\d+", "base_channels=8"),
    (r"batch_size=\d+", "batch_size=4"),
    (r"\bepochs=\d+", "epochs=1"),
    (r"epochs_per_iteration=\d+", "epochs_per_iteration=1"),
    (r"num_iterations=\d+", "num_iterations=1"),
    (r"\bn_steps=\d+", "n_steps=2"),
    (r"\bnum_steps=\d+", "num_steps=5"),
    (r'device="cuda"', 'device="cpu"'),
    (r'solver="dopri5"', 'solver="euler"'),
]


def python_blocks(page: str) -> list[str]:
    text = (ROOT / page).read_text()
    blocks = re.findall(
        r"^(<!-- docs-test: skip.*?-->\n)?```python\n(.*?)^```", text, flags=re.S | re.M
    )
    return [code for skip, code in blocks if not skip]


def shrink(code: str) -> str:
    for pattern, replacement in SHRINK:
        code = re.sub(pattern, replacement, code)
    return code


@pytest.fixture(autouse=True)
def restore_logging():
    yield
    flowpde.enable_default_handler()
    flowpde.set_verbosity("INFO")


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_page_examples_run(page, tmp_path, monkeypatch):
    blocks = python_blocks(page)
    assert blocks, f"{page} has no ```python blocks"

    monkeypatch.chdir(tmp_path)  # examples write results/... relative paths
    namespace: dict = {"__name__": "__docs_example__"}
    for i, block in enumerate(blocks, start=1):
        code = compile(shrink(block), f"{page} [python block {i}]", "exec")
        exec(code, namespace)
