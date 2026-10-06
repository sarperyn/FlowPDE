# Contributing to FlowPDE

Thanks for your interest in improving FlowPDE. Bug reports, documentation fixes,
new PDE generators and new flow components are all welcome.

## Reporting bugs and asking for features

Open an [issue](https://github.com/sarperyn/FlowPDE/issues/new/choose). For a bug,
include a minimal script that reproduces it, the full traceback, and the output of:

```bash
python -c "import flowpde, torch, sys; print(flowpde.__version__, torch.__version__, sys.version)"
```

## Development setup

FlowPDE uses [uv](https://docs.astral.sh/uv/). One command installs the package in
editable mode with the `data` extra, pytest and ruff, at the versions in `uv.lock`:

```bash
git clone https://github.com/sarperyn/FlowPDE.git
cd FlowPDE
uv sync
```

With pip instead: `pip install -e ".[data]" pytest ruff`.

## Before opening a pull request

```bash
uv run ruff check flowpde tests          # lint (add --fix for import order etc.)
uv run -m pytest -m "not slow"           # fast tests, ~15 s
uv run -m pytest                         # full suite, incl. Exponax and docs examples
```

CI runs ruff and the full suite on Python 3.11, 3.12 and 3.13, and checks that the
package builds and imports without the optional `data` extra.

Guidelines:

- **Tests.** New behaviour comes with a test; a bug fix comes with a test that
  failed before it.
- **Docstrings.** Public classes and functions use Google-style docstrings; the API
  reference is generated from them by mkdocstrings.
- **Docs examples are tested.** The Python blocks in `README.md` and
  `docs/getting_started/` are executed by `tests/test_docs_examples.py`, so keep
  them runnable.
- **Optional dependencies.** Code outside `flowpde.datasets.exponax` must not import
  JAX or Exponax.
- **Output.** Log through `logging.getLogger(__name__)` rather than `print`, so users
  can silence it with `flowpde.set_verbosity`.
- **Changelog.** Add a line under `## [Unreleased]` in `CHANGELOG.md` for anything a
  user would notice.

To preview the documentation locally:

```bash
uv run --with ".[docs]" mkdocs serve
```

## Releasing (maintainers)

1. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new version heading.
2. Set `__version__` in `flowpde/__init__.py` and `version` / `date-released` in
   `CITATION.cff`.
3. Commit, then tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.

The [release workflow](.github/workflows/release.yml) checks that the tag matches
`__version__`, publishes to PyPI with trusted publishing and creates the GitHub
release, which Zenodo archives. Running the workflow by hand publishes to TestPyPI
instead. PyPI never accepts the same version twice.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
