"""Build the benchmark table from recorded runs.

Examples:
    python -m benchmarks.report benchmarks/runs/full
    python -m benchmarks.report benchmarks/runs/full --write docs/benchmarks.md

Reads every ``metrics.json`` under the given folders and prints a Markdown
table.  With ``--write``, replaces the table between the
``<!-- benchmark-table:start -->`` and ``<!-- benchmark-table:end -->``
markers in that file, so the docs only ever show recorded numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

START = "<!-- benchmark-table:start -->"
END = "<!-- benchmark-table:end -->"


def load_runs(roots: List[Path]) -> List[Dict[str, Any]]:
    runs = []
    for root in roots:
        for path in sorted(Path(root).rglob("metrics.json")):
            runs.append(json.loads(path.read_text()))
    order = {"full": 0, "quick": 1, "smoke": 2}
    return sorted(runs, key=lambda r: (order.get(r["scale"], 9), r["task"]))


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _params(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def markdown_table(runs: List[Dict[str, Any]]) -> str:
    if not runs:
        return "_No benchmark runs recorded yet._"
    lines = [
        "| Task | Grid | Backbone | Train / epochs | Test rel. L2 | Test H1 | "
        "Sampler | ms / sample | Device | Scale |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        test, sampling, env = run["test"], run["sampling"], run["environment"]
        lines.append(
            f"| `{run['task']}` | {run['grid']} "
            f"| {run['backbone']} ({_params(run['parameters'])}) "
            f"| {run['train_samples']} / {run['epochs']} "
            f"| {_fmt(test.get('rel_l2'))} | {_fmt(test.get('h1'))} "
            f"| {sampling['solver']}, {sampling['n_steps']} steps "
            f"| {sampling['seconds_per_sample'] * 1e3:.2f} "
            f"| {env['device']} | {run['scale']} |"
        )
    return "\n".join(lines)


def write_table(path: Path, table: str) -> None:
    text = path.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{path} has no {START} ... {END} markers to replace.")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(f"{head}{START}\n{table}\n{END}{tail}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.report",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("roots", nargs="+", type=Path, help="run folders to collect")
    parser.add_argument("--write", type=Path, help="Markdown file whose table to replace")
    args = parser.parse_args(argv)

    table = markdown_table(load_runs(args.roots))
    if args.write:
        write_table(args.write, table)
        print(f"Updated {args.write}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
