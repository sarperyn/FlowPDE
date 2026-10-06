"""Register trained models in flowpde's pretrained registry.

Examples:
    python -m benchmarks.publish benchmarks/runs/full --tag weights-v1

For every ``<task>/<task>.pt`` under the given folders this computes the
sha256, adds an entry to ``flowpde/pretrained/registry.json`` pointing at the
GitHub release asset ``https://github.com/<repo>/releases/download/<tag>/<file>``,
and prints the ``gh`` command that uploads the files to that release.

Registry names are ``<task>_<backbone>``, e.g. ``burgers_forward_convnet``.
Smoke-scale models are refused unless ``--allow-smoke`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import List, Optional

REGISTRY = Path(__file__).resolve().parents[1] / "flowpde" / "pretrained" / "registry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.publish",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("roots", nargs="+", type=Path, help="run folders to publish")
    parser.add_argument("--tag", required=True, help="GitHub release tag, e.g. weights-v1")
    parser.add_argument("--repo", default="sarperyn/FlowPDE", help="owner/name on GitHub")
    parser.add_argument("--registry", type=Path, default=REGISTRY, help="registry file to update")
    parser.add_argument("--allow-smoke", action="store_true",
                        help="also publish smoke-scale models (for testing only)")
    args = parser.parse_args(argv)

    registry = json.loads(args.registry.read_text()) if args.registry.exists() else {}
    files = []
    for root in args.roots:
        for metrics_path in sorted(root.rglob("metrics.json")):
            run = json.loads(metrics_path.read_text())
            if run["scale"] == "smoke" and not args.allow_smoke:
                print(f"skip {run['task']}: smoke scale (pass --allow-smoke to include)")
                continue
            weights = metrics_path.parent / f"{run['task']}.pt"
            if not weights.is_file():
                print(f"skip {run['task']}: {weights} is missing")
                continue

            name = f"{run['task']}_{run['backbone'].lower()}"
            asset = f"{name}.pt"
            registry[name] = {
                "url": f"https://github.com/{args.repo}/releases/download/{args.tag}/{asset}",
                "sha256": sha256(weights),
                "task": run["task"],
                "description": run["description"],
                "scale": run["scale"],
                "test_rel_l2": run["test"]["rel_l2"],
            }
            files.append((weights, asset))
            print(f"registered {name}  rel_l2={run['test']['rel_l2']:.4f}  ({run['scale']})")

    if not files:
        print("Nothing to publish.")
        return 1

    args.registry.write_text(json.dumps(dict(sorted(registry.items())), indent=2) + "\n")
    print(f"\nUpdated {args.registry}")

    # Release assets take the uploaded file's name, so upload copies named
    # after their registry entries.
    copies = " && ".join(
        f"cp {shlex.quote(str(src))} {shlex.quote(asset)}" for src, asset in files
    )
    assets = " ".join(shlex.quote(asset) for _, asset in files)
    print("\nUpload the weights (from an empty folder), then commit the registry:\n")
    print(f"  {copies}")
    print(f"  gh release create {args.tag} --repo {args.repo} "
          f"--title 'Pretrained weights ({args.tag})' --notes 'See docs/benchmarks.md' {assets}")
    print("\n  # or, to add to an existing release:")
    print(f"  gh release upload {args.tag} --repo {args.repo} {assets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
