#!/usr/bin/env python3
"""Run a per-leaf pipeline script over every leaf under a data root."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def iter_leaves(root: Path, marker: str):
    for marker_path in sorted(root.rglob(marker)):
        leaf = marker_path.parent
        yield leaf, leaf.relative_to(root).as_posix()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a per-leaf pipeline script over every leaf under a data root.")
    p.add_argument("--root", required=True)
    p.add_argument("--marker", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--input-arg", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--py", default=sys.executable)
    p.add_argument("--prefix", default=None)
    p.add_argument("extra", nargs=argparse.REMAINDER)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra

    root = Path(a.root)
    if not root.is_dir():
        print(f"run_leaves: root {root} does not exist", file=sys.stderr)
        return 1

    leaves = list(iter_leaves(root, a.marker))
    if not leaves:
        print(f"run_leaves: no leaves with marker {a.marker!r} under {root}", file=sys.stderr)
        return 1

    failures = 0
    for leaf, rel in leaves:
        out_dir = Path(a.out_root) / rel
        print(f"== {a.out_root}/{rel}")
        cmd = [
            a.py, a.script,
            f"--{a.input_arg}", str(leaf / a.marker),
            "--out", str(out_dir),
            *extra,
        ]
        if a.prefix is not None:
            cmd.extend(["--prefix", a.prefix.replace("{rel}", rel.replace("/", "_"))])
        if subprocess.run(cmd).returncode != 0:
            print(f"run_leaves: {a.script} failed for {rel}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
