#!/usr/bin/env python
"""Leaf-folder orchestrator for the synthetic-wrench pipeline.

Several pipeline steps (images, label, postprocess, visualize) operate per
"leaf" — a directory under a data root that contains a marker file
(`prompts.txt`, `manifest.jsonl`, or `labels.jsonl`).  The same
"walk the root, find leaves, mirror the relative path into an output root,
spawn the per-leaf script" loop was previously inlined as Python heredocs in
tasks/pipeline.yml.  It lives here instead so the task file stays declarative.

Usage:
  python pipeline/run_leaves.py \
      --root data/prompts --marker prompts.txt \
      --out-root data/images --input-arg prompts \
      --script pipeline/generate_images.py \
      -- --model black-forest-labs/FLUX.2-klein-4B --size 768 --steps 8

For each leaf `<root>/.../<rel>/<marker>` this runs:
  <py> <script> <input-arg> <leaf>/<marker> --out <out-root>/<rel> [extra...]

`--limit N` (for test variants) and any other flags passed after `--` are
forwarded verbatim to every per-leaf invocation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def iter_leaves(root: Path, marker: str):
    """Yield (leaf_dir, rel_posix) for every dir under root containing marker."""
    for marker_path in sorted(root.rglob(marker)):
        leaf = marker_path.parent
        yield leaf, leaf.relative_to(root).as_posix()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a per-leaf pipeline script over every leaf under a data root.",
    )
    p.add_argument("--root", required=True, help="data root to walk (e.g. data/prompts)")
    p.add_argument("--marker", required=True,
                   help="filename that identifies a leaf (e.g. prompts.txt, manifest.jsonl)")
    p.add_argument("--out-root", required=True, help="output root; leaf rel-path is appended")
    p.add_argument("--input-arg", required=True,
                   help="per-leaf script flag (bare name, e.g. 'prompts'); '--' is prepended")
    p.add_argument("--script", required=True, help="per-leaf python script to run")
    p.add_argument("--py", default=sys.executable, help="python interpreter to use")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="extra args forwarded to each leaf (use `--` to separate)")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    # argparse REMAINDER keeps a leading `--`; drop it so callers can pass `-- --limit 3`.
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra

    root = Path(a.root)
    if not root.is_dir():
        print(f"run_leaves: root {root} does not exist or is not a dir", file=sys.stderr)
        return 1

    leaves = list(iter_leaves(root, a.marker))
    if not leaves:
        print(f"run_leaves: no leaves with marker {a.marker!r} under {root}", file=sys.stderr)
        return 1

    failures = 0
    for leaf, rel in leaves:
        marker_path = leaf / a.marker
        out_dir = Path(a.out_root) / rel
        print(f"== {a.out_root}/{rel}")
        cmd = [
            a.py, a.script,
            f"--{a.input_arg}", str(marker_path),
            "--out", str(out_dir),
            *extra,
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"run_leaves: {a.script} failed for {rel} (rc={rc})", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
