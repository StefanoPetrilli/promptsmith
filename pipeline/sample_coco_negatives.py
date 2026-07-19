#!/usr/bin/env python3
"""Stage 1b — pull real negative images from COCO val2017 into the synthetic tree.

Pure negatives (frames guaranteed to contain *no* wrench, no confuser) are cheap to
source from a large real corpus, so we no longer render them synthetically. This step
deterministically samples N images from COCO val2017 and writes them into the same
synthetic-image leaf the rest of the pipeline already consumes:

    data/synthetic/images/pure_negative/<stem>.<ext>
    data/synthetic/images/pure_negative/manifest.jsonl   (same schema as stage 2)

Downstream stages run unchanged on that leaf:

    label       — SAM 3 finds no "wrench" -> empty YOLO label (background)
    visualize   — overlay (no boxes) for review
    handpick    — human discards the rare COCO frame that *does* contain a wrench
    postprocess — Albumentations degradation
    train       — included as negative (empty label)

No COCO annotations are needed: we assume wrench-free, and the handpick step removes
the occasional collision. The full COCO val2017 set is ~5k images / ~1 GB and is
downloaded once to ``data/coco/`` (gitignored, regenerable).

Because these negatives are essentially free, the default ``--n`` is doubled relative
to the previous synthetic ``pure_negative`` count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("coco-neg")

COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMG_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_OUT = "data/synthetic/images/pure_negative"
DEFAULT_COCO_ROOT = "data/coco"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample real negative images from COCO val2017 (stage 1b).")
    p.add_argument("--n", type=int, default=40,
                   help="number of COCO images to sample (default: 40, doubled vs the previous "
                        "synthetic pure_negative count of 20)")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="destination synthetic-image leaf (default: %(default)s)")
    p.add_argument("--coco-root", default=DEFAULT_COCO_ROOT,
                   help="local cache for COCO val2017 images (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="sample only N images (test mode); overrides --n when > 0")
    p.add_argument("--copy", action="store_true",
                   help="copy images into --out instead of symlinking (uses disk)")
    p.add_argument("--no-download", action="store_true",
                   help="do not download COCO; fail if the cache is missing")
    return p.parse_args()


def ensure_coco(coco_root: Path, no_download: bool) -> Path:
    """Make sure COCO val2017 images are available under coco_root/val2017/."""
    val_dir = coco_root / "val2017"
    images = sorted(p for p in val_dir.glob("*") if p.suffix.lower() in IMG_EXTS) if val_dir.is_dir() else []
    if images:
        return val_dir

    if no_download:
        raise FileNotFoundError(
            f"COCO val2017 not found under {val_dir} and --no-download was set")

    coco_root.mkdir(parents=True, exist_ok=True)
    zip_path = coco_root / "val2017.zip"
    if not zip_path.exists():
        log.info("Downloading COCO val2017.zip (~1 GB) from %s ...", COCO_VAL2017_URL)
        urllib.request.urlretrieve(COCO_VAL2017_URL, zip_path)

    log.info("Extracting %s -> %s ...", zip_path, coco_root)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(coco_root)

    images = sorted(p for p in val_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise RuntimeError(f"extraction produced no images under {val_dir}")
    log.info("COCO val2017 ready: %d images in %s", len(images), val_dir)
    return val_dir


def index_files(paths: list[Path]) -> str:
    """Stable content fingerprint of the sorted file list (for resume de-dupe)."""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
    return h.hexdigest()[:12]


def main() -> int:
    a = parse_args()
    n = a.limit if a.limit > 0 else a.n
    if n <= 0:
        log.error("--n must be > 0 (got %d)", n)
        return 2

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_dir = ensure_coco(Path(a.coco_root), a.no_download)
    pool = sorted(p for p in val_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not pool:
        log.error("no COCO images found in %s", val_dir)
        return 2
    if n > len(pool):
        log.warning("requested %d but only %d COCO images available; using all", n, len(pool))
        n = len(pool)

    rng = random.Random(a.seed)
    sample = rng.sample(pool, n)
    log.info("Sampled %d COCO negatives (seed=%d) -> %s", n, a.seed, out_dir)

    def place(src: Path, dst: Path) -> None:
        if a.copy:
            import shutil
            shutil.copyfile(src, dst)
        else:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as mf:
        for i, src in enumerate(sample):
            dst = out_dir / f"coco_{i:04d}_{src.name}"
            place(src, dst)
            mf.write(json.dumps({
                "index": i,
                "image": str(dst),
                "prompt": "",
                "source": "coco-val2017",
                "original": src.name,
            }) + "\n")
            mf.flush()

    log.info("Wrote %s with %d entries (%s).", manifest, n,
             "copied" if a.copy else "symlinked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())