#!/usr/bin/env python3
"""Assemble the real-image verification (hold-out) dataset into a flat, versioned layout.

Stages images from verification_dataset/{positive, hard_negatives, negatives} into
verification_dataset/assembled/{images,labels} (flat YOLO layout):
  - positive images      -> copy + their YOLO label from --positive-labels
  - hard_negatives       -> copy + empty label (background, false-positive test)
  - negatives            -> copy + empty label (background, false-positive test)

Also writes:
  - assembled/manifest.json : per-file sha256, dims, box count, split, counts, version
  - assembled/VERSION       : short version string (content-addressed)

The assembled dir is self-contained (copies, not symlinks) so it can be committed
to git as a versioned dataset artifact and consumed directly by train_yolo.py:
    --val-images verification_dataset/assembled/images \
    --val-labels verification_dataset/assembled/labels
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".jfif", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble + version the verification dataset.")
    p.add_argument("--root", default="verification_dataset")
    p.add_argument("--positive", default="positive")
    p.add_argument("--hard-negatives", default="hard_negatives")
    p.add_argument("--negatives", default="negatives")
    p.add_argument("--positive-labels", default="positive_labels")
    p.add_argument("--out", default="verification_dataset/assembled")
    p.add_argument("--version", default=None, help="override version string (default: v1 + date)")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(dir_: Path) -> list[Path]:
    return sorted(p for p in dir_.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def stage_split(
    split: str,
    imgs: list[Path],
    pos_labels: Path,
    out_img: Path,
    out_lbl: Path,
    manifest_files: list[dict],
) -> int:
    n = 0
    for img in imgs:
        dst_img = out_img / f"{split}_{img.name}"
        shutil.copyfile(img, dst_img)
        w, h = Image.open(img).size

        # label: positives read from positive_labels (flat, by stem); negatives -> empty
        src_lbl = pos_labels / (img.stem + ".txt")
        boxes = 0
        if src_lbl.exists():
            text = src_lbl.read_text(encoding="utf-8")
            dst_lbl = out_lbl / (dst_img.stem + ".txt")
            dst_lbl.write_text(text, encoding="utf-8")
            boxes = sum(1 for ln in text.splitlines() if ln.strip())
        else:
            (out_lbl / (dst_img.stem + ".txt")).write_text("", encoding="utf-8")

        manifest_files.append({
            "split": split,
            "image": dst_img.name,
            "source": str(img),
            "label": dst_img.stem + ".txt",
            "boxes": boxes,
            "width": w,
            "height": h,
            "sha256": sha256(dst_img),
        })
        n += 1
    return n


def main() -> int:
    a = parse_args()
    root = Path(a.root).resolve()
    pos = collect(root / a.positive)
    hneg = collect(root / a.hard_negatives)
    neg = collect(root / a.negatives)
    if not pos:
        raise SystemExit(f"no positive images in {root / a.positive}")

    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out_img = out / "images"
    out_lbl = out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    # manifest + VERSION live at the source root so they can be committed to pin a
    # snapshot even though the assembled images/labels are regenerable (gitignored).

    pos_labels = root / a.positive_labels
    files: list[dict] = []
    n_pos = stage_split("positive", pos, pos_labels, out_img, out_lbl, files)
    n_hneg = stage_split("hard_negative", hneg, pos_labels, out_img, out_lbl, files)
    n_neg = stage_split("negative", neg, pos_labels, out_img, out_lbl, files)

    total_boxes = sum(f["boxes"] for f in files)
    # content-addressed version: short hash of the sorted image+label set
    digest = hashlib.sha256()
    for f in sorted(files, key=lambda x: x["image"]):
        digest.update(f["image"].encode())
        digest.update(f["sha256"].encode())
        digest.update(str(f["boxes"]).encode())
    content_hash = digest.hexdigest()[:12]
    version = a.version or f"v1-{dt.datetime.utcnow().strftime('%Y%m%d')}-{content_hash}"

    manifest = {
        "name": "numeri verification dataset",
        "task": "single-class object detection (wrench)",
        "version": version,
        "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "classes": ["wrench"],
        "splits": {
            "positive": n_pos,
            "hard_negative": n_hneg,
            "negative": n_neg,
        },
        "totals": {
            "images": n_pos + n_hneg + n_neg,
            "labeled_boxes": total_boxes,
            "negatives": n_hneg + n_neg,
        },
        "layout": {"images": "images/", "labels": "labels/", "format": "YOLO (0 xc yc w h, normalized)"},
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "VERSION").write_text(version + "\n", encoding="utf-8")

    print(f"assembled {n_pos + n_hneg + n_neg} images "
          f"(positive={n_pos}, hard_negative={n_hneg}, negative={n_neg}), "
          f"{total_boxes} boxes -> {out}")
    print(f"version: {version}")
    print("consume with:")
    print(f"  --val-images {out_img} --val-labels {out_lbl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())