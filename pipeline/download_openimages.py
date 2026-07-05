#!/usr/bin/env python3
"""Download a single Open Images class and convert its bounding boxes to YOLO format.

Open Images v6 bbox coordinates are already normalized to [0, 1], so converting to YOLO is a
simple coordinate transform. By default `group-of` and `depiction` boxes are skipped so the
validation set contains only real, individual wrenches.

Note: Open Images has only ~1 wrench image in the official validation/test splits but ~200 in
the train split. For a usable real-world validation set, use `--split train`.

Usage:
  python pipeline/download_openimages.py --class-name Wrench --split train \
      --out data/openimages_val --limit 500

Outputs:
  <out>/images/   — downloaded .jpg images
  <out>/labels/   — YOLO .txt labels (class 0 for the single requested class)
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import urllib.request
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("openimages")

# Open Images v6 / 2018_04 public URLs.
URLS = {
    "class-descriptions": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
    "train-bbox": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation-bbox": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test-bbox": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "train-images": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation-images": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test-images": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}

SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download one Open Images class as YOLO validation data.")
    p.add_argument("--class-name", default="Wrench",
                   help="DisplayName of the Open Images class to download")
    p.add_argument("--split", default="validation", choices=SPLITS,
                   help="Open Images split to download")
    p.add_argument("--out", required=True, help="output directory for images/ and labels/")
    p.add_argument("--cache", default="data/.openimages_cache",
                   help="directory to cache downloaded CSVs")
    p.add_argument("--limit", type=int, default=None,
                   help="max number of images to download (default: all that match)")
    p.add_argument("--skip-group-of", action=argparse.BooleanOptionalAction, default=True,
                   help="skip group-of boxes (default: --skip-group-of)")
    p.add_argument("--skip-depiction", action=argparse.BooleanOptionalAction, default=True,
                   help="skip depiction/illustration boxes (default: --skip-depiction)")
    p.add_argument("--skip-occluded", action=argparse.BooleanOptionalAction, default=False,
                   help="skip occluded boxes (default: --no-skip-occluded)")
    p.add_argument("--min-area", type=float, default=0.0,
                   help="minimum normalized box area to keep")
    p.add_argument("--retries", type=int, default=3,
                   help="retries per image download")
    p.add_argument("--timeout", type=int, default=30,
                   help="HTTP timeout in seconds")
    return p.parse_args()


def download_file(url: str, dest: Path, timeout: int = 30) -> None:
    """Download a file to dest if it does not already exist."""
    if dest.exists():
        log.info("Using cached %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with dest.open("wb") as f:
                shutil.copyfileobj(response, f)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def load_class_map(path: Path) -> dict[str, str]:
    """Map LabelName -> DisplayName from class-descriptions CSV."""
    mapping = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    return mapping


def find_label_name(class_map: dict[str, str], display_name: str) -> str:
    for label_name, name in class_map.items():
        if name.lower() == display_name.lower():
            return label_name
    raise ValueError(f"Class '{display_name}' not found in Open Images class descriptions")


def load_image_urls(path: Path) -> dict[str, str]:
    """Map ImageID -> OriginalURL."""
    urls = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls[row["ImageID"]] = row["OriginalURL"]
    return urls


def load_bboxes(
    path: Path,
    label_name: str,
    skip_group_of: bool,
    skip_depiction: bool,
    skip_occluded: bool,
    min_area: float,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Map ImageID -> list of (xmin, xmax, ymin, ymax) for the requested label."""
    boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["LabelName"] != label_name:
                continue
            # Columns differ slightly between train and val/test, but these names are common.
            if row.get("IsGroupOf", "0") == "1" and skip_group_of:
                continue
            if row.get("IsDepiction", "0") == "1" and skip_depiction:
                continue
            if row.get("IsOccluded", "0") == "1" and skip_occluded:
                continue
            xmin, xmax = float(row["XMin"]), float(row["XMax"])
            ymin, ymax = float(row["YMin"]), float(row["YMax"])
            area = (xmax - xmin) * (ymax - ymin)
            if area < min_area:
                continue
            boxes[row["ImageID"]].append((xmin, xmax, ymin, ymax))
    return boxes


def to_yolo_line(xmin: float, xmax: float, ymin: float, ymax: float) -> str:
    """Convert Open Images normalized box to YOLO line (class 0)."""
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def download_image(url: str, dest: Path, retries: int, timeout: int) -> bool:
    """Download one image with retries. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with dest.open("wb") as f:
                    shutil.copyfileobj(response, f)
            return True
        except Exception as exc:
            log.debug("Download attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
    return False


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # 1. Class descriptions.
    class_desc_path = cache / "class-descriptions-boxable.csv"
    download_file(URLS["class-descriptions"], class_desc_path, args.timeout)
    class_map = load_class_map(class_desc_path)
    label_name = find_label_name(class_map, args.class_name)
    log.info("Open Images class '%s' -> LabelName %s", args.class_name, label_name)

    # 2. Bounding boxes and image URLs for the chosen split.
    bbox_path = cache / f"{args.split}-annotations-bbox.csv"
    download_file(URLS[f"{args.split}-bbox"], bbox_path, args.timeout)
    image_url_path = cache / f"{args.split}-images-with-rotation.csv"
    download_file(URLS[f"{args.split}-images"], image_url_path, args.timeout)

    log.info("Loading bounding boxes for %s...", args.split)
    bboxes = load_bboxes(
        bbox_path, label_name,
        args.skip_group_of, args.skip_depiction, args.skip_occluded, args.min_area,
    )
    log.info("Found %d images with at least one '%s' box", len(bboxes), args.class_name)

    log.info("Loading image URLs...")
    urls = load_image_urls(image_url_path)

    image_ids = list(bboxes.keys())
    if args.limit:
        image_ids = image_ids[:args.limit]
        log.info("Limiting download to %d images", len(image_ids))

    n_ok = 0
    n_fail = 0
    for image_id in tqdm(image_ids, desc="Downloading images"):
        url = urls.get(image_id)
        if not url:
            log.warning("No URL for %s", image_id)
            n_fail += 1
            continue

        img_path = img_dir / f"{image_id}.jpg"
        if not download_image(url, img_path, args.retries, args.timeout):
            log.warning("Failed to download %s", image_id)
            n_fail += 1
            continue

        lbl_path = lbl_dir / f"{image_id}.txt"
        lines = [to_yolo_line(*box) for box in bboxes[image_id]]
        lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_ok += 1

    log.info("Done. Downloaded %d images, failed %d. Labels in %s", n_ok, n_fail, lbl_dir)
    log.info("You can now use this as a validation set with:")
    log.info("  --val-images %s --val-labels %s", img_dir, lbl_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
