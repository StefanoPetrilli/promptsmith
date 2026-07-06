#!/usr/bin/env python3
"""Download a single Open Images class and convert its bounding boxes to YOLO format."""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("openimages")

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
    p.add_argument("--class-name", default="Wrench")
    p.add_argument("--split", default="validation", choices=SPLITS)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default="data/openimages/cache")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-group-of", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-depiction", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-occluded", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-area", type=float, default=0.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=int, default=30)
    return p.parse_args()


def download_file(url: str, dest: Path, timeout: int = 30) -> None:
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
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f) if len(row) >= 2}


def find_label_name(class_map: dict[str, str], display_name: str) -> str:
    for label_name, name in class_map.items():
        if name.lower() == display_name.lower():
            return label_name
    raise ValueError(f"Class '{display_name}' not found in Open Images class descriptions")


def load_image_urls(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["ImageID"]: row["OriginalURL"] for row in csv.DictReader(f)}


def keep_row(row: dict, skip_group_of: bool, skip_depiction: bool, skip_occluded: bool) -> bool:
    if row.get("IsGroupOf", "0") == "1" and skip_group_of:
        return False
    if row.get("IsDepiction", "0") == "1" and skip_depiction:
        return False
    if row.get("IsOccluded", "0") == "1" and skip_occluded:
        return False
    return True


def load_bboxes(
    path: Path,
    label_name: str,
    skip_group_of: bool,
    skip_depiction: bool,
    skip_occluded: bool,
    min_area: float,
) -> dict[str, list[tuple[float, float, float, float]]]:
    boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] != label_name:
                continue
            if not keep_row(row, skip_group_of, skip_depiction, skip_occluded):
                continue
            xmin, xmax = float(row["XMin"]), float(row["XMax"])
            ymin, ymax = float(row["YMin"]), float(row["YMax"])
            if (xmax - xmin) * (ymax - ymin) < min_area:
                continue
            boxes[row["ImageID"]].append((xmin, xmax, ymin, ymax))
    return boxes


def to_yolo_line(xmin: float, xmax: float, ymin: float, ymax: float) -> str:
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    return f"0 {xc:.6f} {yc:.6f} {xmax - xmin:.6f} {ymax - ymin:.6f}"


def download_image(url: str, dest: Path, retries: int, timeout: int) -> bool:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with dest.open("wb") as f:
                    shutil.copyfileobj(response, f)
            return True
        except Exception:
            log.debug("Download attempt %d/%d failed for %s", attempt, retries, url)
    return False


def fetch_csvs(args) -> tuple[Path, Path, Path]:
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    class_desc = cache / "class-descriptions-boxable.csv"
    download_file(URLS["class-descriptions"], class_desc, args.timeout)

    bbox_path = cache / f"{args.split}-annotations-bbox.csv"
    download_file(URLS[f"{args.split}-bbox"], bbox_path, args.timeout)

    image_url_path = cache / f"{args.split}-images-with-rotation.csv"
    download_file(URLS[f"{args.split}-images"], image_url_path, args.timeout)

    return class_desc, bbox_path, image_url_path


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    class_desc, bbox_path, image_url_path = fetch_csvs(args)

    class_map = load_class_map(class_desc)
    label_name = find_label_name(class_map, args.class_name)
    log.info("Open Images class '%s' -> LabelName %s", args.class_name, label_name)

    bboxes = load_bboxes(
        bbox_path, label_name,
        args.skip_group_of, args.skip_depiction, args.skip_occluded, args.min_area,
    )
    urls = load_image_urls(image_url_path)
    log.info("Found %d images with at least one '%s' box", len(bboxes), args.class_name)

    image_ids = list(bboxes.keys())
    if args.limit:
        image_ids = image_ids[:args.limit]
        log.info("Limiting download to %d images", len(image_ids))

    n_ok, n_fail = 0, 0
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

        lines = [to_yolo_line(*box) for box in bboxes[image_id]]
        (lbl_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_ok += 1

    log.info("Done. Downloaded %d images, failed %d. Labels in %s", n_ok, n_fail, lbl_dir)
    log.info("Use with: --val-images %s --val-labels %s", img_dir, lbl_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
