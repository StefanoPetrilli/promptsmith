#!/usr/bin/env python3
"""Visualize YOLO labels for a folder of real images (e.g. Open Images validation).

Reads images from --images and matching YOLO .txt labels from --labels, draws the boxes,
and writes overlaid copies to --out. Useful for sanity-checking manually corrected boxes.

Usage:
  python pipeline/visualize_openimages.py \
      --images data/openimages_val/images \
      --labels data/openimages_val/labels \
      --out data/openimages_val/visuals
"""
from __future__ import annotations

import argparse
import colorsys
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("visualize_openimages")

IMG_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay YOLO boxes on real images.")
    p.add_argument("--images", required=True, help="directory of images")
    p.add_argument("--labels", required=True, help="directory of YOLO .txt labels")
    p.add_argument("--out", required=True, help="output directory for overlaid images")
    p.add_argument("--limit", type=int, default=0, help="0 = all; else first N")
    p.add_argument("--box-width", type=int, default=3)
    p.add_argument("--alpha", type=int, default=110, help="unused, kept for consistency")
    return p.parse_args()


def palette(n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(max(n, 1)):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def load_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Convert YOLO normalized boxes to pixel xyxy."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            log.warning("skipping malformed line in %s: %s", label_path, line)
            continue
        try:
            _, xc, yc, bw, bh = map(float, parts)
        except ValueError:
            log.warning("skipping non-numeric line in %s: %s", label_path, line)
            continue
        pw, ph = bw * w, bh * h
        px, py = xc * w, yc * h
        x1 = int(max(0, px - pw / 2))
        y1 = int(max(0, py - ph / 2))
        x2 = int(min(w, px + pw / 2))
        y2 = int(min(h, py + ph / 2))
        boxes.append((x1, y1, x2, y2))
    return boxes


def render(image_path: Path, boxes: list, box_width: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    cols = palette(max(len(boxes), 1))
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = cols[i % len(cols)]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=box_width)
        label = f"{i + 1}"
        try:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        except Exception:  # noqa: BLE001
            tw, th = 14, 11
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=c)
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0), font=font)
    return img


def main() -> int:
    a = parse_args()
    img_dir = Path(a.images)
    lbl_dir = Path(a.labels)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    if a.limit > 0:
        images = images[:a.limit]
    if not images:
        log.error("no images found in %s", img_dir)
        return 2

    log.info("Rendering %d images from %s -> %s", len(images), img_dir, out_dir)
    index = out_dir / "visuals.jsonl"
    n_ok = 0
    with index.open("w", encoding="utf-8") as ix:
        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes = load_boxes(lbl_path, *Image.open(img_path).size)
            log.info("%s: %d box(es)", img_path.name, len(boxes))
            try:
                rendered = render(img_path, boxes, a.box_width)
            except Exception as e:  # noqa: BLE001
                log.error("failed to render %s: %s", img_path.name, e)
                continue
            dst = out_dir / (img_path.stem + ".png")
            rendered.save(dst)
            ix.write(json.dumps({
                "image": str(img_path),
                "label": str(lbl_path),
                "visual": str(dst),
                "n_boxes": len(boxes),
            }) + "\n")
            ix.flush()
            n_ok += 1

    log.info("Done. %d/%d visuals written to %s (index: %s)", n_ok, len(images), out_dir, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
