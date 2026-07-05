#!/usr/bin/env python3
"""Visualize YOLO labels for a folder of real images (e.g. Open Images validation)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline.utils import palette

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("visualize_openimages")

IMG_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay YOLO boxes on real images.")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--box-width", type=int, default=3)
    return p.parse_args()


def load_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
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


def text_size(draw: ImageDraw.ImageDraw, label: str, font: ImageFont.FreeTypeFont | None):
    try:
        return draw.textbbox((0, 0), label, font=font)[2:]
    except Exception:
        return 14, 11


def draw_boxes(draw: ImageDraw.ImageDraw, boxes: list, box_width: int):
    font = ImageFont.load_default()
    cols = palette(max(len(boxes), 1))
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = cols[i % len(cols)]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=box_width)
        label = f"{i + 1}"
        tw, th = text_size(draw, label, font)
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=c)
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0), font=font)


def render(image_path: Path, boxes: list, box_width: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw_boxes(draw, boxes, box_width)
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
        log.error("No images found in %s", img_dir)
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
            except Exception as exc:
                log.error("failed to render %s: %s", img_path.name, exc)
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

    log.info("Done. %d/%d visuals written to %s", n_ok, len(images), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
