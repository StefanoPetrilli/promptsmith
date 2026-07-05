#!/usr/bin/env python3
"""Step 4 — overlay boxes (and SAM masks, when available) on images.

Supports two label formats:
  sam  — a labels.jsonl file from step 3, with boxes, scores and instance masks.
  yolo — a directory of YOLO .txt files paired with a directory of images.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.utils import palette

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("visualize")

IMG_SUFFIXES = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay boxes/masks on images (step 4).")
    p.add_argument("--labels", required=True,
                   help="labels.jsonl for --format sam, or labels directory for --format yolo")
    p.add_argument("--images", default=None,
                   help="image directory (required for --format yolo)")
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["sam", "yolo"], default="sam")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--alpha", type=int, default=110)
    p.add_argument("--box-width", type=int, default=3)
    p.add_argument("--min-confidence", type=float, default=0.55,
                   help="only applies to --format sam")
    p.add_argument("--masks-subdir", default="masks",
                   help="only applies to --format sam")
    return p.parse_args()


# --- SAM format helpers ------------------------------------------------------


def load_sam_entries(path: Path, limit: int) -> list[dict]:
    seen = set()
    entries = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        img = d.get("image")
        if img not in seen:
            seen.add(img)
            entries.append(d)
    return entries[:limit] if limit > 0 else entries


def load_instance_mask(entry: dict, masks_subdir: str) -> np.ndarray | None:
    mask_path = entry.get("mask") or Path(entry["label"]).parent / masks_subdir / (Path(entry["image"]).stem + ".png")
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return None
    inst = np.array(Image.open(mask_path))
    if inst.ndim != 2 or inst.max() == 0:
        return None
    return inst


def filter_by_confidence(boxes_xyxy: list, scores: list, inst: np.ndarray | None, min_conf: float):
    keep = [i for i, s in enumerate(scores) if float(s) >= min_conf]
    boxes = [boxes_xyxy[i] for i in keep]
    scores = [scores[i] for i in keep]
    if inst is not None and inst.max() > 0:
        keep_ids = {i + 1 for i in keep}
        relabel = np.zeros_like(inst)
        for new_i, old_id in enumerate(sorted(keep_ids), start=1):
            relabel[inst == old_id] = new_i
        inst = relabel
    return boxes, scores, inst


# --- YOLO format helpers -----------------------------------------------------


def collect_images(images_dir: Path, limit: int) -> list[Path]:
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    return imgs[:limit] if limit > 0 else imgs


def load_yolo_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
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


# --- drawing helpers ---------------------------------------------------------


def build_overlay(inst: np.ndarray, alpha: int) -> Image.Image:
    H, W = inst.shape
    n_inst = int(inst.max())
    cols = palette(n_inst)
    color_layer = np.zeros((H, W, 3), dtype=np.uint8)
    alpha_layer = np.zeros((H, W), dtype=np.uint8)
    for i in range(n_inst):
        m = inst == (i + 1)
        if not m.any():
            continue
        color_layer[m] = cols[i]
        alpha_layer[m] = alpha
    return Image.fromarray(np.dstack([color_layer, alpha_layer]), "RGBA")


def text_size(draw: ImageDraw.ImageDraw, label: str, font: ImageFont.FreeTypeFont | None):
    try:
        return draw.textbbox((0, 0), label, font=font)[2:]
    except Exception:
        return 28, 11


def draw_boxes(draw: ImageDraw.ImageDraw, boxes: list, scores: list, box_width: int):
    font = ImageFont.load_default()
    cols = palette(max(len(boxes), 1))
    for i, (box, score) in enumerate(zip(boxes, scores)):
        c = cols[i % len(cols)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=c + (255,), width=box_width)
        label = f"{float(score):.2f}" if score is not None else f"{i + 1}"
        tw, th = text_size(draw, label, font)
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=c + (255,))
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0, 255), font=font)


# --- rendering ---------------------------------------------------------------


def render_sam(image_path: str, boxes_xyxy: list, scores: list, inst: np.ndarray | None,
               alpha: int, box_width: int, min_conf: float) -> Image.Image:
    base = Image.open(image_path).convert("RGBA")
    boxes, scores, inst = filter_by_confidence(boxes_xyxy, scores, inst, min_conf)

    if inst is not None:
        base = Image.alpha_composite(base, build_overlay(inst, alpha))

    draw = ImageDraw.Draw(base)
    draw_boxes(draw, [tuple(b) for b in boxes], scores, box_width)
    return base.convert("RGB")


def render_yolo(image_path: Path, label_path: Path, box_width: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    boxes = load_yolo_boxes(label_path, *img.size)
    draw = ImageDraw.Draw(img)
    draw_boxes(draw, boxes, [None] * len(boxes), box_width)
    return img


# --- main --------------------------------------------------------------------


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = out_dir / "visuals.jsonl"
    n_ok = 0

    with index.open("w", encoding="utf-8") as ix:
        if a.format == "sam":
            entries = load_sam_entries(Path(a.labels), a.limit)
            if not entries:
                log.error("No entries in %s", a.labels)
                return 2
            log.info("Rendering %d SAM-labelled images (min-confidence=%.2f)...",
                     len(entries), a.min_confidence)
            for i, e in enumerate(entries):
                img_path = e["image"]
                stem = Path(img_path).stem
                log.info("[%d/%d] %s", i + 1, len(entries), Path(img_path).name)
                try:
                    inst = load_instance_mask(e, a.masks_subdir)
                    n_in = len(e.get("boxes_xyxy", []))
                    img = render_sam(
                        img_path, e.get("boxes_xyxy", []), e.get("scores", []),
                        inst, a.alpha, a.box_width, a.min_confidence,
                    )
                except Exception as exc:
                    log.error("  failed: %s", exc)
                    continue
                dst = out_dir / f"{stem}.png"
                img.save(dst)
                kept = len([s for s in e.get("scores", []) if float(s) >= a.min_confidence])
                ix.write(json.dumps({
                    "image": img_path,
                    "visual": str(dst),
                    "n_boxes_in": n_in,
                    "n_boxes_kept": kept,
                    "min_confidence": a.min_confidence,
                }) + "\n")
                ix.flush()
                n_ok += 1

        else:
            if not a.images:
                log.error("--images is required for --format yolo")
                return 2
            img_dir = Path(a.images)
            lbl_dir = Path(a.labels)
            images = collect_images(img_dir, a.limit)
            if not images:
                log.error("No images found in %s", img_dir)
                return 2
            log.info("Rendering %d YOLO-labelled images -> %s", len(images), out_dir)
            for img_path in images:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                boxes = load_yolo_boxes(lbl_path, *Image.open(img_path).size)
                log.info("%s: %d box(es)", img_path.name, len(boxes))
                try:
                    rendered = render_yolo(img_path, lbl_path, a.box_width)
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

    log.info("Done. %d visuals written to %s (index: %s)", n_ok, out_dir, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
