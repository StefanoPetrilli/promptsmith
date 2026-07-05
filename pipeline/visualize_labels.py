#!/usr/bin/env python3
"""Step 4 — overlay SAM 3 segmentation + boxes on labelled images."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.utils import palette

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("visualize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay SAM 3 segmentation + boxes (step 4).")
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--alpha", type=int, default=110)
    p.add_argument("--box-width", type=int, default=3)
    p.add_argument("--boxes-from", choices=["labels", "masks"], default="labels")
    p.add_argument("--min-confidence", type=float, default=0.55)
    p.add_argument("--masks-subdir", default="masks")
    return p.parse_args()


def load_entries(path: Path, limit: int) -> list[dict]:
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


def mask_boxes(inst: np.ndarray) -> list[tuple[int, int, int, int] | None]:
    boxes = []
    for i in range(1, int(inst.max()) + 1):
        ys, xs = np.where(inst == i)
        if xs.size == 0:
            boxes.append(None)
        else:
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes


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
        label = f"{float(score):.2f}"
        tw, th = text_size(draw, label, font)
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=c + (255,))
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0, 255), font=font)


def select_boxes(boxes_xyxy: list, scores: list, inst: np.ndarray | None, boxes_from: str):
    if boxes_from == "masks" and inst is not None:
        mboxes = mask_boxes(inst)
        boxes = [b for b in mboxes if b is not None]
        scores = [scores[i] for i, b in enumerate(mboxes) if b is not None]
        return boxes, scores
    return [tuple(b) for b in boxes_xyxy], scores


def render(image_path: str, boxes_xyxy: list, scores: list, inst: np.ndarray | None,
           alpha: int, box_width: int, boxes_from: str, min_conf: float) -> Image.Image:
    base = Image.open(image_path).convert("RGBA")
    boxes, scores, inst = filter_by_confidence(boxes_xyxy, scores, inst, min_conf)

    if inst is not None:
        base = Image.alpha_composite(base, build_overlay(inst, alpha))

    boxes, scores = select_boxes(boxes, scores, inst, boxes_from)
    draw = ImageDraw.Draw(base)
    draw_boxes(draw, boxes, scores, box_width)
    return base.convert("RGB")


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(Path(a.labels), a.limit)
    if not entries:
        log.error("No entries in %s", a.labels)
        return 2

    log.info("Rendering %d images (boxes-from=%s, min-confidence=%.2f)...",
             len(entries), a.boxes_from, a.min_confidence)

    index = out_dir / "visuals.jsonl"
    n_ok = 0
    with index.open("w", encoding="utf-8") as ix:
        for i, e in enumerate(entries):
            img_path = e["image"]
            stem = Path(img_path).stem
            log.info("[%d/%d] %s", i + 1, len(entries), Path(img_path).name)
            try:
                inst = load_instance_mask(e, a.masks_subdir)
                n_in = len(e.get("boxes_xyxy", []))
                img = render(img_path, e.get("boxes_xyxy", []), e.get("scores", []),
                             inst, a.alpha, a.box_width, a.boxes_from, a.min_confidence)
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
                "boxes_from": a.boxes_from,
            }) + "\n")
            ix.flush()
            n_ok += 1

    log.info("Done. %d/%d visuals written to %s", n_ok, len(entries), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
