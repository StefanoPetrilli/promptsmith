#!/usr/bin/env python3
"""Step 5 — render copies of the images with the segmentation overlaid + boxes drawn.

This is a pure post-processing step: it consumes the outputs of steps 2 & 3 (images, the
`labels.jsonl` audit log with `boxes_xyxy` + `scores`, and the per-image instance-mask PNGs
written by step 3 under `masks/`) and produces, per image, a copy with:

  * the SAM 3 instance segmentation **overlaid** as semi-transparent colored regions;
  * the **bounding boxes** drawn on top, with the confidence score annotated.

No model is loaded — nothing leaves the CPU — so this is cheap to run on the whole set.

Which boxes are drawn? By default the same boxes that were written to the YOLO `.txt`
labels (i.e. SAM 3's `pred_boxes`, the actual training targets), so the visualization shows
exactly what the detector will learn. Pass `--boxes-from masks` to instead draw the tight
axis-aligned bbox computed from each segmentation mask's pixels (a literal
"box-based-on-the-segmentation" view), which is useful for spotting over/under-segmentation.

Outputs: `<out>/img_XXXX.png` (RGB) for every image that has a labels.jsonl entry, plus a
`visuals.jsonl` index.

Test mode: `--limit 3` renders just the first 3 images.

Usage:
  python pipeline/visualize_labels.py --labels data/labels/labels.jsonl --out data/visuals
  python pipeline/visualize_labels.py --labels data/labels_test/labels.jsonl --out data/visuals_test --limit 3
"""
from __future__ import annotations

import argparse
import colorsys
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("visualize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay SAM 3 segmentation + boxes (step 5).")
    p.add_argument("--labels", required=True,
                   help="labels.jsonl produced by step 3 (one entry per image)")
    p.add_argument("--out", required=True, help="output dir for overlaid PNG copies")
    p.add_argument("--limit", type=int, default=0, help="0 = all; else first N (test)")
    p.add_argument("--alpha", type=int, default=110, help="mask overlay alpha 0..255")
    p.add_argument("--box-width", type=int, default=3)
    p.add_argument("--boxes-from", choices=["labels", "masks"], default="labels",
                   help="'labels' = SAM 3 pred_boxes (the YOLO targets); "
                        "'masks' = tight bbox computed from each segmentation mask")
    p.add_argument("--min-confidence", type=float, default=0.55,
                   help="discard boxes (and their mask instances) with score below this")
    p.add_argument("--masks-subdir", default="masks",
                   help="subdir (relative to each mask path's parent, or --labels dir) "
                        "holding the instance PNGs written by step 3")
    return p.parse_args()


def palette(n: int) -> list[tuple[int, int, int]]:
    """Distinct RGB colors spread around the HSV wheel."""
    out = []
    for i in range(max(n, 1)):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def load_instance_mask(entry: dict, masks_subdir: str) -> np.ndarray | None:
    """Load the 16-bit instance-id PNG for an image. Returns None if absent/empty."""
    # step 3 records the absolute mask path; fall back to <labels dir>/masks/<stem>.png
    mask_path = entry.get("mask")
    if not mask_path:
        labels_dir = Path(entry["label"]).parent
        mask_path = labels_dir / masks_subdir / (Path(entry["image"]).stem + ".png")
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return None
    inst = np.array(Image.open(mask_path))  # uint16, 0=bg
    if inst.ndim != 2 or inst.max() == 0:
        return None
    return inst


def mask_boxes(inst: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Tight xyxy pixel boxes, one per instance id (1..inst.max()), skipping empty ones."""
    boxes = []
    for i in range(1, int(inst.max()) + 1):
        ys, xs = np.where(inst == i)
        if xs.size == 0:
            boxes.append(None)  # keep index aligned with instance id
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes


def render(image_path: str, boxes_xyxy: list, scores: list, inst: np.ndarray | None,
           alpha: int, box_width: int, boxes_from: str, min_conf: float) -> Image.Image:
    base = Image.open(image_path).convert("RGBA")
    W, H = base.size

    # --- drop low-confidence boxes + their mask instances (keeps overlay & boxes in sync) ---
    keep_idx = [i for i, s in enumerate(scores) if float(s) >= min_conf]
    boxes_xyxy = [boxes_xyxy[i] for i in keep_idx]
    scores = [scores[i] for i in keep_idx]
    if inst is not None and inst.max() > 0:
        keep_ids = {i + 1 for i in keep_idx}
        relabel = np.zeros_like(inst)
        for new_i, old_id in enumerate(sorted(keep_ids), start=1):
            relabel[inst == old_id] = new_i
        inst = relabel

    # --- segmentation overlay (only if we have instance masks) ---
    if inst is not None:
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
        overlay = Image.fromarray(np.dstack([color_layer, alpha_layer]), "RGBA")
        base = Image.alpha_composite(base, overlay)
    else:
        n_inst = 0

    # --- choose which boxes to draw ---
    if boxes_from == "masks" and inst is not None:
        mboxes = mask_boxes(inst)
        boxes = [b for b in mboxes if b is not None]
        # align scores by filtering None entries
        scores_local = []
        mi = 0
        for b in mboxes:
            if b is None:
                continue
            scores_local.append(scores[mi] if mi < len(scores) else 0.0)
            mi += 1
        scores = scores_local
    else:
        boxes = [tuple(b) for b in boxes_xyxy]

    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    cols = palette(max(len(boxes), 1))
    for i, (b, s) in enumerate(zip(boxes, scores)):
        c = cols[i % len(cols)]
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=c + (255,), width=box_width)
        label = f"{float(s):.2f}"
        ty = max(0, y1 - 12)
        # text background for legibility
        try:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        except Exception:  # noqa: BLE001
            tw, th = 28, 11
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=c + (255,))
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0, 255), font=font)

    return base.convert("RGB")


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for line in Path(a.labels).read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    if a.limit > 0:
        entries = entries[:a.limit]
    if not entries:
        log.error("no entries in %s", a.labels)
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
            except Exception as ex:  # noqa: BLE001
                log.error("  failed: %s", ex)
                continue
            dst = out_dir / f"{stem}.png"
            img.save(dst)
            kept = len([s for s in e.get("scores", []) if float(s) >= a.min_confidence])
            ix.write(json.dumps({"image": img_path, "visual": str(dst),
                                 "n_boxes_in": n_in, "n_boxes_kept": kept,
                                 "min_confidence": a.min_confidence,
                                 "boxes_from": a.boxes_from}) + "\n")
            ix.flush()
            n_ok += 1

    log.info("Done. %d/%d visuals written to %s (index: %s)",
             n_ok, len(entries), out_dir, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
