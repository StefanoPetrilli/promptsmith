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
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)),
                   help="parallel render processes (default: CPU count)")
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
    imgs = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)
    if limit <= 0:
        return imgs
    # per-leaf (immediate parent) limit, mirroring run_leaves' --limit semantics;
    # flat directories fall through to a single group.
    groups: dict[str, list[Path]] = {}
    for p in imgs:
        try:
            rel = p.relative_to(images_dir)
        except ValueError:
            rel = Path(p.name)
        key = rel.parts[0] if len(rel.parts) > 1 else ""
        groups.setdefault(key, []).append(p)
    out: list[Path] = []
    for k in sorted(groups):
        out.extend(groups[k][:limit])
    return out


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


# --- worker process entry points (picklable; take primitive args only) --------


def _render_sam_one(entry: dict, alpha: int, box_width: int, min_conf: float,
                    masks_subdir: str, out_dir: str) -> dict:
    """Render one SAM-labelled image in a worker process."""
    img_path = entry["image"]
    stem = Path(img_path).stem
    try:
        inst = load_instance_mask(entry, masks_subdir)
        n_in = len(entry.get("boxes_xyxy", []))
        img = render_sam(img_path, entry.get("boxes_xyxy", []), entry.get("scores", []),
                        inst, alpha, box_width, min_conf)
        dst = Path(out_dir) / f"{stem}.png"
        img.save(dst)
        kept = len([s for s in entry.get("scores", []) if float(s) >= min_conf])
        return {"image": img_path, "visual": str(dst), "n_boxes_in": n_in,
                "n_boxes_kept": kept, "min_confidence": min_conf,
                "ok": True, "error": None, "name": Path(img_path).name}
    except Exception as exc:
        return {"image": img_path, "ok": False, "error": str(exc),
                "name": Path(img_path).name}


def _render_yolo_one(img_path: str, img_dir: str, lbl_dir: str,
                     out_dir: str, box_width: int) -> dict:
    """Render one YOLO-labelled image in a worker process."""
    ip = Path(img_path); idir = Path(img_dir); ldir = Path(lbl_dir); odir = Path(out_dir)
    try:
        try:
            rel = ip.relative_to(idir)
        except ValueError:
            rel = Path(ip.name)
        lbl_path = ldir / rel.with_suffix(".txt")
        boxes = load_yolo_boxes(lbl_path, *Image.open(ip).size)
        rendered = render_yolo(ip, lbl_path, box_width)
        dst = odir / rel.with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(dst)
        return {"image": str(ip), "label": str(lbl_path), "visual": str(dst),
                "n_boxes": len(boxes), "ok": True, "error": None, "name": ip.name}
    except Exception as exc:
        return {"image": str(ip), "ok": False, "error": str(exc), "name": ip.name}


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
            log.info("Rendering %d SAM-labelled images (min-confidence=%.2f) with %d workers...",
                     len(entries), a.min_confidence, a.workers)
            tasks = [(e, a.alpha, a.box_width, a.min_confidence,
                      a.masks_subdir, str(out_dir)) for e in entries]
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(_render_sam_one, *t) for t in tasks]
                done = 0
                for fut in as_completed(futs):
                    res = fut.result()
                    done += 1
                    if res["ok"]:
                        log.info("[%d/%d] %s", done, len(futs), res["name"])
                        ix.write(json.dumps({
                            "image": res["image"],
                            "visual": res["visual"],
                            "n_boxes_in": res["n_boxes_in"],
                            "n_boxes_kept": res["n_boxes_kept"],
                            "min_confidence": res["min_confidence"],
                        }) + "\n")
                        ix.flush()
                        n_ok += 1
                    else:
                        log.error("[%d/%d] failed %s: %s",
                                  done, len(futs), res["name"], res["error"])

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
            log.info("Rendering %d YOLO-labelled images -> %s with %d workers...",
                     len(images), out_dir, a.workers)
            tasks = [(str(p), str(img_dir), str(lbl_dir), str(out_dir), a.box_width)
                     for p in images]
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(_render_yolo_one, *t) for t in tasks]
                done = 0
                for fut in as_completed(futs):
                    res = fut.result()
                    done += 1
                    if res["ok"]:
                        log.info("[%d/%d] %s: %d box(es)",
                                 done, len(futs), res["name"], res["n_boxes"])
                        ix.write(json.dumps({
                            "image": res["image"],
                            "label": res["label"],
                            "visual": res["visual"],
                            "n_boxes": res["n_boxes"],
                        }) + "\n")
                        ix.flush()
                        n_ok += 1
                    else:
                        log.error("[%d/%d] failed %s: %s",
                                  done, len(futs), res["name"], res["error"])

    log.info("Done. %d visuals written to %s (index: %s)", n_ok, out_dir, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
