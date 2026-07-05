#!/usr/bin/env python3
"""Run YOLO inference on a folder of images and save visualizations + predictions.

Usage:
  python pipeline/inference.py \
      --model data/dataset/runs/train/weights/best.pt \
      --source data/dataset/images/train \
      --out data/inference_train

Outputs:
  <out>/images/   — input copies with predicted boxes drawn
  <out>/labels/   — YOLO-format .txt predictions (one per image)
  <out>/predictions.jsonl — per-image box list with scores
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("inference")

IMG_SUFFIXES = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLO inference on images.")
    p.add_argument("--model", required=True, help="path to .pt weights")
    p.add_argument("--source", required=True, help="directory of images")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference size")
    p.add_argument("--limit", type=int, default=0, help="0 = all; else first N")
    p.add_argument("--device", default="0", help="cuda device or 'cpu'")
    return p.parse_args()


def collect_images(source: Path, limit: int) -> list[Path]:
    imgs = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
    if limit > 0:
        imgs = imgs[:limit]
    return imgs


def yolo_to_txt(boxes: list) -> str:
    """boxes: list of (cls, xc, yc, w, h, conf) normalized."""
    lines = []
    for b in boxes:
        cls, xc, yc, w, h, conf = b
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    a = parse_args()
    source = Path(a.source)
    out = Path(a.out)
    out_images = out / "images"
    out_labels = out / "labels"
    for d in (out, out_images, out_labels):
        d.mkdir(parents=True, exist_ok=True)

    imgs = collect_images(source, a.limit)
    if not imgs:
        log.error("no images found in %s", source)
        return 2

    from ultralytics import YOLO
    log.info("Loading model %s", a.model)
    model = YOLO(a.model)

    log.info("Running inference on %d images (conf=%.2f, imgsz=%d)", len(imgs), a.conf, a.imgsz)
    predictions = []
    for img_path in imgs:
        results = model.predict(
            source=str(img_path),
            conf=a.conf,
            imgsz=a.imgsz,
            device=a.device,
            verbose=False,
        )[0]

        boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                xywhn = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                boxes.append((cls, *xywhn, conf))

        # Save visualization
        rendered = results.plot(line_width=2, font_size=0.5)
        pil_img = Image.fromarray(rendered[:, :, ::-1])  # BGR -> RGB
        dst_img = out_images / (img_path.stem + ".png")
        pil_img.save(dst_img)

        # Save prediction txt
        dst_lbl = out_labels / (img_path.stem + ".txt")
        dst_lbl.write_text(yolo_to_txt(boxes), encoding="utf-8")

        predictions.append({
            "image": str(img_path),
            "visual": str(dst_img),
            "label": str(dst_lbl),
            "n_boxes": len(boxes),
            "boxes": [
                {"class": int(cls), "xc": xc, "yc": yc, "w": w, "h": h, "conf": conf}
                for cls, xc, yc, w, h, conf in boxes
            ],
        })
        log.info("%s: %d box(es)", img_path.name, len(boxes))

    index = out / "predictions.jsonl"
    with index.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    log.info("Done. %d images inferred -> %s", len(imgs), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
