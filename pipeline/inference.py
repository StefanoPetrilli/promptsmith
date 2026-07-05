#!/usr/bin/env python3
"""Run YOLO inference on a folder of images and save visualizations + predictions."""
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
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="0")
    return p.parse_args()


def collect_images(source: Path, limit: int) -> list[Path]:
    imgs = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
    return imgs[:limit] if limit > 0 else imgs


def boxes_to_txt(boxes: list) -> str:
    lines = [
        f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}"
        for cls, xc, yc, w, h, conf in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def run_inference(model, img_path: Path, conf: float, imgsz: int, device: str) -> tuple[list, Image.Image]:
    results = model.predict(
        source=str(img_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    boxes = []
    if results.boxes is not None:
        for box in results.boxes:
            xywhn = box.xywhn[0].tolist()
            boxes.append((int(box.cls[0]), *xywhn, float(box.conf[0])))

    rendered = results.plot(line_width=2, font_size=0.5)
    return boxes, Image.fromarray(rendered[:, :, ::-1])


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
        log.error("No images found in %s", source)
        return 2

    from ultralytics import YOLO
    log.info("Loading model %s", a.model)
    model = YOLO(a.model)

    log.info("Running inference on %d images (conf=%.2f, imgsz=%d)", len(imgs), a.conf, a.imgsz)
    predictions = []
    for img_path in imgs:
        boxes, visual = run_inference(model, img_path, a.conf, a.imgsz, a.device)

        dst_img = out_images / (img_path.stem + ".png")
        visual.save(dst_img)

        dst_lbl = out_labels / (img_path.stem + ".txt")
        dst_lbl.write_text(boxes_to_txt(boxes), encoding="utf-8")

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
