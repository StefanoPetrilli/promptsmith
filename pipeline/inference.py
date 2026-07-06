#!/usr/bin/env python3
"""Run YOLO inference on a folder of images and save visualizations + predictions."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
    p.add_argument("--gt-labels", default=None,
                   help="YOLO .txt label directory to compare against (ground truth)")
    p.add_argument("--gt-color", default="0,255,0",
                   help="RGB for ground-truth boxes (default green)")
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


def parse_color(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3 or not all(0 <= c <= 255 for c in parts):
        raise ValueError(f"bad color {s!r}; use R,G,B")
    return tuple(parts)


def load_yolo_boxes(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Return list of (xc, yc, bw, bh) in normalized YOLO format."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            _cls, xc, yc, bw, bh = map(float, parts[:5])
        except ValueError:
            continue
        boxes.append((xc, yc, bw, bh))
    return boxes


def xywhn_to_xyxy(xc: float, yc: float, bw: float, bh: float, w: int, h: int) -> tuple[int, int, int, int]:
    pw, ph = bw * w, bh * h
    px, py = xc * w, yc * h
    return (int(max(0, px - pw / 2)), int(max(0, py - ph / 2)),
            int(min(w, px + pw / 2)), int(min(h, py + ph / 2)))


def _dashed_rectangle(draw, xy, color, width=1, dash=6, gap=4):
    x1, y1, x2, y2 = xy
    sides = [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for sx1, sy1, sx2, sy2 in sides:
        length = abs(sx2 - sx1) + abs(sy2 - sy1)
        if length == 0:
            continue
        dx = (sx2 - sx1) / length
        dy = (sy2 - sy1) / length
        d = 0.0
        while d < length:
            a = d
            b = min(d + dash, length)
            draw.line([(sx1 + dx * a, sy1 + dy * a), (sx1 + dx * b, sy1 + dy * b)],
                      fill=color, width=width)
            d += dash + gap


def overlay_gt(img: Image.Image, gt_boxes: list[tuple[float, float, float, float]],
                color: tuple[int, int, int]) -> Image.Image:
    """Draw ground-truth boxes as dashed outlines + small \"GT\" tags."""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    w, h = img.size
    for xc, yc, bw, bh in gt_boxes:
        x1, y1, x2, y2 = xywhn_to_xyxy(xc, yc, bw, bh, w, h)
        _dashed_rectangle(draw, [x1, y1, x2, y2], color + (255,), width=3, dash=8)
        draw.rectangle([x1, max(0, y1 - 12), x1 + 22, max(0, y1 - 1)], fill=color + (255,))
        draw.text((x1 + 2, max(0, y1 - 11)), "GT", fill=(0, 0, 0, 255), font=font)
    return Image.alpha_composite(img, overlay).convert("RGB")


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aarea = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    barea = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (aarea + barea - inter)


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

    gt_dir = Path(a.gt_labels) if a.gt_labels else None
    gt_color = parse_color(a.gt_color)

    log.info("Running inference on %d images (conf=%.2f, imgsz=%d)%s",
             len(imgs), a.conf, a.imgsz,
             f"; comparing to GT in {gt_dir}" if gt_dir else "")
    predictions = []
    total_tp = total_fp = total_fn = 0
    for img_path in imgs:
        boxes, visual = run_inference(model, img_path, a.conf, a.imgsz, a.device)

        gt_boxes_norm: list[tuple[float, float, float, float]] = []
        n_gt = 0
        if gt_dir is not None:
            gt_lbl = gt_dir / (img_path.stem + ".txt")
            w, h = visual.size
            gt_boxes_norm = load_yolo_boxes(gt_lbl, w, h)
            n_gt = len(gt_boxes_norm)
            visual = overlay_gt(visual, gt_boxes_norm, gt_color)

        dst_img = out_images / (img_path.stem + ".png")
        visual.save(dst_img)

        dst_lbl = out_labels / (img_path.stem + ".txt")
        dst_lbl.write_text(boxes_to_txt(boxes), encoding="utf-8")

        # greedy IoU matching for TP/FP/FN counts
        tp = fp = fn = 0
        if gt_dir is not None:
            matched = [False] * n_gt
            w, h = visual.size
            gt_xyxy = [xywhn_to_xyxy(*b, w, h) for b in gt_boxes_norm]
            pred_xyxy = [xywhn_to_xyxy(xc, yc, bw, bh, w, h)
                         for _cls, xc, yc, bw, bh, _conf in boxes]
            for pb in pred_xyxy:
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gt_xyxy):
                    if matched[j]:
                        continue
                    iou = _iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= 0.5 and best_j >= 0:
                    matched[best_j] = True
                    tp += 1
                else:
                    fp += 1
            fn = n_gt - sum(matched)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        predictions.append({
            "image": str(img_path),
            "visual": str(dst_img),
            "label": str(dst_lbl),
            "n_pred": len(boxes),
            "n_gt": n_gt,
            "tp": tp, "fp": fp, "fn": fn,
            "boxes": [
                {"class": int(cls), "xc": xc, "yc": yc, "w": w, "h": h, "conf": conf}
                for cls, xc, yc, w, h, conf in boxes
            ],
        })
        log.info("%s: %d pred / %d gt (tp=%d fp=%d fn=%d)",
                 img_path.name, len(boxes), n_gt, tp, fp, fn)

    index = out / "predictions.jsonl"
    with index.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    if gt_dir is not None:
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        summary = {
            "n_images": len(imgs),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "iou_threshold": 0.5,
        }
        (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("Comparison summary (IoU>=0.5): P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)",
                 prec, rec, f1, total_tp, total_fp, total_fn)

    log.info("Done. %d images inferred -> %s", len(imgs), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
