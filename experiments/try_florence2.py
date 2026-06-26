"""Florence-2 labeling probe: draw YOLO bboxes around text/digits in generated images.

Uses `microsoft/Florence-2-base-ft` (tiny, ~230M / 0.5 GB fp16) — purpose-built for
image-text grounding and OCR-with-regions. Fits trivially in 6 GB VRAM.

Pipeline:
  1. For each input image, run Florence-2 with `<OCR_WITH_REGION>` to get text
     regions (text string + bbox) and with `<OD>` to get generic object boxes.
  2. Normalize bboxes to YOLO `cls cx cy w h` (0..1, per-line).
  3. Write YOLO label files next to the image.
  4. Draw overlays for visual inspection.

Granularity caveat: Florence-2 returns *text-region* boxes, not necessarily per-character.
For a house-number like "42" it may produce one box for the whole string. This experiment
reveals whether that's acceptable or if we need a per-digit model (e.g. Qwen2.5-VL) next.

Usage:
  python experiments/try_florence2.py \
      --images experiments/out/flux2 \
      --out experiments/out/flux2_labels \
      --task ocr
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("try_florence2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Florence-2 YOLO-label probe")
    p.add_argument("--model", default="microsoft/Florence-2-base-ft")
    p.add_argument("--images", default="experiments/out/flux2")
    p.add_argument("--out", default="experiments/out/flux2_labels")
    p.add_argument("--task", choices=["ocr", "od"], default="ocr",
                   help="ocr = text-with-region boxes; od = generic object detection")
    p.add_argument("--cls", type=int, default=0,
                   help="YOLO class index to assign to every detected box (fallback)")
    p.add_argument("--conf", type=float, default=0.0,
                   help="minimum text-length confidence filter (ignored by OD)")
    p.add_argument("--max", type=int, default=10)
    return p.parse_args()


def find_font(size: int = 16):
    """Return a PIL font or None if system has no suitable fonts."""
    for name in ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial.ttf", "FreeSansBold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def florence_bbox_to_yolo(bbox: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    """Florence-2 bboxes are normalized 0..1000 ints; convert to YOLO xywh 0..1.

    Accepts either [x1,y1,x2,y2] (OCR) or quadrilateral [x1,y1,x2,y2,x3,y3,x4,y4]
    (also OCR_WITH_REGION) — we take the enclosing axis-aligned rectangle.
    """
    xs = [v for i, v in enumerate(bbox) if i % 2 == 0]
    ys = [v for i, v in enumerate(bbox) if i % 2 == 1]
    x1, x2 = min(xs) / 1000.0, max(xs) / 1000.0
    y1, y2 = min(ys) / 1000.0, max(ys) / 1000.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(0.001, x2 - x1)
    bh = max(0.001, y2 - y1)
    return cx, cy, bw, bh


def yolo_line(cls: int, cx: float, cy: float, w: float, h: float) -> str:
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def extract_digits(text: str) -> list[int]:
    """Return all digit characters as int class ids."""
    return [int(ch) for ch in text if ch.isdigit()]


def draw_overlay(img: Image.Image, boxes: list[tuple[int, tuple[float, float, float, float], str]]):
    """Draw normalized boxes + text on a copy of img. boxes = [(cls, (cx,cy,w,h), label)]"""
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = find_font(16)
    w, h = out.size
    for cls, (cx, cy, bw, bh), label in boxes:
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{cls}:{label}" if label else str(cls)
        if font:
            draw.text((x1 + 2, y1 + 2), text, fill="yellow", font=font)
        else:
            draw.text((x1 + 2, y1 + 2), text, fill="yellow")
    return out


def run_florence(model, processor, image: Image.Image, task: str):
    """Run a single Florence-2 task and return parsed dict."""
    task_token = f"<{task.upper()}>"
    inputs = processor(text=task_token, images=image, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        early_stopping=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task_token, image_size=(image.width, image.height)
    )
    return parsed


def process_image(image_path: Path, model, processor, args) -> tuple[Path, Path, list] | None:
    """Run Florence-2, write YOLO label + overlay."""
    img = Image.open(image_path).convert("RGB")
    task_token = "<OCR_WITH_REGION>" if args.task == "ocr" else "<OD>"
    parsed = run_florence(model, processor, img, args.task)
    result = parsed.get(task_token, {})

    boxes: list[tuple[int, tuple[float, float, float, float], str]] = []
    if args.task == "ocr":
        for label, bbox in zip(result.get("labels", []), result.get("bboxes", [])):
            text = str(label)
            if len(text) < 1 or (args.conf and not any(ch.isdigit() for ch in text)):
                continue
            digits = extract_digits(text)
            cx, cy, bw, bh = florence_bbox_to_yolo(bbox, img.width, img.height)
            if not digits:
                # no digit recognized -> single box with fallback class
                boxes.append((args.cls, (cx, cy, bw, bh), text))
            else:
                # naive per-string split: assign each digit a sub-box of equal width.
                # This is a heuristic; real per-digit would need a per-character model.
                sub_w = bw / len(digits)
                for i, d in enumerate(digits):
                    sub_cx = cx - bw / 2 + sub_w * (i + 0.5)
                    boxes.append((d, (sub_cx, cy, sub_w * 0.95, bh), str(d)))
    else:
        # OD: generic object boxes; just use fallback class
        for bbox, label in zip(result.get("bboxes", []), result.get("labels", [])):
            cx, cy, bw, bh = florence_bbox_to_yolo(bbox, img.width, img.height)
            boxes.append((args.cls, (cx, cy, bw, bh), str(label)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = image_path.stem
    label_path = out_dir / f"{base}.txt"
    label_path.write_text("\n".join(yolo_line(c, *b) for c, b, _ in boxes) + "\n")

    overlay = draw_overlay(img, boxes)
    overlay_path = out_dir / f"{base}_overlay.png"
    overlay.save(overlay_path)

    return label_path, overlay_path, boxes


def main() -> int:
    a = parse_args()
    from transformers import AutoModelForCausalLM, AutoProcessor

    log.info("Loading Florence-2 (%s) on cuda (fp16)...", a.model)
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()

    images = sorted(Path(a.images).glob("*.png"))[:a.max]
    log.info("Found %d images to label", len(images))

    for p in images:
        try:
            lp, op, boxes = process_image(p, model, processor, a)
            log.info("%s -> %d boxes, saved %s", p.name, len(boxes), op.name)
        except Exception as e:
            log.error("Failed on %s: %s", p.name, e, exc_info=True)

    # Build a grid of overlays for quick review
    overlays = sorted(Path(a.out).glob("*_overlay.png"))[:a.max]
    if overlays:
        cell = 512
        rows = (len(overlays) + 1) // 2
        grid = Image.new("RGB", (cell * 2, cell * rows), (255, 255, 255))
        for i, ovp in enumerate(overlays):
            im = Image.open(ovp).convert("RGB").resize((cell, cell))
            grid.paste(im, ((i % 2) * cell, (i // 2) * cell))
        grid_path = Path(a.out) / "grid.jpg"
        grid.save(grid_path, quality=85)
        log.info("Saved review grid: %s", grid_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())