"""Qwen2.5-VL-3B-Instruct labeling probe: per-digit YOLO bboxes on generated images.

Florence-2's custom code is incompatible with our transformers 5.12.1, so this probe uses
Qwen2.5-VL-3B-Instruct with bitsandbytes NF4 quantization. The language model is ~3B ->
NF4 ~1.5 GB; plus the vision tower it still fits comfortably in 6 GB VRAM.

Prompting strategy: ask the model to return one line per digit as:
  <digit> <x1> <y1> <x2> <y2>
where coordinates are normalized 0..1. The script parses that, falls back to `<box>...`
tokens / JSON if the model emits those instead, then writes YOLO `.txt` labels and overlay
visualizations.

Usage:
  python experiments/try_qwen25vl.py \
      --images experiments/out/flux2 \
      --out experiments/out/flux2_labels_qwen25 \
      --limit 1    # quick smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("try_qwen25vl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-VL-3B per-digit YOLO-label probe")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--images", default="experiments/out/flux2")
    p.add_argument("--out", default="experiments/out/flux2_labels_qwen25")
    p.add_argument("--limit", type=int, default=0, help="0 = all; else first N")
    p.add_argument("--quant", choices=["fp16", "int8", "nf4"], default="fp16",
                   help="Weight format: fp16 (CPU offload, most accurate), int8, nf4 (fastest)")
    p.add_argument("--gpu-mem", default="5GiB",
                   help="Max GPU RAM for device_map='auto' offloading (e.g. 5GiB)")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return p.parse_args()


def find_font(size: int = 16):
    for name in ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial.ttf", "FreeSansBold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    x1, x2 = clamp01(min(x1, x2)), clamp01(max(x1, x2))
    y1, y2 = clamp01(min(y1, y2)), clamp01(max(y1, y2))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(0.001, x2 - x1)
    h = max(0.001, y2 - y1)
    return cx, cy, w, h


def parse_line_text(text: str) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Parse lines like '7 0.12 0.34 0.56 0.78' -> (cls, (cx,cy,w,h)).

    If the label is a multi-digit integer (e.g. '99'), split it into individual
    digit boxes tiled horizontally across the detected region (heuristic fallback).
    """
    boxes = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*`]+", "", line).strip()
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) < 5:
            continue
        try:
            label = int(float(nums[0]))
        except ValueError:
            continue
        coords = [float(x) for x in nums[1:5]]
        scale = 1000.0 if max(coords) > 1.0 else 1.0
        x1, y1, x2, y2 = [c / scale for c in coords]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        digits = [int(d) for d in str(label) if d.isdigit()]
        if not digits:
            continue
        # Heuristic: split the full bbox horizontally into equal slices per digit.
        sub_w = (x2 - x1) / len(digits)
        for i, d in enumerate(digits):
            sx1 = x1 + i * sub_w
            sx2 = sx1 + sub_w
            boxes.append((d, xyxy_to_yolo(sx1, y1, sx2, y2)))
    return boxes


def parse_grounding(text: str) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Parse Qwen native grounding: digit(s) + <box>[[x1,y1],[x2,y2]]</box>.

    Also accepts the flat-array variant the model sometimes emits:
    <box>[[x1,y1,x2,y2]]</box>. Splits multi-digit labels across a single box
    heuristically.
    """
    boxes = []
    # Match <box>...</box> capturing the inner array string.
    box_matches = list(re.finditer(r"<box>\s*(\[\[.*?\]\])\s*</box>", text))
    for bm in box_matches:
        try:
            inner = bm.group(1).replace("'", '"')
            arr = json.loads(inner)
            if not arr or not isinstance(arr[0], list):
                continue
            # Two supported shapes:
            #   nested: [[x1,y1],[x2,y2]]   (list of two 2-element lists)
            #   flat:   [[x1,y1,x2,y2]]     (list containing one 4-element list)
            if len(arr) == 2 and len(arr[0]) == 2 and len(arr[1]) == 2:
                [[x1, y1], [x2, y2]] = arr
            elif len(arr) == 1 and len(arr[0]) == 4:
                [x1, y1, x2, y2] = arr[0]
            else:
                continue

            scale = 1000.0 if max(abs(x1), abs(x2), abs(y1), abs(y2)) > 1.0 else 1.0
            x1, x2 = x1 / scale, x2 / scale
            y1, y2 = y1 / scale, y2 / scale
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Extract label digits from immediately before or after the box token,
            # avoiding digits that live inside the coordinate array.
            before = text[max(0, bm.start() - 20):bm.start()]
            after = text[bm.end():bm.end() + 20]
            label_str = None
            # prefer a digit token directly before: e.g. "7 <box>..."
            m_before = re.search(r"(\d+)\s*$", before)
            if m_before:
                label_str = m_before.group(1)
            else:
                # or after: e.g. "<box>...<box> 7"
                m_after = re.search(r"^\s*(\d+)", after)
                if m_after:
                    label_str = m_after.group(1)
            digits = [int(d) for d in label_str] if label_str else [-1]

            sub_w = (x2 - x1) / len(digits)
            for i, d in enumerate(digits):
                sx1 = x1 + i * sub_w
                sx2 = sx1 + sub_w
                boxes.append((d, xyxy_to_yolo(sx1, y1, sx2, y2)))
        except Exception:
            pass
    return boxes


def parse_json(text: str) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Best-effort JSON extraction from model output."""
    # grab first JSON array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    boxes = []
    for item in data:
        if not isinstance(item, dict):
            continue
        digit = item.get("digit") or item.get("class") or item.get("label")
        bbox = item.get("bbox") or item.get("box") or item.get("bounding_box")
        if digit is None or bbox is None:
            continue
        try:
            cls = int(digit)
            if len(bbox) == 4:
                scale = 1000.0 if max(abs(float(x)) for x in bbox) > 1.0 else 1.0
                x1, y1, x2, y2 = [float(x) / scale for x in bbox]
                boxes.append((cls, xyxy_to_yolo(x1, y1, x2, y2)))
        except Exception:
            continue
    return boxes


def parse_detections(raw: str) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Try native grounding boxes, then strict line format, then JSON."""
    boxes = parse_grounding(raw)
    if boxes:
        return boxes
    boxes = parse_line_text(raw)
    if boxes:
        return boxes
    return parse_json(raw)


def draw_overlay(img: Image.Image, boxes: list[tuple[int, tuple[float, float, float, float]]]):
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = find_font(16)
    w, h = out.size
    for cls, (cx, cy, bw, bh) in boxes:
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = str(cls) if cls >= 0 else "?"
        if font:
            draw.text((x1 + 2, y1 + 2), label, fill="yellow", font=font)
        else:
            draw.text((x1 + 2, y1 + 2), label, fill="yellow")
    return out


def build_messages(prompt_text: str, image_path: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def main() -> int:
    a = parse_args()
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    max_memory = {0: a.gpu_mem, "cpu": "20GiB"}
    extra = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "max_memory": max_memory,
        "low_cpu_mem_usage": True,
    }
    if a.quant == "nf4":
        extra["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif a.quant == "int8":
        extra["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    # fp16 needs no quantization_config

    log.info("Loading %s (quant=%s, gpu_mem=%s)...", a.model, a.quant, a.gpu_mem)
    processor = Qwen2_5_VLProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, trust_remote_code=True, **extra
    ).eval()

    images = sorted(Path(a.images).glob("*.png"))
    if a.limit > 0:
        images = images[:a.limit]
    log.info("Labeling %d images...", len(images))

    prompt = (
        "You are a strict digit detector for house-number images. "
        "Detect every visible Arabic digit (0-9) individually. "
        "Each digit must have its own bounding box; never merge multiple digits into one box. "
        "For each digit output the digit followed by a bounding box in the exact format:\n"
        "<digit> <box>[[x1,y1],[x2,y2]]</box>\n"
        "Coordinates are normalized to 0..1. Example for the number 42:\n"
        "4 <box>[[0.30,0.50],[0.45,0.70]]</box>\n"
        "2 <box>[[0.46,0.50],[0.61,0.70]]</box>\n"
        "Only output these entries, nothing else."
    )

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_boxes: list[list[tuple[int, tuple]]] = []

    for p in images:
        try:
            messages = build_messages(prompt, str(p))
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[Image.open(p).convert("RGB")],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            # keep only newly generated tokens
            gen_ids = out[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

            boxes = parse_detections(raw)
            all_boxes.append(boxes)

            label_path = out_dir / f"{p.stem}.txt"
            label_path.write_text(
                "\n".join(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, (cx, cy, w, h) in boxes)
                + "\n"
            )
            (out_dir / f"{p.stem}.raw.txt").write_text(raw)
            log.info("RAW %s: %r", p.name, raw[:400])

            overlay = draw_overlay(Image.open(p).convert("RGB"), boxes)
            overlay_path = out_dir / f"{p.stem}_overlay.png"
            overlay.save(overlay_path)
            log.info("%s -> raw len=%d parsed=%d boxes; saved %s", p.name, len(raw), len(boxes), overlay_path.name)
            log.debug("raw: %s", raw[:500])
        except Exception as e:
            log.error("Failed on %s: %s", p.name, e, exc_info=True)

    # grid
    overlays = sorted(out_dir.glob("*_overlay.png"))
    if overlays:
        cell = 512
        rows = (len(overlays) + 1) // 2
        grid = Image.new("RGB", (cell * 2, cell * rows), (255, 255, 255))
        for i, ovp in enumerate(overlays):
            im = Image.open(ovp).convert("RGB").resize((cell, cell))
            grid.paste(im, ((i % 2) * cell, (i // 2) * cell))
        grid_path = out_dir / "grid.jpg"
        grid.save(grid_path, quality=85)
        log.info("Saved review grid: %s", grid_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())