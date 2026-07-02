#!/usr/bin/env python3
"""Step 3 — label images with SAM 3 (text prompt "wrench").

Writes one YOLO `.txt` per image (`0 xc yc bw bh`, normalized; single class 0 = wrench), a
`labels.jsonl` audit with raw boxes + scores, and 16-bit instance-mask PNGs
(`masks/<stem>.png`, pixel = instance id, 0 = bg) for step 5.

Weights: `facebook/sam3` is gated, so the open mirror `1038lab/sam3` (`sam3.pt`) is used by
default (`--model-path`). `--limit N` labels only the first N images.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("label")

DEFAULT_MODEL_PATH = "models/sam3/sam3.pt"
CLASS_ID = 0  # single class: wrench
CLASS_NAME = "wrench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 3 text-grounded wrench labeling (step 3).")
    p.add_argument("--images", required=True, help="dir of images, or a manifest.jsonl from step 2")
    p.add_argument("--out", required=True, help="output dir for YOLO .txt labels + labels.jsonl")
    p.add_argument("--limit", type=int, default=0, help="0 = all; else first N (test)")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                   help="path to sam3.pt checkpoint (default: open mirror cache)")
    p.add_argument("--prompt", default=CLASS_NAME, help="SAM 3 text prompt (default: 'wrench')")
    p.add_argument("--confidence", type=float, default=0.55,
                   help="SAM 3 confidence threshold (boxes below this are dropped)")
    p.add_argument("--resolution", type=int, default=1008,
                   help="SAM 3 input resolution (lower to save VRAM, e.g. 768)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=True,
                   help="persist instance masks to <out>/masks/<stem>.png for step 5")
    return p.parse_args()


def collect_images(arg: str, limit: int) -> list[Path]:
    src = Path(arg)
    if src.is_file() and src.suffix == ".jsonl":
        imgs = []
        seen = set()
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            pth = Path(d["image"])
            if pth in seen:
                continue  # skip duplicate manifest entries
            seen.add(pth)
            imgs.append(pth)
        return imgs[:limit] if limit > 0 else imgs
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        return imgs[:limit] if limit > 0 else imgs
    raise FileNotFoundError(arg)


def load_predictor(model_path: str, resolution: int, device: str, confidence: float):
    """Build the SAM 3 image model + processor (loaded once, reused for every image)."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    log.info("Loading SAM 3 from %s (device=%s)...", model_path, device)
    model = build_sam3_image_model(
        device=device, eval_mode=True,
        checkpoint_path=model_path, load_from_HF=False,
        enable_segmentation=True, enable_inst_interactivity=False,
    )
    proc = Sam3Processor(model, resolution=resolution, device=device,
                        confidence_threshold=confidence)
    log.info("SAM 3 ready. VRAM allocated: %.0f MiB", torch.cuda.memory_allocated() / 1e6)
    return proc


def save_instance_masks(masks: torch.Tensor, out_dir: Path, stem: str):
    """Write a 16-bit instance-id PNG (0=bg, i+1=instance i). Returns path or None.

    `masks` is the SAM 3 processor output: a [N,1,H,W] bool tensor in original image
    coords. Pixels are assigned first-wins (an instance keeps only the pixels not already
    claimed by an earlier instance), which is fine for overlay/visualization.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{stem}.png"
    if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
        return None
    m = masks.squeeze(1).to(torch.bool).cpu().numpy()  # [N,H,W]
    inst = np.zeros(m.shape[1:], dtype=np.uint16)
    for i in range(m.shape[0]):
        free = inst == 0
        inst[free & m[i]] = i + 1
    Image.fromarray(inst).save(dst)
    return dst


def boxes_to_yolo(boxes_xyxy: torch.Tensor, w: int, h: int) -> list[str]:
    """Convert pixel xyxy boxes to normalized YOLO `class xc yc bw bh` lines."""
    out = []
    for b in boxes_xyxy.tolist():
        x1, y1, x2, y2 = b
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        out.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return out


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(a.images, a.limit)
    if not images:
        log.error("no images found in %s", a.images)
        return 2

    proc = load_predictor(a.model_path, a.resolution, a.device, a.confidence)
    log.info("Labeling %d images with SAM 3 prompt=%r (conf>=%.2f, res=%d)",
             len(images), a.prompt, a.confidence, a.resolution)

    audit = out_dir / "labels.jsonl"
    n_ok, n_empty = 0, 0
    with audit.open("w", encoding="utf-8") as af:
        for i, img_path in enumerate(tqdm(images, desc="label", unit="img")):
            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                st = proc.set_image(img)
                st = proc.set_text_prompt(a.prompt, state=st)
                boxes = st["boxes"].detach().cpu()       # xyxy, pixel coords
                scores = st["scores"].detach().cpu().tolist()
                masks = st["masks"]                       # [N,1,H,W] bool (orig res)
            except Exception as e:  # noqa: BLE001
                log.error("  failed: %s", e)
                af.write(json.dumps({"image": str(img_path), "error": str(e)}) + "\n")
                af.flush()
                continue

            yolo_lines = boxes_to_yolo(boxes, w, h)
            label_path = out_dir / (img_path.stem + ".txt")
            label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                                  encoding="utf-8")
            mask_path = (save_instance_masks(masks, out_dir / "masks", img_path.stem)
                         if a.save_masks else None)
            af.write(json.dumps({
                "image": str(img_path), "label": str(label_path),
                "boxes_xyxy": boxes.tolist(), "scores": scores,
                "n": len(yolo_lines), "prompt": a.prompt,
                "mask": str(mask_path) if mask_path else None,
            }, ensure_ascii=False) + "\n")
            af.flush()
            if yolo_lines:
                n_ok += 1
            else:
                n_empty += 1
            tqdm.write(f"  {img_path.name}: {len(yolo_lines)} wrench boxes "
                       f"(scores={[round(s, 2) for s in scores]})")
            torch.cuda.empty_cache()

    log.info("Done. %d/%d images with >=1 box, %d empty. Audit: %s",
             n_ok, len(images), n_empty, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
