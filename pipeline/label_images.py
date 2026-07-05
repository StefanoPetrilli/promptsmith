#!/usr/bin/env python3
"""Step 3 — label images with SAM 3 (text prompt "wrench")."""
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
CLASS_ID = 0
CLASS_NAME = "wrench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 3 text-grounded wrench labeling (step 3).")
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--prompt", default=CLASS_NAME)
    p.add_argument("--confidence", type=float, default=0.55)
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def collect_images(arg: str, limit: int) -> list[Path]:
    src = Path(arg)
    if src.is_file() and src.suffix == ".jsonl":
        imgs = []
        seen = set()
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            pth = Path(json.loads(line)["image"])
            if pth not in seen:
                seen.add(pth)
                imgs.append(pth)
        return imgs[:limit] if limit > 0 else imgs
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        return imgs[:limit] if limit > 0 else imgs
    raise FileNotFoundError(arg)


def load_predictor(model_path: str, resolution: int, device: str, confidence: float):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    log.info("Loading SAM 3 from %s...", model_path)
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=model_path,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    proc = Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=confidence)
    log.info("SAM 3 ready. VRAM allocated: %.0f MiB", torch.cuda.memory_allocated() / 1e6)
    return proc


def save_instance_masks(masks: torch.Tensor, out_dir: Path, stem: str) -> Path | None:
    if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    m = masks.squeeze(1).to(torch.bool).cpu().numpy()
    inst = np.zeros(m.shape[1:], dtype=np.uint16)
    for i in range(m.shape[0]):
        free = inst == 0
        inst[free & m[i]] = i + 1
    dst = out_dir / f"{stem}.png"
    Image.fromarray(inst).save(dst)
    return dst


def boxes_to_yolo(boxes_xyxy: torch.Tensor, w: int, h: int) -> list[str]:
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy.tolist():
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def label_image(proc, img_path: Path, prompt: str, save_masks: bool, masks_dir: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    state = proc.set_image(img)
    state = proc.set_text_prompt(prompt, state=state)

    boxes = state["boxes"].detach().cpu()
    scores = state["scores"].detach().cpu().tolist()
    masks = state["masks"]

    yolo_lines = boxes_to_yolo(boxes, w, h)
    label_path = masks_dir.parent / (img_path.stem + ".txt")
    label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    mask_path = save_instance_masks(masks, masks_dir, img_path.stem) if save_masks else None
    return {
        "image": str(img_path),
        "label": str(label_path),
        "boxes_xyxy": boxes.tolist(),
        "scores": scores,
        "n": len(yolo_lines),
        "prompt": prompt,
        "mask": str(mask_path) if mask_path else None,
    }


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(a.images, a.limit)
    if not images:
        log.error("No images found in %s", a.images)
        return 2

    proc = load_predictor(a.model_path, a.resolution, a.device, a.confidence)
    log.info("Labeling %d images with prompt=%r conf>=%.2f res=%d",
             len(images), a.prompt, a.confidence, a.resolution)

    masks_dir = out_dir / "masks"
    audit = out_dir / "labels.jsonl"
    n_ok, n_empty = 0, 0

    with audit.open("w", encoding="utf-8") as af:
        for img_path in tqdm(images, desc="label", unit="img"):
            try:
                record = label_image(proc, img_path, a.prompt, a.save_masks, masks_dir)
            except Exception as exc:
                log.error("  failed: %s", exc)
                af.write(json.dumps({"image": str(img_path), "error": str(exc)}) + "\n")
                af.flush()
                continue

            af.write(json.dumps(record, ensure_ascii=False) + "\n")
            af.flush()
            if record["n"]:
                n_ok += 1
            else:
                n_empty += 1
            tqdm.write(f"  {img_path.name}: {record['n']} wrench boxes (scores={record['scores']})")
            torch.cuda.empty_cache()

    log.info("Done. %d/%d images with >=1 box, %d empty. Audit: %s", n_ok, len(images), n_empty, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
