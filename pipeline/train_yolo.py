#!/usr/bin/env python3
"""Step 5 — assemble a single-class (wrench) YOLO dataset and fine-tune.

Pairs images with labels by stem, splits train/val (default 80/20), symlinks both into
`<out>/{images,labels}/{train,val}`, writes `data.yaml`, and runs ultralytics training.
`--allow-dummy-labels` writes a placeholder centered box for unlabeled images (smoke test).

Usage:
  python pipeline/train_yolo.py --images data/images --labels data/labels \
      --out data/dataset --base models/yolov8n.pt --epochs 50 --imgsz 640 --batch 16
  python pipeline/train_yolo.py --images data/images_test --labels data/labels_test \
      --out data/dataset_test --base models/yolov8n.pt --epochs 1 --imgsz 320 --batch 1 \
      --allow-dummy-labels
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("train")

CLASSES = ["wrench"]  # single-class detector (SAM 3 labels only wrenches)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble YOLO dataset + fine-tune (step 5).")
    p.add_argument("--images", required=True, help="dir of images (step 2 output)")
    p.add_argument("--labels", required=True, help="dir of YOLO .txt labels (step 3 output)")
    p.add_argument("--out", required=True, help="dataset dir to assemble (data.yaml written here)")
    p.add_argument("--base", default="yolov8n.pt",
                   help="base YOLO weights (models/yolov8n.pt or models/yolo26n.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-dummy-labels", action="store_true",
                   help="write a placeholder box for unlabeled images (smoke test only)")
    p.add_argument("--dry-run", action="store_true",
                   help="assemble the dataset and write data.yaml, then exit (no training)")
    return p.parse_args()


def _symlink(src: Path, dst: Path) -> None:
    dst = dst.with_suffix(src.suffix)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def dummy_label(path: Path) -> None:
    """Placeholder: one centered box for class 0 (wrench). Smoke-test only — not real data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")


def assemble_dataset(images_dir: Path, labels_dir: Path, out: Path,
                     val_frac: float, seed: int, allow_dummy: bool) -> Path:
    imgs = sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not imgs:
        raise FileNotFoundError(f"no images in {images_dir}")
    rng = random.Random(seed)
    n_val = max(1, int(len(imgs) * val_frac)) if len(imgs) > 1 else 0
    val = set(rng.sample(range(len(imgs)), n_val)) if n_val else set()

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    n_labeled = 0
    for i, img in enumerate(imgs):
        split = "val" if i in val else "train"
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists() or lbl.stat().st_size == 0:
            if not allow_dummy:
                log.warning("no label for %s — skipping (use --allow-dummy-labels for smoke test)", img.name)
                continue
            dummy_label(out / "labels" / split / (img.stem + ".txt"))
        else:
            dst_lbl = out / "labels" / split / (img.stem + ".txt")
            if dst_lbl.exists() or dst_lbl.is_symlink():
                dst_lbl.unlink()
            dst_lbl.symlink_to(lbl.resolve())
        _symlink(img, out / "images" / split / img.name)
        n_labeled += 1

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(CLASSES)),
        encoding="utf-8",
    )
    log.info("Assembled dataset at %s: %d images (%d val). data.yaml -> %s",
             out, n_labeled, n_val, data_yaml)
    return data_yaml


def main() -> int:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data_yaml = assemble_dataset(Path(a.images), Path(a.labels), out,
                                 a.val_frac, a.seed, a.allow_dummy_labels)

    if a.dry_run:
        log.info("--dry-run: dataset assembled, not training. data.yaml=%s", data_yaml)
        return 0

    from ultralytics import YOLO
    log.info("Loading base weights %s and training %d epochs (imgsz=%s, batch=%s, device=%s)",
             a.base, a.epochs, a.imgsz, a.batch, a.device)
    model = YOLO(a.base)
    project_dir = (out / "runs").resolve()
    model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, device=a.device, project=str(project_dir),
                name="train", exist_ok=True, verbose=True)
    log.info("Done. Weights under %s/runs/train/", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
