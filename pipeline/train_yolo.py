#!/usr/bin/env python3
"""Step 5 — assemble a single-class (wrench) YOLO dataset and fine-tune.

Pairs images with labels by stem, splits train/val (default 80/20), symlinks both into
`<out>/{images,labels}/{train,val}`, writes `data.yaml`, and runs ultralytics training.
`--allow-dummy-labels` writes a placeholder centered box for unlabeled images (smoke test).

Trains on the post-processed tree by default: `--images-pp` (default: data/images_pp/ if it
exists, else falls back to `--images`). The image tree is walked recursively (it is a
`<category>[/<count>]/` tree from steps 2/3.5); labels live in a mirrored `<category>[/<count>]/`
subtree under `--labels` (step 3 is invoked once per leaf). Dataset filenames are flattened from
the relative path (e.g. `clean_positive/1/img_0000.png` -> `clean_positive_1_img_0000.png`) so
cross-category stems never collide, while label lookup stays stem-aligned within each leaf.

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

CLASSES = ["wrench"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble YOLO dataset + fine-tune (step 5).")
    p.add_argument("--images", required=True, help="dir of clean images (step 2 output, fallback)")
    p.add_argument("--images-pp", default=None,
                   help="dir of post-processed images (step 3.5). Default: data/images_pp/ if it "
                        "exists, else fall back to --images.")
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
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _resolve_images(images: Path, images_pp: Path | None) -> Path:
    """Pick the post-processed tree when present, else fall back to the clean tree."""
    if images_pp is not None:
        return images_pp
    pp = Path("data/images_pp")
    return pp if pp.exists() else images


def dummy_label(path: Path) -> None:
    """Placeholder: one centered box for class 0 (wrench). Smoke-test only — not real data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")


def _flat_name(img: Path, images_dir: Path) -> str:
    """Relative path under the image tree -> unique dataset filename (slashes -> underscores)."""
    rel = img.relative_to(images_dir)
    return rel.as_posix().replace("/", "_")


def assemble_dataset(images_dir: Path, labels_dir: Path, out: Path,
                     val_frac: float, seed: int, allow_dummy: bool) -> Path:
    imgs = sorted(p for p in images_dir.rglob("*")
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
        flat = _flat_name(img, images_dir)              # e.g. clean_positive_1_img_0000.png
        stem = Path(flat).stem                            # e.g. clean_positive_1_img_0000
        # Label mirrors the image tree: <labels>/<rel-parent>/<stem>.txt
        lbl = labels_dir / img.relative_to(images_dir).with_suffix(".txt")
        if not lbl.exists() or lbl.stat().st_size == 0:
            if not allow_dummy:
                log.warning("no label for %s — skipping (use --allow-dummy-labels for smoke test)", flat)
                continue
            dummy_label(out / "labels" / split / (stem + ".txt"))
        else:
            dst_lbl = out / "labels" / split / (stem + ".txt")
            if dst_lbl.exists() or dst_lbl.is_symlink():
                dst_lbl.unlink()
            dst_lbl.symlink_to(lbl.resolve())
        _symlink(img, out / "images" / split / flat)
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
    images_dir = _resolve_images(Path(a.images), Path(a.images_pp) if a.images_pp else None)
    log.info("Training on image tree: %s", images_dir)
    data_yaml = assemble_dataset(images_dir, Path(a.labels), out,
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
