#!/usr/bin/env python3
"""Step 5 — assemble a single-class (wrench) YOLO dataset and fine-tune.

Pairs images with labels by stem, splits train/val (default 80/20), symlinks both into
`<out>/{images,labels}/{train,val}`, writes `data.yaml`, and runs ultralytics training.
Images without a label file (or with an empty label file) are kept as negative examples
by default (empty YOLO `.txt`). Use `--allow-dummy-labels` only for smoke tests; it writes
a placeholder centered box instead of an empty negative.

Trains on the post-processed tree by default: `--images-pp` (default: data/images_pp/ if it
exists, else falls back to `--images`). The image tree is walked recursively (it is a
`<category>/` tree from steps 2/3.5); labels live in a mirrored `<category>/`
subtree under `--labels` (step 3 is invoked once per leaf). Dataset filenames are flattened from
the relative path (e.g. `clean_positive/clean_positive_img_0000.png` ->
`clean_positive_clean_positive_img_0000.png`) so cross-category stems never collide, while label
lookup stays stem-aligned within each leaf.

A separate real validation set (e.g. Open Images) can be supplied via `--val-images` and
`--val-labels`. In that case the synthetic tree is used entirely for training and the provided
real images/labels are used for validation.

Usage:
  python pipeline/train_yolo.py --images data/images --labels data/labels \
      --out data/dataset --base models/yolov8n.pt --epochs 50 --imgsz 640 --batch 16
  python pipeline/train_yolo.py --images data/images --labels data/labels \
      --val-images data/openimages_val/images --val-labels data/openimages_val/labels \
      --out data/dataset --base models/yolov8n.pt --epochs 500 --patience 15
  python pipeline/train_yolo.py --images data/images_test --labels data/labels_test \
      --out data/dataset_test --base models/yolov8n.pt --epochs 1 --imgsz 320 --batch 1 \
      --allow-dummy-labels
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("train")

CLASSES = ["wrench"]
IMG_SUFFIXES = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble YOLO dataset + fine-tune (step 5).")
    p.add_argument("--images", required=True, help="dir of clean images (step 2 output, fallback)")
    p.add_argument("--images-pp", default=None,
                   help="dir of post-processed images (step 3.5). Default: data/images_pp/ if it "
                        "exists, else fall back to --images.")
    p.add_argument("--labels", required=True, help="dir of YOLO .txt labels (step 3 output)")
    p.add_argument("--val-images", default=None,
                   help="optional real validation images dir (e.g. Open Images)")
    p.add_argument("--val-labels", default=None,
                   help="optional real validation YOLO labels dir (paired with --val-images)")
    p.add_argument("--out", required=True, help="dataset dir to assemble (data.yaml written here)")
    p.add_argument("--base", default="yolov8n.pt",
                   help="base YOLO weights (models/yolov8n.pt or models/yolo26n.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=50,
                   help="early-stopping patience (epochs with no val improvement before stopping)")
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


def _collect_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)


def _link_image_and_label(
    img: Path,
    images_dir: Path,
    labels_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    allow_dummy: bool,
    missing_as_negative: bool,
) -> bool:
    """Symlink one image and its paired label into the YOLO dataset tree.

    If ``missing_as_negative`` is True, images without a label file (or with an empty
    label file) are kept as negative examples by writing an empty ``.txt``. This is
    useful for synthetic data where the generator sometimes produces no target object
    even in "positive" prompts.
    """
    flat = _flat_name(img, images_dir)
    stem = Path(flat).stem
    lbl = labels_dir / img.relative_to(images_dir).with_suffix(".txt")
    dst_lbl = out_lbl_dir / (stem + ".txt")

    if not lbl.exists() or lbl.stat().st_size == 0:
        if allow_dummy:
            dummy_label(dst_lbl)
        elif missing_as_negative:
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("", encoding="utf-8")
        else:
            log.warning("no label for %s — skipping (use --allow-dummy-labels for smoke test)", flat)
            return False
    else:
        _symlink(lbl, dst_lbl)
    _symlink(img, out_img_dir / flat)
    return True


def assemble_dataset(
    images_dir: Path,
    labels_dir: Path,
    out: Path,
    val_frac: float,
    seed: int,
    allow_dummy: bool,
    missing_as_negative: bool = True,
    val_images_dir: Path | None = None,
    val_labels_dir: Path | None = None,
) -> Path:
    train_imgs = _collect_images(images_dir)
    if not train_imgs:
        raise FileNotFoundError(f"no images in {images_dir}")

    use_real_val = val_images_dir is not None and val_labels_dir is not None
    if use_real_val:
        val_imgs = _collect_images(val_images_dir)
        if not val_imgs:
            raise FileNotFoundError(f"--val-images provided but no images in {val_images_dir}")
        log.info("Using separate real validation set: %d images from %s",
                 len(val_imgs), val_images_dir)
    else:
        rng = random.Random(seed)
        n_val = max(1, int(len(train_imgs) * val_frac)) if len(train_imgs) > 1 else 0
        val_idx = set(rng.sample(range(len(train_imgs)), n_val)) if n_val else set()
        val_imgs = [img for i, img in enumerate(train_imgs) if i in val_idx]
        train_imgs = [img for i, img in enumerate(train_imgs) if i not in val_idx]
        log.info("Random synthetic split: %d train / %d val", len(train_imgs), len(val_imgs))

    # Wipe and recreate dataset dirs so repeated runs are clean.
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n_train = 0
    for img in train_imgs:
        if _link_image_and_label(img, images_dir, labels_dir,
                                 out / "images/train", out / "labels/train",
                                 allow_dummy, missing_as_negative=True):
            n_train += 1

    n_val = 0
    if use_real_val:
        for img in val_imgs:
            if _link_image_and_label(img, val_images_dir, val_labels_dir,
                                     out / "images/val", out / "labels/val",
                                     allow_dummy, missing_as_negative=True):
                n_val += 1
    else:
        for img in val_imgs:
            if _link_image_and_label(img, images_dir, labels_dir,
                                     out / "images/val", out / "labels/val",
                                     allow_dummy, missing_as_negative=True):
                n_val += 1

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(CLASSES)),
        encoding="utf-8",
    )
    log.info("Assembled dataset at %s: %d train / %d val. data.yaml -> %s",
             out, n_train, n_val, data_yaml)
    return data_yaml


def main() -> int:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = _resolve_images(Path(a.images), Path(a.images_pp) if a.images_pp else None)
    log.info("Training on image tree: %s", images_dir)

    val_images = Path(a.val_images) if a.val_images else None
    val_labels = Path(a.val_labels) if a.val_labels else None
    if bool(val_images) != bool(val_labels):
        raise ValueError("--val-images and --val-labels must be provided together")

    data_yaml = assemble_dataset(
        images_dir, Path(a.labels), out,
        a.val_frac, a.seed, a.allow_dummy_labels,
        val_images_dir=val_images, val_labels_dir=val_labels,
    )

    if a.dry_run:
        log.info("--dry-run: dataset assembled, not training. data.yaml=%s", data_yaml)
        return 0

    from ultralytics import YOLO
    log.info("Loading base weights %s and training %d epochs (patience=%d, imgsz=%s, batch=%s, device=%s)",
             a.base, a.epochs, a.patience, a.imgsz, a.batch, a.device)
    model = YOLO(a.base)
    project_dir = (out / "runs").resolve()
    model.train(data=str(data_yaml), epochs=a.epochs, patience=a.patience,
                imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=str(project_dir), name="train", exist_ok=True, verbose=True)
    log.info("Done. Weights under %s/runs/train/", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
