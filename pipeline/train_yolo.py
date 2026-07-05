#!/usr/bin/env python3
"""Step 5 — assemble a single-class YOLO dataset and fine-tune."""
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
    p.add_argument("--images", required=True)
    p.add_argument("--images-pp", default=None)
    p.add_argument("--labels", required=True)
    p.add_argument("--val-images", default=None)
    p.add_argument("--val-labels", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-dummy-labels", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def resolve_images(images: Path, images_pp: Path | None) -> Path:
    if images_pp is not None:
        return images_pp
    pp = Path("data/images_pp")
    return pp if pp.exists() else images


def dummy_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")


def flat_name(img: Path, images_dir: Path) -> str:
    return img.relative_to(images_dir).as_posix().replace("/", "_")


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)


def link_one(
    img: Path,
    images_dir: Path,
    labels_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    allow_dummy: bool,
    missing_as_negative: bool,
) -> bool:
    flat = flat_name(img, images_dir)
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
            log.warning("No label for %s — skipping", flat)
            return False
    else:
        symlink(lbl, dst_lbl)

    symlink(img, out_img_dir / flat)
    return True


def link_split(
    images: list[Path],
    images_dir: Path,
    labels_dir: Path,
    out: Path,
    split: str,
    allow_dummy: bool,
    missing_as_negative: bool = True,
) -> int:
    img_dir = out / f"images/{split}"
    lbl_dir = out / f"labels/{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in images:
        if link_one(img, images_dir, labels_dir, img_dir, lbl_dir, allow_dummy, missing_as_negative):
            n += 1
    return n


def split_train_val(images: list[Path], val_frac: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    n_val = max(1, int(len(images) * val_frac)) if len(images) > 1 else 0
    val_idx = set(rng.sample(range(len(images)), n_val)) if n_val else set()
    val = [img for i, img in enumerate(images) if i in val_idx]
    train = [img for i, img in enumerate(images) if i not in val_idx]
    return train, val


def write_data_yaml(out: Path) -> Path:
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(CLASSES)),
        encoding="utf-8",
    )
    return data_yaml


def assemble_dataset(
    images_dir: Path,
    labels_dir: Path,
    out: Path,
    val_frac: float,
    seed: int,
    allow_dummy: bool,
    val_images_dir: Path | None = None,
    val_labels_dir: Path | None = None,
) -> Path:
    train_imgs = collect_images(images_dir)
    if not train_imgs:
        raise FileNotFoundError(f"no images in {images_dir}")

    use_real_val = val_images_dir is not None and val_labels_dir is not None
    if use_real_val:
        val_imgs = collect_images(val_images_dir)
        if not val_imgs:
            raise FileNotFoundError(f"--val-images provided but no images in {val_images_dir}")
        log.info("Using real validation set: %d images", len(val_imgs))
    else:
        train_imgs, val_imgs = split_train_val(train_imgs, val_frac, seed)
        log.info("Synthetic split: %d train / %d val", len(train_imgs), len(val_imgs))

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n_train = link_split(train_imgs, images_dir, labels_dir, out, "train", allow_dummy)
    if use_real_val:
        n_val = link_split(val_imgs, val_images_dir, val_labels_dir, out, "val", allow_dummy)
    else:
        n_val = link_split(val_imgs, images_dir, labels_dir, out, "val", allow_dummy)

    data_yaml = write_data_yaml(out)
    log.info("Assembled dataset at %s: %d train / %d val", out, n_train, n_val)
    return data_yaml


def main() -> int:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = resolve_images(Path(a.images), Path(a.images_pp) if a.images_pp else None)
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
        log.info("--dry-run: dataset assembled, not training.")
        return 0

    from ultralytics import YOLO
    log.info("Loading %s and training %d epochs (imgsz=%s, batch=%s, device=%s)",
             a.base, a.epochs, a.imgsz, a.batch, a.device)
    model = YOLO(a.base)
    project_dir = (out / "runs").resolve()
    model.train(
        data=str(data_yaml),
        epochs=a.epochs,
        patience=a.patience,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    log.info("Done. Weights under %s/runs/train/", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
