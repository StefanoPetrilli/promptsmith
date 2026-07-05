#!/usr/bin/env python3
"""Step 3.5 — deterministic Albumentations degradation of rendered images.

Runs AFTER step 3 (SAM 3 boxes are already drawn/recorded on the clean images, where SAM 3 is
most accurate). Produces a degraded image tree that step 5 trains on, while step 3's labels stay
aligned by stem: the output keeps the same indices/filenames as the input so `<stem>.txt` labels
still match.

    Input : data/images/<category>/  (+ manifest.jsonl from step 2)
    Output: data/images_pp/<category>/<prefix>img_{i:04d}.png + manifest_pp.jsonl

CPU-only (Albumentations). Never loads FLUX or SAM. Strength is sampled from a seeded RNG
(`seed + i`, matching step 2's per-image seeding) and kept mild — pixel-only ops, no geometric
transforms, so YOLO boxes computed on the clean images remain valid for the degraded images.

Resume: on startup the output dir is scanned for existing `img_{i:04d}.png` files; those indices
are skipped and `manifest_pp.jsonl` is backfilled if missing (mirrors step 2's resume contract).
`--limit N` processes only the first N entries (test).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("postprocess")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Albumentations degradation (step 3.5).")
    p.add_argument("--images", required=True,
                   help="a step-2 manifest.jsonl, or a dir of img_*.png")
    p.add_argument("--out", required=True, help="output dir for degraded PNGs + manifest_pp.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 = all; else first N (test)")
    p.add_argument("--restart", action="store_true",
                   help="ignore existing output images; reprocess everything")
    return p.parse_args()


def collect_entries(arg: str, limit: int) -> list[dict]:
    """Return [{index, image, prompt}, ...] preserving step-2 indices/stems.

    A manifest.jsonl keeps the original indices; a bare dir is enumerated in sorted filename
    order with index = position (stems still match because filenames are preserved)."""
    src = Path(arg)
    if src.is_file() and src.suffix == ".jsonl":
        entries = []
        seen = set()
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            ip = Path(d["image"])
            if ip in seen:
                continue
            seen.add(ip)
            entries.append({"index": int(d["index"]), "image": str(ip),
                            "prompt": d.get("prompt", "")})
        return entries[:limit] if limit > 0 else entries
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if limit > 0:
            imgs = imgs[:limit]
        return [{"index": i, "image": str(p), "prompt": ""} for i, p in enumerate(imgs)]
    raise FileNotFoundError(arg)


def build_pipeline():
    """Mild, pixel-only Albumentations (no geometry -> label boxes stay valid)."""
    import albumentations as A

    return A.Compose([
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.12), p=1.0),
        ], p=0.3),
        A.ImageCompression(compression_type="jpeg", quality_range=(55, 88), p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.Blur(blur_limit=(3, 3), p=1.0),
        ], p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.RandomFog(fog_coef_range=(0.05, 0.25), alpha_coef=0.08, p=0.15),
        A.CLAHE(clip_limit=(1.0, 3.0), p=0.1),
        A.ToGray(p=0.05),
    ], p=1.0)


def find_completed(out_dir: Path, stems: list[str]) -> set[str]:
    done = set()
    for s in stems:
        if (out_dir / f"{s}.png").exists():
            done.add(s)
    return done


def backfill_manifest(out_dir: Path, entries: list[dict], done: set[str],
                      manifest: Path) -> None:
    have = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                have.add(int(rec["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    missing = [e for e in entries if e["index"] not in have
               and Path(e["image"]).stem in done]
    if missing:
        with manifest.open("a", encoding="utf-8") as mf:
            for e in missing:
                src = Path(e["image"])
                dst = out_dir / f"{src.stem}.png"
                mf.write(json.dumps({"index": e["index"], "image": str(dst),
                                     "source": str(src), "prompt": e["prompt"]}) + "\n")
        log.info("Backfilled %d missing manifest_pp entries.", len(missing))


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = collect_entries(a.images, a.limit)
    if not entries:
        log.error("no images found in %s", a.images)
        return 2

    stems = [Path(e["image"]).stem for e in entries]
    manifest = out_dir / "manifest_pp.jsonl"

    done = set() if a.restart else find_completed(out_dir, stems)
    if done:
        backfill_manifest(out_dir, entries, done, manifest)
        log.info("Resume: %d/%d outputs already present -> skipping them.", len(done), len(entries))
        if len(done) == len(entries):
            log.info("All %d outputs already present. Nothing to do.", len(entries))
            return 0
    else:
        log.info("Starting from scratch (no existing images in %s).", out_dir)

    pipe = build_pipeline()
    todo = [e for e in entries if Path(e["image"]).stem not in done]
    with manifest.open("a", encoding="utf-8") as mf:
        for e in tqdm(todo, total=len(todo), desc="postprocess", unit="img"):
            src = Path(e["image"])
            stem = src.stem
            # Per-image deterministic RNG (seed + i), matching step 2's seeding scheme.
            iseed = (a.seed + int(e["index"])) % (2**32)
            random.seed(iseed)
            np.random.seed(iseed)
            pipe.set_random_seed(iseed)
            img = Image.open(src).convert("RGB")
            arr = np.array(img)
            out = pipe(image=arr)["image"]
            dst = out_dir / f"{stem}.png"
            Image.fromarray(out).save(dst)
            mf.write(json.dumps({"index": e["index"], "image": str(dst),
                                 "source": str(src), "prompt": e["prompt"]}) + "\n")
            mf.flush()
            tqdm.write(f"  saved {dst.name} <- {src.name}")

    log.info("Done. %d degraded images in %s (manifest: %s)", len(entries), out_dir, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
