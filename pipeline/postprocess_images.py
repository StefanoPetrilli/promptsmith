#!/usr/bin/env python3
"""Step 3.5 — Albumentations degradation of rendered images.

The degradation suite is tuned to mimic the full mix of capture devices
we expect in production: USB webcams, integrated laptop cameras, phone
cameras, dashcams, CCTV / analogue-surveillance, and IP cameras. Each of
these has its own signature failure mode (sensor noise, on-device ISP
sharpening, harsh MPEG/MJPEG compression, low resolution, lens optics,
colour cast from auto white balance, flare/ghosting, and light drops).
Transforms are grouped into "failure modes" so that a degraded image
reads like one plausible acquisition rather than a random soup of
artifacts. Execution is parallelised across worker processes so we
actually use the cores we have.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("postprocess")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Albumentations degradation (step 3.5).")
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes. Set to 1 to disable parallelism.",
    )
    return p.parse_args()


def collect_entries(arg: str, limit: int) -> list[dict]:
    src = Path(arg)
    if src.is_file() and src.suffix == ".jsonl":
        entries = []
        seen = set()
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            ip = Path(d["image"])
            if ip not in seen:
                seen.add(ip)
                entries.append({"index": int(d["index"]), "image": str(ip), "prompt": d.get("prompt", "")})
        return entries[:limit] if limit > 0 else entries
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if limit > 0:
            imgs = imgs[:limit]
        return [{"index": i, "image": str(p), "prompt": ""} for i, p in enumerate(imgs)]
    raise FileNotFoundError(arg)


def build_pipeline():
    """Assemble the low-quality-camera degradation pipeline.

    Transforms are grouped into independent failure modes, each wrapped in
    a OneOf so a plausible subset is applied per image. Probabilities/ranges
    sit between "sterile rendering" and "extreme glitch" — enough to harden
    the downstream model against the capture conditions we will see in
    production without washing out all signal.
    """
    import cv2
    import albumentations as A

    # NOTE: tuned "mild" — close to typical phone/webcam photos rather than
    # worst-case CCTV. Heavy GlassBlur and harsh JPEG were pushing the synthetic
    # distribution *away* from the real verification photos, hurting transfer.
    return A.Compose([
        # --- low resolution / soft optics (mild) ---
        A.OneOf([
            A.Downscale(
                scale_range=(0.7, 0.9),
                interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                p=1.0,
            ),
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=1.0),
            A.ZoomBlur(max_factor=(1.0, 1.08), step_factor=(0.01, 0.02), p=1.0),
        ], p=0.3),

        # --- motion / general blur (rare, mild) ---
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.AdvancedBlur(blur_limit=(3, 5), sigma_x_limit=(0.2, 0.8), sigma_y_limit=(0.2, 0.8),
                           rotate_limit=(-45, 45), beta_limit=(0.5, 6.0), p=1.0),
            A.Blur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),

        # --- cheap-ISP oversharpening + JPEG ringing ---
        A.OneOf([
            A.UnsharpMask(blur_limit=(3, 5), sigma_limit=(0.0, 0.8), alpha=(0.2, 0.5), p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.RingingOvershoot(blur_limit=(5, 11), cutoff=(0.78, 1.57), p=1.0),
        ], p=0.2),

        # --- sensor noise (low-light / cheap sensors, mild) ---
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.25), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ShotNoise(scale_range=(0.1, 0.25), p=1.0),
            A.SaltAndPepper(amount=(0.001, 0.008), salt_vs_pepper=(0.4, 0.6), p=1.0),
        ], p=0.3),

        # --- codec compression (realistic phone JPEG, not harsh) ---
        A.ImageCompression(compression_type="jpeg", quality_range=(70, 92), p=0.3),

        # --- white balance / colour cast / ISP colour ---
        A.OneOf([
            A.PlanckianJitter(mode="blackbody", sampling_method="uniform", p=1.0),
            A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.88, 1.12),
                          saturation=(0.8, 1.2), hue=(-0.04, 0.04), p=1.0),
            A.HueSaturationValue(hue_shift_limit=(-15, 15), sat_shift_limit=(-25, 25),
                                 val_shift_limit=(-15, 15), p=1.0),
        ], p=0.3),

        # --- exposure / auto-gain ---
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.35),
        A.RandomGamma(gamma_limit=(80, 120), p=0.25),

        # --- cheap-lens chromatic aberration ---
        A.ChromaticAberration(
            primary_distortion_limit=(-0.03, 0.03),
            secondary_distortion_limit=(-0.06, 0.06),
            mode="random", p=0.15,
        ),

        # --- light / environment artefacts (flare, shadow, light haze) ---
        A.OneOf([
            A.RandomShadow(shadow_roi=(0.0, 0.5, 1.0, 1.0), num_shadows_limit=(1, 2),
                           shadow_intensity_range=(0.3, 0.5), p=1.0),
            (A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5), angle_range=(0.0, 1.0),
                               num_flare_circles_range=(6, 10), method="physics_based",
                               p=1.0)
             if hasattr(A, "RandomSunFlare") else A.NoOp(p=1.0)),
            A.RandomFog(fog_coef_range=(0.05, 0.15), alpha_coef=0.08, p=1.0),
        ], p=0.1),

        # --- rare flat / muted capture ---
        A.OneOf([
            A.ToGray(p=1.0),
            A.ToSepia(p=1.0),
        ], p=0.03),

        # --- modest tone curve wobble (auto contrast heuristics) ---
        A.RandomToneCurve(scale=0.1, p=0.1),
    ], p=1.0)


def find_completed(out_dir: Path, stems: list[str]) -> set[str]:
    return {s for s in stems if (out_dir / f"{s}.png").exists()}


def backfill_manifest(out_dir: Path, entries: list[dict], done: set[str], manifest: Path) -> None:
    have = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                have.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    missing = [
        e for e in entries
        if e["index"] not in have and Path(e["image"]).stem in done
    ]
    if missing:
        with manifest.open("a", encoding="utf-8") as mf:
            for e in missing:
                src = Path(e["image"])
                dst = out_dir / f"{src.stem}.png"
                mf.write(json.dumps({
                    "index": e["index"],
                    "image": str(dst),
                    "source": str(src),
                    "prompt": e["prompt"],
                }) + "\n")
        log.info("Backfilled %d missing manifest_pp entries.", len(missing))


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------
#
# Each worker process builds the pipeline once and reuses it. Compose objects
# are picklable, but rebuilding them in the worker is cheaper than repeatedly
# sending them over the pipe, and it avoids any lazy state bleeding over
# between processes.
_PIPELINE_CACHE: dict[str, object] = {}


def _get_pipeline():
    p = _PIPELINE_CACHE.get("main")
    if p is None:
        p = build_pipeline()
        _PIPELINE_CACHE["main"] = p
    return p


def _worker(task: dict) -> dict:
    """Run in a worker process: degrade one image, write it, return a manifest line."""
    src = Path(task["image"])
    out_dir = Path(task["out"])
    seed = task["seed"]
    index = task["index"]
    prompt = task["prompt"]

    iseed = (seed + index) % (2**32)
    random.seed(iseed)
    np.random.seed(iseed)
    pipeline = _get_pipeline()
    pipeline.set_random_seed(iseed)

    arr = np.array(Image.open(src).convert("RGB"))
    img = Image.fromarray(pipeline(image=arr)["image"])

    dst = out_dir / f"{src.stem}.png"
    img.save(dst)
    return {
        "path": str(dst),
        "source": str(src),
        "index": index,
        "prompt": prompt,
    }


def process_one(src: Path, pipeline, seed: int, index: int) -> Image.Image:
    iseed = (seed + index) % (2**32)
    random.seed(iseed)
    np.random.seed(iseed)
    pipeline.set_random_seed(iseed)
    arr = np.array(Image.open(src).convert("RGB"))
    return Image.fromarray(pipeline(image=arr)["image"])


def _write_manifest_line(mf, index: int, dst: str, src: str, prompt: str) -> None:
    mf.write(json.dumps({
        "index": index,
        "image": str(dst),
        "source": str(src),
        "prompt": prompt,
    }) + "\n")
    mf.flush()


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = collect_entries(a.images, a.limit)
    if not entries:
        log.error("No images found in %s", a.images)
        return 2

    stems = [Path(e["image"]).stem for e in entries]
    manifest = out_dir / "manifest_pp.jsonl"

    done = find_completed(out_dir, stems)
    if done:
        backfill_manifest(out_dir, entries, done, manifest)
        log.info("Resume: %d/%d outputs already present.", len(done), len(entries))
        if len(done) == len(entries):
            log.info("All outputs already present.")
            return 0

    todo = [e for e in entries if Path(e["image"]).stem not in done]
    if not todo:
        log.info("Nothing to do.")
        return 0

    tasks = [
        {**e, "out": str(out_dir), "seed": a.seed}
        for e in todo
    ]
    workers = max(1, a.workers)
    log.info("Postprocessing %d images with %d worker(s).", len(todo), workers)

    with manifest.open("a", encoding="utf-8") as mf:
        if workers == 1:
            # Serial path — keep behaviour identical without the pool overhead.
            pipeline = build_pipeline()
            for e in tqdm(todo, total=len(todo), desc="postprocess", unit="img"):
                src = Path(e["image"])
                img = process_one(src, pipeline, a.seed, e["index"])
                dst = out_dir / f"{src.stem}.png"
                img.save(dst)
                _write_manifest_line(mf, e["index"], str(dst), str(src), e["prompt"])
                tqdm.write(f"  saved {dst.name} <- {src.name}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_worker, t): t for t in tasks}
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="postprocess", unit="img"):
                    res = fut.result()
                    _write_manifest_line(mf, res["index"], res["path"],
                                          res["source"], res["prompt"])

    log.info("Done. %d degraded images in %s", len(entries), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())