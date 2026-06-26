"""Standalone SANA quality probe (not wired into the numeri pipeline yet).

Goal: before committing SANA as a generation backend, eyeball whether it can render
readable, varied house-number digits in plausible SVHN-style street scenes on this
machine (RTX 2060, 6 GB VRAM).

Usage:
  python experiments/try_sana.py --model mit-han-lab/sana-pipeline-DMD-4steps-512 \\
      --prompts experiments/svhn_prompts.txt --out experiments/out/sana \\
      --size 512 --steps 4 --guidance 4.0 --dtype fp16 --offload

Notes:
  - 512px DMD 4-step variant is recommended for 6 GB VRAM. 1024px may OOM without
    `--offload` (and even then it's tight).
  - Models must be supported by `diffusers.SanaPipeline`. If the auto pipeline lacks
    SANA, bump diffusers to a recent version (>=0.31).
  - First run downloads the model into the HF cache (~1-2 GB).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from diffusers import SanaPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("try_sana")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SANA quality probe for SVHN-style synthetic images")
    p.add_argument("--model", default="mit-han-lab/sana-pipeline-DMD-4steps-512",
                   help="HF id of a SANA pipeline (e.g. ...-512, ...-1024, DMD 4-step variants).")
    p.add_argument("--prompts", default="experiments/svhn_prompts.txt")
    p.add_argument("--out", default="experiments/out/sana")
    p.add_argument("--size", type=int, default=512, help="edge of square image (matches model config ideally)")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--offload", action="store_true",
                   help="enable_model_cpu_offload to survive tight VRAM (slower)")
    p.add_argument("--limit", type=int, default=0, help="0 = all prompts; else first N")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in Path(a.prompts).read_text().splitlines()
               if l.strip() and not l.lstrip().startswith("#")]
    if a.limit > 0:
        prompts = prompts[:a.limit]
    log.info("Loaded %d prompts", len(prompts))

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    log.info("Loading SANA pipeline: %s (dtype=%s, device=%s, offload=%s)",
             a.model, a.dtype, a.device, a.offload)

    pipe = SanaPipeline.from_pretrained(a.model, torch_dtype=dtype)
    if a.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(a.device)
    # Tiny throughput hint
    try:
        pipe.set_progress_bar_config(disable=False)
    except Exception:
        pass

    generator = torch.Generator(device="cpu").manual_seed(a.seed)
    for i, prompt in enumerate(prompts):
        log.info("[%d/%d] generating (steps=%s, size=%s): %s", i + 1, len(prompts), a.steps, a.size, prompt[:80])
        try:
            img = pipe(
                prompt=prompt,
                num_inference_steps=a.steps,
                guidance_scale=a.guidance,
                height=a.size,
                width=a.size,
                generator=generator,
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            log.error("OOM on prompt %d. Retry with --offload or smaller --size/cuda.", i)
            torch.cuda.empty_cache()
            return 3
        path = out_dir / f"sana_{i:03d}.png"
        img.save(path)
        log.info("  -> saved %s", path)

    log.info("Done. Inspect grid in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())