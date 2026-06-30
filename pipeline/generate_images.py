#!/usr/bin/env python3
"""Step 2 — render prompts into images with FLUX.2-klein-4B (NF4, 6 GB Turing GPU).

Reuses the validated loader from experiments/try_flux2.py:

  * Qwen3 text encoder runs on **CPU**, prompt embeddings are precomputed + cached to disk,
    then Qwen3 is released (it never goes on the GPU).
  * The 4B transformer is loaded as **bitsandbytes NF4** on cuda (~2 GB).
  * VAE (fp16) on cuda. No `enable_model_cpu_offload` (would pull Qwen3 -> GPU and OOM).

This script is a thin, importable wrapper around that loader so the pipeline can call it
with any prompts file produced by step 1. Inputs are read as one prompt per line (`#`
comments and blank lines skipped); outputs are PNGs named `img_{i:04d}.png` plus a
`manifest.jsonl` recording prompt -> image mapping (used by step 3).

Test mode: `--limit 3` renders just the first 3 prompts for a quick smoke test.

Usage:
  python pipeline/generate_images.py --prompts data/prompts.txt --out data/images
  python pipeline/generate_images.py --prompts data/prompts_test.txt --out data/images_test --limit 3
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("flux2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX.2-klein-4B image generation (step 2).")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--prompts", required=True, help="one prompt per line (output of step 1)")
    p.add_argument("--out", required=True, help="output directory for PNGs + manifest")
    p.add_argument("--size", type=int, default=768, help="square image edge (multiple of 16)")
    p.add_argument("--steps", type=int, default=8, help="distilled model -> few steps OK")
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--limit", type=int, default=0, help="0 = all prompts; else first N (test)")
    return p.parse_args()


def load_prompts(path: str, limit: int) -> list[str]:
    lines = [l.strip() for l in Path(path).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    return lines[:limit] if limit > 0 else lines


def embed_cache_path(out: str, model: str, prompts: list[str]) -> Path:
    key = hashlib.sha1(
        json.dumps({"m": model, "n": len(prompts), "p": prompts}).encode()
    ).hexdigest()[:16]
    return Path(out) / f"_embeds_{key}.pt"


def precompute_prompt_embeds(model_id: str, prompts: list[str], dtype: torch.dtype):
    """Load Qwen3 on CPU, compute FLUX.2 prompt embeddings (layers 9,18,27), release it."""
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    log.info("Loading Qwen3 text encoder on CPU (fp16) for prompt-embedding precompute...")
    tok = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    te = Qwen3ForCausalLM.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cpu").eval()
    log.info("Qwen3 loaded. Computing embeddings for %d prompts...", len(prompts))

    embeds = []
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            e = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                text_encoder=te,
                tokenizer=tok,
                prompt=prompt,
                device=torch.device("cpu"),
                dtype=dtype,
                max_sequence_length=512,
                hidden_states_layers=(9, 18, 27),
            )
            embeds.append(e.detach().to("cpu", dtype=dtype))
            log.info("  embed [%d/%d] shape=%s", i + 1, len(prompts), tuple(e.shape))

    log.info("Releasing Qwen3 from CPU RAM...")
    del te
    gc.collect()
    return embeds, tok


def load_or_compute_embeds(model_id: str, out: str, prompts: list[str], dtype: torch.dtype):
    """Cache prompt embeddings to disk so reruns skip the slow CPU Qwen3 pass."""
    cache = embed_cache_path(out, model_id, prompts)
    if cache.exists():
        log.info("Found cached embeddings: %s -> loading (skipping Qwen3)", cache)
        data = torch.load(cache, map_location="cpu", weights_only=False)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        return data["embeds"], tok
    embeds, tok = precompute_prompt_embeds(model_id, prompts, dtype)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeds": embeds, "prompts": prompts}, cache)
    log.info("Saved cached embeddings: %s", cache)
    return embeds, tok


def build_pipeline(model_id: str, dtype: torch.dtype, tokenizer):
    """Flux2KleinPipeline with NF4 transformer + fp16 VAE on cuda; dummy text_encoder."""
    from transformers import BitsAndBytesConfig
    from diffusers import (
        AutoencoderKLFlux2,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
        FlowMatchEulerDiscreteScheduler,
    )

    class Flux2KleinPipelineFixed(Flux2KleinPipeline):
        @property
        def _execution_device(self):
            return torch.device("cuda")

        @property
        def device(self):
            return torch.device("cuda")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    log.info("Loading Flux2Transformer2DModel as NF4 (~7.75 GB download on first run)...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", quantization_config=bnb_cfg, torch_dtype=dtype,
    )
    transformer.to("cuda")
    torch.cuda.empty_cache()

    log.info("Loading AutoencoderKLFlux2 (fp16) -> cuda...")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to("cuda")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    dummy_te = torch.nn.Linear(1, 1, bias=True)  # Qwen3 replaced by precomputed embeds
    pipe = Flux2KleinPipelineFixed(
        scheduler=scheduler, vae=vae, text_encoder=dummy_te,
        tokenizer=tokenizer, transformer=transformer, is_distilled=True,
    )
    return pipe


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    prompts = load_prompts(a.prompts, a.limit)
    if not prompts:
        log.error("no prompts loaded from %s", a.prompts)
        return 2
    log.info("Loaded %d prompts from %s", len(prompts), a.prompts)

    embeds, tok = load_or_compute_embeds(a.model, a.out, prompts, dtype)
    pipe = build_pipeline(a.model, dtype, tok)

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as mf:
        gen = torch.Generator(device="cuda").manual_seed(a.seed)
        for i, (prompt, pe) in enumerate(zip(prompts, embeds)):
            pe_cuda = pe.to("cuda", dtype=dtype)
            log.info("[%d/%d] generating (steps=%s, size=%s): %s",
                     i + 1, len(prompts), a.steps, a.size, prompt[:80])
            t0 = time.time()
            try:
                img = pipe(
                    prompt=None, prompt_embeds=pe_cuda,
                    height=a.size, width=a.size,
                    num_inference_steps=a.steps, guidance_scale=a.guidance,
                    generator=gen,
                ).images[0]
            except torch.cuda.OutOfMemoryError as e:
                log.error("OOM on prompt %d (%.1fs): %s", i, time.time() - t0, e)
                torch.cuda.empty_cache()
                return 3
            path = out_dir / f"img_{i:04d}.png"
            img.save(path)
            mf.write(json.dumps({"index": i, "image": str(path), "prompt": prompt}) + "\n")
            mf.flush()
            log.info("  -> saved %s (%.1fs)", path, time.time() - t0)
            torch.cuda.empty_cache()

    log.info("Done. %d images in %s (manifest: %s)", len(prompts), out_dir, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
