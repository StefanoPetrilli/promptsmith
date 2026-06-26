"""Standalone FLUX.2-klein-4B quality probe (NF4 on a 6 GB Turing GPU).

Why this loader is custom:
  - FLUX.2-klein-4B's text encoder is **Qwen3** (~8 GB fp16). It cannot fit the 6 GB GPU
    alongside the 4B transformer. We load Qwen3 on **CPU**, precompute prompt embeddings,
    delete it, then feed `prompt_embeds` into the pipeline so Qwen3 is never called again
    (and never moved to GPU).
  - The transformer is bf16 (8 GB) -> too big for 6 GB. We quantize it to **bitsandbytes NF4**
    on load via `quantization_config=BitsAndBytesConfig(load_in_4bit=True, nf4, double_quant,
    compute_dtype=fp16)`. On Turing (sm75), ordinary fp8 storage upcasts to bf16 and OOMs;
    NF4 actually works and keeps the transformer at ~2 GB.
  - We do NOT call `enable_model_cpu_offload()` (it would move Qwen3 -> GPU and OOM). Instead
    we precompute embeddings and manually place transformer+VAE on cuda. The pipeline's
    `_execution_device` resolves to cuda because VAE (first registered module) is on cuda.

Usage:
  python experiments/try_flux2.py --model black-forest-labs/FLUX.2-klein-4B \
      --prompts experiments/svhn_prompts.txt --out experiments/out/flux2 \
      --size 768 --steps 8 --dtype fp16 --limit 0

First run downloads ~16 GB (transformer 7.75 GB + Qwen3 8 GB + VAE/sched). Cached afterwards.
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
log = logging.getLogger("try_flux2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX.2-klein-4B NF4 quality probe for SVHN-style digits")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--prompts", default="experiments/svhn_prompts.txt")
    p.add_argument("--out", default="experiments/out/flux2")
    p.add_argument("--size", type=int, default=768, help="square image edge (must be multiple of 16)")
    p.add_argument("--steps", type=int, default=8, help="distilled model -> few steps OK")
    p.add_argument("--guidance", type=float, default=4.0, help="(ignored for distilled CFG but kept for API)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--limit", type=int, default=0, help="0 = all prompts; else first N")
    return p.parse_args()


def load_prompts(path: str, limit: int) -> list[str]:
    lines = [l.strip() for l in Path(path).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    return lines[:limit] if limit > 0 else lines


def embed_cache_path(args, prompts: list[str]) -> Path:
    key = hashlib.sha1(
        json.dumps({"m": args.model, "n": len(prompts), "p": prompts}).encode()
    ).hexdigest()[:16]
    return Path(args.out) / f"_embeds_{key}.pt"


def load_or_compute_embeds(args, prompts: list[str], dtype: torch.dtype):
    """Cache prompt embeddings to disk so reruns skip the ~9-min CPU Qwen3 pass.
    Keyed by model + exact prompt list (hash), stored alongside outputs."""
    cache = embed_cache_path(args, prompts)
    if cache.exists():
        log.info("Found cached embeddings: %s -> loading (skipping Qwen3)", cache)
        data = torch.load(cache, map_location="cpu", weights_only=False)
        # tokenizer is cheap to reload from HF cache; Qwen3 is the expensive part
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
        return data["embeds"], tok
    embeds, tok = precompute_prompt_embeds(args.model, prompts, dtype)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeds": embeds, "prompts": prompts}, cache)
    log.info("Saved cached embeddings: %s", cache)
    return embeds, tok


def precompute_prompt_embeds(model_id: str, prompts: list[str], dtype: torch.dtype) -> list[torch.Tensor]:
    """Load Qwen3 + tokenizer on CPU, compute FLUX.2 prompt embeddings (layers 9,18,27).

    Uses the pipeline's own static method so the embedding layout matches what
    Flux2KleinPipeline.__call__ expects via `prompt_embeds`. Returns one (1, L, D) tensor
    per prompt on CPU (fp16).
    """
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


def build_pipeline(model_id: str, dtype: torch.dtype, tokenizer):
    """Build Flux2KleinPipeline with NF4 transformer on cuda + fp16 VAE on cuda.

    text_encoder is replaced by a tiny dummy module: Qwen3 was only needed to compute
    prompt embeddings, which we now pass directly via `prompt_embeds`. The dummy keeps
    construction happy without occupying any notable RAM/VRAM.

    We subclass Flux2KleinPipeline to override `_execution_device` -> cuda explicitly: the
    default property falls back to `self.device`, which can't be inferred from a bitsandbytes
    NF4 transformer + dummy text_encoder, and raises AttributeError. We bypass that.
    """
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
    log.info("Loading Flux2Transformer2DModel as NF4 (this downloads + quantizes ~7.75 GB)...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=bnb_cfg,
        torch_dtype=dtype,
    )
    transformer.to("cuda")
    # free CUDA caches after quantization shuffling
    torch.cuda.empty_cache()

    log.info("Loading AutoencoderKLFlux2 (fp16) -> cuda...")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to("cuda")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    # dummy text_encoder with one param so Flux2KleinPipeline __init__ is happy.
    dummy_te = torch.nn.Linear(1, 1, bias=True)
    pipe = Flux2KleinPipelineFixed(
        scheduler=scheduler,
        vae=vae,
        text_encoder=dummy_te,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    )
    # IMPORTANT: do NOT enable_model_cpu_offload (would lift dummy/real te offloading rules;
    # here it's harmless, but we deliberately keep transformer+vae on cuda and Qwen3 is gone).
    return pipe


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    prompts = load_prompts(a.prompts, a.limit)
    log.info("Loaded %d prompts", len(prompts))

    embeds, tok = load_or_compute_embeds(a, prompts, dtype)
    pipe = build_pipeline(a.model, dtype, tok)

    generator = torch.Generator(device="cuda").manual_seed(a.seed)
    for i, (prompt, pe) in enumerate(zip(prompts, embeds)):
        pe_cuda = pe.to("cuda", dtype=dtype)
        log.info("[%d/%d] generating (steps=%s, size=%s): %s",
                 i + 1, len(prompts), a.steps, a.size, prompt[:80])
        t0 = time.time()
        try:
            img = pipe(
                prompt=None,
                prompt_embeds=pe_cuda,
                height=a.size,
                width=a.size,
                num_inference_steps=a.steps,
                guidance_scale=a.guidance,
                generator=generator,
            ).images[0]
        except torch.cuda.OutOfMemoryError as e:
            log.error("OOM on prompt %d (%.1fs): %s", i, time.time() - t0, e)
            torch.cuda.empty_cache()
            return 3
        path = out_dir / f"flux2_{i:03d}.png"
        img.save(path)
        log.info("  -> saved %s (%.1fs)", path, time.time() - t0)
        torch.cuda.empty_cache()

    log.info("Done. Inspect %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())