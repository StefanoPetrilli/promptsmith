#!/usr/bin/env python3
"""Step 2 — render prompts into images with FLUX.2-klein-4B (fp16).

Embeddings from Qwen3 are precomputed once, streamed to a numpy memmap, and released
before the transformer loads. Generation resumes from existing PNGs and uses an
independent per-image seed so resumes are deterministic.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("flux2")

EMBED_MEMMAP_SUFFIX = ".embeds.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX.2-klein-4B image generation (step 2).")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--pin-policy", choices=["partial", "sequential"], default="partial",
    )
    p.add_argument("--vram-margin", type=float, default=1.2)
    p.add_argument("--restart", action="store_true")
    p.add_argument("--prefix", default="")
    return p.parse_args()


def load_prompts(path: str, limit: int) -> list[str]:
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return lines[:limit] if limit > 0 else lines


def embed_cache_key(model: str, prompts: list[str]) -> str:
    payload = json.dumps({"m": model, "n": len(prompts), "p": prompts})
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def embed_cache_path(out: str, model: str, prompts: list[str]) -> Path:
    return Path(out) / f"_embeds_{embed_cache_key(model, prompts)}{EMBED_MEMMAP_SUFFIX}"


def embed_legacy_cache_path(out: str, model: str, prompts: list[str]) -> Path:
    return Path(out) / f"_embeds_{embed_cache_key(model, prompts)}.pt"


def torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def np_store_dtype(dtype: torch.dtype) -> np.dtype:
    return np.float32 if dtype == torch.float32 else np.uint16


def tensor_to_nprow(tensor: torch.Tensor, dtype: torch.dtype) -> np.ndarray:
    if dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy()
    if dtype == torch.float16:
        return tensor.numpy().view(np.uint16)
    return tensor.numpy()


def nprow_to_tensor(row: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(row.copy())
    if tensor.dtype == torch.uint16:
        tensor = tensor.view(dtype)
    return tensor


def module_size_mb(module: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in module.parameters()) / 1e6


def free_vram_gb(device: torch.device) -> float:
    free_b, _ = torch.cuda.mem_get_info(device)
    return free_b / 1e9


def _offload_hooks(module: torch.nn.Module, cuda: torch.device, cpu: torch.device):
    def pre(m, _args):
        m.to(cuda)

    def post(m, _args, output):
        m.to(cpu)
        return output

    return module.register_forward_pre_hook(pre), module.register_forward_hook(post)


def pin_layers(layers: list[torch.nn.Module], margin_gb: float) -> tuple[int, int]:
    """Pin as many layers on GPU as fit; offload the rest via hooks."""
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")
    layer_mb = module_size_mb(layers[0])
    threshold_b = margin_gb * 1e9 + layer_mb * 1e6

    n_pin = 0
    for layer in layers:
        if torch.cuda.mem_get_info(cuda)[0] < threshold_b:
            break
        layer.to(cuda)
        torch.cuda.empty_cache()
        n_pin += 1

    for layer in layers[n_pin:]:
        _offload_hooks(layer, cuda, cpu)

    return n_pin, len(layers) - n_pin


def _capture_hooks(layers: list[torch.nn.Module], target_indices: tuple[int, ...]):
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _args, output):
            captured[idx] = output
            return output
        return hook

    handles = [layers[k].register_forward_hook(make_hook(k)) for k in target_indices]
    return captured, handles


def precompute_prompt_embeds(
    model_id: str, prompts: list[str], dtype: torch.dtype, cache_path: Path, tokenizer,
):
    from transformers import Qwen3ForCausalLM

    log.info("Loading Qwen3 text encoder (bfloat16) on CPU...")
    te = Qwen3ForCausalLM.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to("cpu").eval()

    inner = te.model
    inner.embed_tokens.to("cuda")
    inner.rotary_emb.to("cuda")
    inner.norm.to("cuda")
    n_pin, n_off = pin_layers(list(inner.layers), margin_gb=0.60)
    log.info("Qwen3 pinned %d/%d decoder layers; offloading %d.", n_pin, len(inner.layers), n_off)

    target_layers = (9, 18, 27)
    captured, handles = _capture_hooks(list(inner.layers), target_layers)

    store_np = np_store_dtype(dtype)

    def embed_one(prompt: str) -> torch.Tensor:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=512,
        )
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        captured.clear()
        with torch.no_grad():
            inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        stacked = torch.stack([captured[k] for k in target_layers], dim=1).to(dtype=dtype, device="cuda")
        b, c, seq, h = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(b, seq, c * h)[0].to("cpu", dtype=dtype)

    first = embed_one(prompts[0])
    L, D = first.shape
    mm = np.lib.format.open_memmap(cache_path, mode="w+", dtype=store_np, shape=(len(prompts), L, D))
    mm[0] = tensor_to_nprow(first, dtype)
    del first
    for i in range(1, len(prompts)):
        mm[i] = tensor_to_nprow(embed_one(prompts[i]), dtype)
        if i % 50 == 0:
            mm.flush()
    mm.flush()
    del mm

    for h in handles:
        h.remove()
    del te
    gc.collect()


def memmap_is_valid(mm: np.ndarray, chunk: int = 64) -> bool:
    for start in range(0, mm.shape[0], chunk):
        ok = np.any(mm[start:start + chunk], axis=(1, 2))
        if not ok.all():
            bad = (np.where(~ok)[0] + start).tolist()
            log.warning("Memmap has zero rows (first bad: %s); recomputing.", bad[:5])
            return False
    return True


def convert_legacy_cache(legacy: Path, cache: Path, dtype: torch.dtype):
    log.info("Converting legacy .pt cache %s -> %s", legacy.name, cache.name)
    data = torch.load(legacy, map_location="cpu", mmap=True, weights_only=False)
    embeds = data["embeds"]
    first = embeds[0]
    L, D = first.shape[1], first.shape[2]
    store_np = np_store_dtype(dtype)
    mm = np.lib.format.open_memmap(cache, mode="w+", dtype=store_np, shape=(len(embeds), L, D))
    for i, e in enumerate(tqdm(embeds, desc="convert embeds", unit="row")):
        mm[i] = tensor_to_nprow(e[0].to(dtype), dtype)
    mm.flush()
    del mm, data, embeds
    gc.collect()


def load_or_compute_embeds(model_id: str, out: str, prompts: list[str], dtype: torch.dtype):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    cache = embed_cache_path(out, model_id, prompts)

    if cache.exists():
        mm = np.lib.format.open_memmap(cache, mode="r")
        if memmap_is_valid(mm):
            log.info("Using cached embeddings: %s", cache)
            return mm, tokenizer
        del mm
        cache.unlink()

    cache.parent.mkdir(parents=True, exist_ok=True)

    legacy = embed_legacy_cache_path(out, model_id, prompts)
    if legacy.exists():
        convert_legacy_cache(legacy, cache, dtype)
        return np.lib.format.open_memmap(cache, mode="r"), tokenizer

    precompute_prompt_embeds(model_id, prompts, dtype, cache, tokenizer)
    return np.lib.format.open_memmap(cache, mode="r"), tokenizer


def apply_partial_pin(pipe, vram_margin_gb: float = 1.2):
    cuda = torch.device("cuda")
    transformer = pipe.transformer

    pipe.vae.to(cuda)
    pinned_mb = module_size_mb(pipe.vae)

    small_modules = [
        "pos_embed", "time_guidance_embed",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation", "x_embedder", "context_embedder",
        "norm_out", "proj_out",
    ]
    for name in small_modules:
        getattr(transformer, name).to(cuda)
        pinned_mb += module_size_mb(getattr(transformer, name))

    for block in transformer.transformer_blocks:
        block.to(cuda)
        pinned_mb += module_size_mb(block)

    single_blocks = transformer.single_transformer_blocks
    single_mb = module_size_mb(single_blocks[0]) if single_blocks else 0.0
    free_gb = free_vram_gb(cuda)
    budget_gb = free_gb - vram_margin_gb
    n_pin = max(0, min(len(single_blocks), int(budget_gb * 1e3 / single_mb))) if single_mb else len(single_blocks)

    log.info(
        "partial-pin: free=%.2f GB, margin=%.1f GB -> pin %d/%d single blocks (%.0f MB), offload %d.",
        free_gb, vram_margin_gb, n_pin, len(single_blocks), n_pin * single_mb,
        len(single_blocks) - n_pin,
    )

    for i, block in enumerate(single_blocks):
        if i < n_pin:
            block.to(cuda)
            pinned_mb += module_size_mb(block)
        else:
            _offload_hooks(block, cuda, torch.device("cpu"))

    torch.cuda.empty_cache()
    return n_pin, len(single_blocks) - n_pin


def build_pipeline(model_id: str, dtype: torch.dtype, tokenizer, pin_policy: str, vram_margin: float):
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

    class DummyTextEncoder(torch.nn.Module):
        dtype = torch.float32

        def forward(self, *args, **kwargs):
            return None

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype,
    )
    vae = AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)

    pipe = Flux2KleinPipelineFixed(
        scheduler=scheduler,
        vae=vae,
        text_encoder=DummyTextEncoder(),
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    )

    if pin_policy == "partial":
        apply_partial_pin(pipe, vram_margin_gb=vram_margin)
    else:
        log.info("Sequential offload enabled (slow).")
        pipe.enable_sequential_cpu_offload(device=torch.device("cuda"))

    torch.cuda.empty_cache()
    return pipe


def image_name(prefix: str, i: int) -> str:
    return f"{prefix}img_{i:04d}.png"


def find_completed(out_dir: Path, prefix: str, n: int) -> set[int]:
    return {i for i in range(n) if (out_dir / image_name(prefix, i)).exists()}


def backfill_manifest(
    out_dir: Path,
    prefix: str,
    prompts: list[str],
    done: set[int],
    manifest: Path,
) -> set[int]:
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

    missing = sorted(done - have)
    if missing:
        with manifest.open("a", encoding="utf-8") as mf:
            for i in missing:
                path = out_dir / image_name(prefix, i)
                mf.write(json.dumps({"index": i, "image": str(path), "prompt": prompts[i]}) + "\n")
        log.info("Backfilled %d missing manifest entries.", len(missing))
    return have


def generate_one(
    pipe,
    embeds_mm: np.ndarray,
    dtype: torch.dtype,
    prompt: str,
    i: int,
    seed: int,
    size: int,
    steps: int,
    guidance: float,
):
    generator = torch.Generator(device="cuda").manual_seed(seed + i)
    prompt_embeds = nprow_to_tensor(embeds_mm[i:i + 1], dtype).to("cuda")
    t0 = time.time()
    try:
        img = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            height=size,
            width=size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
    except torch.cuda.OutOfMemoryError as exc:
        log.error("OOM on prompt %d (%.1fs): %s", i, time.time() - t0, exc)
        torch.cuda.empty_cache()
        raise
    return img, time.time() - t0


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch_dtype(a.dtype)
    prompts = load_prompts(a.prompts, a.limit)
    if not prompts:
        log.error("No prompts loaded from %s", a.prompts)
        return 2

    embeds_mm, tokenizer = load_or_compute_embeds(a.model, a.out, prompts, dtype)
    if embeds_mm.shape[0] != len(prompts):
        log.error("Embed cache has %d rows but %d prompts loaded", embeds_mm.shape[0], len(prompts))
        return 2

    done = set() if a.restart else find_completed(out_dir, a.prefix, len(prompts))
    manifest = out_dir / "manifest.jsonl"
    if done:
        backfill_manifest(out_dir, a.prefix, prompts, done, manifest)
        log.info("Resume: %d/%d images already present.", len(done), len(prompts))
        if len(done) == len(prompts):
            log.info("All images already generated.")
            return 0

    pipe = build_pipeline(a.model, dtype, tokenizer, a.pin_policy, a.vram_margin)

    with manifest.open("a", encoding="utf-8") as mf:
        todo = [i for i in range(len(prompts)) if i not in done]
        for i in tqdm(todo, total=len(todo), desc="generate", unit="img"):
            img, dt = generate_one(
                pipe, embeds_mm, dtype, prompts[i], i, a.seed, a.size, a.steps, a.guidance,
            )
            path = out_dir / image_name(a.prefix, i)
            img.save(path)
            mf.write(json.dumps({"index": i, "image": str(path), "prompt": prompts[i]}) + "\n")
            mf.flush()
            tqdm.write(f"  saved {path.name} ({dt:.1f}s)")
            del img
            torch.cuda.empty_cache()

    log.info("Done. %d images in %s", len(prompts), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
