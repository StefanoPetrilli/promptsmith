#!/usr/bin/env python3
"""Step 2 — render prompts into images with FLUX.2-klein-9B (fp16).

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

for _name in ("httpx", "huggingface_hub", "urllib3", "filelock", "diffusers", "transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)
try:
    from huggingface_hub import logging as hf_logging
    hf_logging.set_verbosity_warning()
except Exception:
    pass

EMBED_MEMMAP_SUFFIX = ".embeds.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX.2-klein-9B image generation (step 2).")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-9B")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--quant", choices=["none", "int8"], default="none",
                   help="int8: bitsandbytes 8-bit weight-only quantization of the transformer "
                        "(halves VRAM -> more blocks pinned; requires sm>=7.5)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--vram-margin", type=float, default=1.2)
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


def offload_hooks(module: torch.nn.Module, cuda: torch.device, cpu: torch.device):
    def pre(m, _args):
        m.to(cuda)

    def post(m, _args, output):
        m.to(cpu)
        return output

    module.register_forward_pre_hook(pre)
    module.register_forward_hook(post)


def pin_layers(layers: list[torch.nn.Module], margin_gb: float) -> tuple[int, int]:
    """Pin as many layers on GPU as fit; offload the rest via hooks."""
    cuda = torch.device("cuda")
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
        offload_hooks(layer, cuda, torch.device("cpu"))

    return n_pin, len(layers) - n_pin


def capture_hooks(layers: list[torch.nn.Module], target_indices: tuple[int, ...]):
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
    captured, handles = capture_hooks(list(inner.layers), target_layers)
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
    captured.clear()
    # Release Qwen3 fully before FLUX loads: drop the module, collect refs
    # (pinned layers, hooks, embed_tokens), then return cached VRAM to the driver.
    te.to("cpu")
    del te
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Qwen3 released; free VRAM: %.2f GB.", free_vram_gb(torch.device("cuda")))


def memmap_is_valid(mm: np.ndarray, chunk: int = 64) -> bool:
    for start in range(0, mm.shape[0], chunk):
        ok = np.any(mm[start:start + chunk], axis=(1, 2))
        if not ok.all():
            bad = (np.where(~ok)[0] + start).tolist()
            log.warning("Memmap has zero rows (first bad: %s); recomputing.", bad[:5])
            return False
    return True


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
    precompute_prompt_embeds(model_id, prompts, dtype, cache, tokenizer)
    return np.lib.format.open_memmap(cache, mode="r"), tokenizer


def apply_partial_pin(pipe, vram_margin_gb: float = 1.2):
    cuda = torch.device("cuda")
    transformer = pipe.transformer

    pipe.vae.to(cuda)

    small_modules = [
        "pos_embed", "time_guidance_embed",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation", "x_embedder", "context_embedder",
        "norm_out", "proj_out",
    ]
    for name in small_modules:
        getattr(transformer, name).to(cuda)

    free_gb = free_vram_gb(cuda)
    margin_b = vram_margin_gb * 1e9

    def pin_group(blocks: list[torch.nn.Module], name: str) -> int:
        """Greedily pin blocks on GPU while the margin holds; offload the rest via hooks.

        Measured by actually moving each block (parameter introspection misreports
        sizes for quantized tensor subclasses, e.g. torchao int8)."""
        n = 0
        for block in blocks:
            block.to(cuda)
            if torch.cuda.mem_get_info(cuda)[0] < margin_b:
                block.to("cpu")
                torch.cuda.empty_cache()
                break
            n += 1
        for block in blocks[n:]:
            offload_hooks(block, cuda, torch.device("cpu"))
        log.info("partial-pin: %s -> pinned %d/%d, offloaded %d.", name, n, len(blocks), len(blocks) - n)
        return n

    blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
    pin_group(blocks, "transformer blocks")
    log.info("partial-pin: free=%.2f GB, margin=%.1f GB.", free_gb, vram_margin_gb)

    torch.cuda.empty_cache()


def build_pipeline(model_id: str, dtype: torch.dtype, tokenizer, vram_margin: float, quant: str = "none"):
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
    tf_kwargs: dict = {"torch_dtype": dtype}
    if quant == "int8":
        from diffusers import BitsAndBytesConfig
        tf_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True,
        )
        gpu_budget = max(1.0, free_vram_gb(torch.device("cuda")) - vram_margin)
        tf_kwargs["device_map"] = "auto"
        tf_kwargs["max_memory"] = {0: f"{gpu_budget:.1f}GiB", "cpu": "12GiB"}
        log.info("Quantizing transformer to int8 (bitsandbytes); GPU budget %.1f GiB.", gpu_budget)
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", **tf_kwargs,
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
    if quant == "int8":
        # accelerate already placed/dispatched the quantized transformer via device_map.
        pipe.vae.to(torch.device("cuda"))
    else:
        apply_partial_pin(pipe, vram_margin_gb=vram_margin)
    pipe.set_progress_bar_config(disable=True)
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
    i: int,
    seed: int,
    size: int,
    steps: int,
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
            guidance_scale=1.0,
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

    done = find_completed(out_dir, a.prefix, len(prompts))
    manifest = out_dir / "manifest.jsonl"
    if done:
        backfill_manifest(out_dir, a.prefix, prompts, done, manifest)
        log.info("Resume: %d/%d images already present.", len(done), len(prompts))
        if len(done) == len(prompts):
            log.info("All images already generated.")
            return 0

    # Defensive: nothing but FLUX should hold VRAM from here on.
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Free VRAM before loading FLUX: %.2f GB.", free_vram_gb(torch.device("cuda")))

    pipe = build_pipeline(a.model, dtype, tokenizer, a.vram_margin, a.quant)

    with manifest.open("a", encoding="utf-8") as mf:
        todo = [i for i in range(len(prompts)) if i not in done]
        for i in tqdm(todo, total=len(todo), desc="generate", unit="img"):
            img, dt = generate_one(
                pipe, embeds_mm, dtype, i, a.seed, a.size, a.steps,
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
