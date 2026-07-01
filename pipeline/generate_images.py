#!/usr/bin/env python3
"""Step 2 — render prompts into images with FLUX.2-klein-4B (fp16).

Single transformer backend:
  * fp16 (`--model black-forest-labs/FLUX.2-klein-4B`): unquantized fp16 weights loaded from the
    diffusers `transformer/` subfolder. Lossless. Combined with `--pin-policy partial` this pins
    most blocks on the GPU and offloads the rest per-block (~3x less PCIe traffic than sequential
    offload).

(The FLUX.2-klein-9B GGUF path was removed: on a 6 GB GPU the Q4_K_M quantization produced
visible artifacts, the unquantized 9B doesn't fit, and NF4 text-encoder loading OOMs. The 4B
fp16 path is lossless and produces cleaner images on this hardware.)

Qwen3 text encoder (~8 GB in bf16): embeddings are precomputed + cached to disk, then released
before the transformer loads (the two never coexist on the GPU). Loaded in **bfloat16**, the
checkpoint's native dtype (`text_encoder/config.json` -> `"dtype": "bfloat16"`), to avoid
from_pretrained casting bf16->fp16 — that cast temporarily holds BOTH the mmap'd bf16 pages and
the new fp16 copy (~2x peak). Final embeddings are cast to `--dtype` in _embed_one, so loading in
bf16 does not change the cached output. Hybrid GPU+CPU fp16 (lossless): ~4 GB on GPU, ~4 GB on
CPU.

Two offload policies (`--pin-policy`):
  * `partial` (default): pin the VAE + all small transformer modules + all double-stream
    blocks + as many single-stream blocks as fit in VRAM on the GPU permanently; offload the
    remaining single blocks one-at-a-time via pre/post-forward hooks (cuda -> compute -> cpu).
    The latent tensor stays on cuda the whole time, so only the offloaded block weights shuttle
    across PCIe. Lossless.
  * `sequential`: the old `enable_sequential_cpu_offload` — one sub-module on the GPU at a
    time, slow, fits a 6 GB card. Kept as a fallback.

Embeddings are cached as a numpy memmap (`.embeds.npy`, shape (N, L, D)) holding the raw
bit-pattern of the fp16/bf16 tensors. During generation the file is opened read-only and only
row `i` is paged into RAM per image (~6 MB resident vs ~8 GB if the whole list were held in
RAM). This keeps the generation loop out of swap — critical because the transformer already
occupies ~8 GB of the machine's RAM. A legacy `.pt` cache (list of per-prompt tensors) from
older runs is auto-converted to the memmap format on first use, no recompute needed.

Resume: on startup the output dir is scanned for existing `img_{i:04d}.png` files; those
indices are skipped and their manifest entries backfilled if missing. Generation continues
from the first missing image. Each image uses an independent generator seeded `seed + i` so a
resumed run produces the same noise per image as a fresh run (not a shared RNG stream).

Inputs: one prompt per line (`#`/blank lines skipped). Outputs: `img_{i:04d}.png` plus a
`manifest.jsonl` (prompt -> image, used by step 3). `--limit N` renders only the first N.
"""
from __future__ import annotations

import os
# Set before `import torch` / any CUDA context initialization so the caching allocator uses
# expandable segments — eliminates fragmentation and reclaims reserved-but-unallocated VRAM.
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

EMBED_MEMMAP_SUFFIX = ".embeds.npy"  # numpy memmap, shape (N, L, D), raw bit-pattern


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX.2-klein-4B image generation (step 2).")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B",
                   help="HF repo for transformer/VAE/scheduler/tokenizer/text_encoder.")
    p.add_argument("--prompts", required=True, help="one prompt per line (output of step 1)")
    p.add_argument("--out", required=True, help="output directory for PNGs + manifest")
    p.add_argument("--size", type=int, default=768, help="square image edge (multiple of 16)")
    p.add_argument("--steps", type=int, default=8, help="distilled model -> few steps OK")
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--limit", type=int, default=0, help="0 = all prompts; else first N (test)")
    p.add_argument(
        "--pin-policy", choices=["partial", "sequential"], default="partial",
        help="partial (default): pin VAE + small modules + double blocks + as many single blocks "
             "as fit in VRAM, offload the rest per-block. sequential: old one-module-at-a-time offload.",
    )
    p.add_argument(
        "--vram-margin", type=float, default=1.2,
        help="GB of VRAM to reserve for activations/display when auto-pinning single blocks (partial).",
    )
    p.add_argument(
        "--restart", action="store_true",
        help="ignore existing images in --out; regenerate everything (manifest is appended).",
    )
    return p.parse_args()


def load_prompts(path: str, limit: int) -> list[str]:
    lines = [l.strip() for l in Path(path).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    return lines[:limit] if limit > 0 else lines


def embed_cache_key(model: str, prompts: list[str]) -> str:
    return hashlib.sha1(
        json.dumps({"m": model, "n": len(prompts), "p": prompts}).encode()
    ).hexdigest()[:16]


def embed_cache_path(out: str, model: str, prompts: list[str]) -> Path:
    return Path(out) / f"_embeds_{embed_cache_key(model, prompts)}{EMBED_MEMMAP_SUFFIX}"


def embed_legacy_cache_path(out: str, model: str, prompts: list[str]) -> Path:
    return Path(out) / f"_embeds_{embed_cache_key(model, prompts)}.pt"


def _np_store_dtype(dtype: torch.dtype) -> np.dtype:
    """Numpy dtype used to store the raw bit-pattern of `dtype` tensors in the memmap file."""
    if dtype == torch.float32:
        return np.float32
    return np.uint16  # 2-byte raw bit pattern for fp16 / bf16 (numpy has no bfloat16)


def _embed_to_nprow(e: torch.Tensor, store_np: np.dtype, dtype: torch.dtype):
    """(L, D) torch tensor in `dtype` -> numpy array of `store_np` sharing the exact bit pattern."""
    if dtype == torch.bfloat16:
        return e.view(torch.uint16).numpy()  # numpy has no bfloat16 -> store raw uint16 bits
    if dtype == torch.float16:
        return e.numpy().view(np.uint16)  # reinterpret fp16 bytes as uint16
    return e.numpy()  # float32


def _row_to_torch(np_row, dtype: torch.dtype) -> torch.Tensor:
    """Read a (1, L, D) numpy view from the memmap -> torch tensor in `dtype` (shares the page)."""
    t = torch.from_numpy(np_row)  # uint16 or float32; shares the memmap page (lazy page-in)
    if t.dtype == torch.uint16:
        t = t.view(dtype)  # reinterpret bits -> fp16 / bf16
    return t


def precompute_prompt_embeds(
    model_id: str, prompts: list[str], dtype: torch.dtype, cache_path: Path, tokenizer,
):
    """Load Qwen3 (bfloat16, native ckpt dtype), stream FLUX.2 prompt embeddings row-by-row into
    a memmap file, then release Qwen3. Only one embedding (~6 MB) + Qwen3 are resident at a time;
    the full embedding set is never held in RAM.

    Hybrid GPU+CPU bf16 (~8 GB TE): ~4 GB on GPU, ~4 GB on CPU. Lossless. If no CUDA is available,
    loads fully on CPU. Final embeddings are cast to `dtype` in _embed_one, so loading the model in
    bf16 does not change the cached output (same rounding happens, just once at the end instead of
    during from_pretrained).
    """
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen3ForCausalLM

    if torch.cuda.is_available():
        log.info("Loading Qwen3 text encoder hybrid GPU+CPU (bfloat16, ~8 GB, native ckpt dtype — no cast peak)...")
        te = Qwen3ForCausalLM.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="balanced",
            max_memory={0: "4GiB", "cpu": "8GiB"},
        ).eval()
        enc_device = torch.device("cuda")
    else:
        log.info("Loading Qwen3 text encoder on CPU (bfloat16)...")
        te = Qwen3ForCausalLM.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to("cpu").eval()
        enc_device = torch.device("cpu")
    log.info("Qwen3 loaded. Streaming embeddings for %d prompts -> %s ...", len(prompts), cache_path.name)

    store_np = _np_store_dtype(dtype)

    def _embed_one(prompt: str) -> torch.Tensor:
        e = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=te,
            tokenizer=tokenizer,
            prompt=prompt,
            device=enc_device,
            dtype=dtype,
            max_sequence_length=512,
            hidden_states_layers=(9, 18, 27),
        )  # (1, L, D)
        return e[0].detach().to("cpu", dtype=dtype)  # (L, D)

    # Compute the first prompt to learn (L, D), then create the memmap file, then stream the rest.
    first = _embed_one(prompts[0])
    L, D = first.shape
    N = len(prompts)
    mm = np.lib.format.open_memmap(cache_path, mode="w+", dtype=store_np, shape=(N, L, D))
    mm[0] = _embed_to_nprow(first, store_np, dtype)
    del first
    for i in range(1, N):
        e = _embed_one(prompts[i])
        mm[i] = _embed_to_nprow(e, store_np, dtype)
        del e
        if i % 50 == 0:
            mm.flush()
    mm.flush()
    del mm

    log.info("Releasing Qwen3 from RAM...")
    del te
    gc.collect()


def load_or_compute_embeds(model_id: str, out: str, prompts: list[str], dtype: torch.dtype):
    """Return (numpy memmap (N,L,D) read-only, tokenizer). Builds the memmap from Qwen3 if
    absent, or converts a legacy `.pt` cache if present. Never holds the full embedding set in
    RAM after this returns — the caller indexes row i, paging in only that ~6 MB slice."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    cache = embed_cache_path(out, model_id, prompts)

    if cache.exists():
        log.info("Found memmap embeddings: %s -> opening read-only (lazy paging)", cache)
        return np.lib.format.open_memmap(cache, mode="r"), tok

    cache.parent.mkdir(parents=True, exist_ok=True)

    legacy = embed_legacy_cache_path(out, model_id, prompts)
    if legacy.exists():
        log.info("Converting legacy .pt cache %s -> memmap %s (no Qwen3 recompute)...", legacy.name, cache.name)
        data = torch.load(legacy, map_location="cpu", mmap=True, weights_only=False)
        embeds_list = data["embeds"]
        N = len(embeds_list)
        first = embeds_list[0]  # (1, L, D)
        L, D = first.shape[1], first.shape[2]
        store_np = _np_store_dtype(dtype)
        mm = np.lib.format.open_memmap(cache, mode="w+", dtype=store_np, shape=(N, L, D))
        for i, e in enumerate(tqdm(embeds_list, desc="convert embeds", unit="row")):
            mm[i] = _embed_to_nprow(e[0].to(dtype), store_np, dtype)
        mm.flush()
        del mm, data, embeds_list
        gc.collect()
        return np.lib.format.open_memmap(cache, mode="r"), tok

    precompute_prompt_embeds(model_id, prompts, dtype, cache, tok)
    return np.lib.format.open_memmap(cache, mode="r"), tok


def _module_size_mb(m: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6


def apply_partial_pin(pipe, vram_margin_gb: float = 1.2):
    """Pin VAE + small transformer modules + all double-stream blocks + as many single-stream
    blocks as fit in VRAM on cuda permanently; offload the remaining single blocks one-at-a-time
    via pre/post-forward hooks (cuda -> compute -> cpu).

    The latent tensor is created on cuda by the pipeline (via our overridden _execution_device)
    and only dtype-cast during the denoising loop, so it stays on cuda throughout. Pinned blocks
    run with zero PCIe transfer. Each offloaded block is moved to cuda in its pre-hook, computes
    on the cuda latent, and is moved back to cpu in its post-hook — so at most one offloaded
    block (~245 MB) is on the GPU at any instant. Lossless: identical fp16 weights, just placed
    to minimize transfers. Do NOT also call enable_sequential_cpu_offload (it would re-hook every
    submodule and yank pinned blocks back to cpu)."""
    cuda = torch.device("cuda")
    transformer = pipe.transformer

    # 1. VAE on cuda (small, ~200 MB; used for encode/decode, must match latent device).
    pipe.vae.to(cuda)

    # 2. Small transformer modules on cuda (embedders, modulations, norm/proj — ~273 MB total).
    small_names = [
        "pos_embed", "time_guidance_embed",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation", "x_embedder", "context_embedder",
        "norm_out", "proj_out",
    ]
    pinned_mb = _module_size_mb(pipe.vae)
    for name in small_names:
        mod = getattr(transformer, name)
        mod.to(cuda)
        pinned_mb += _module_size_mb(mod)

    # 3. All double-stream blocks on cuda (5 x ~491 MB = ~2454 MB).
    for blk in transformer.transformer_blocks:
        blk.to(cuda)
        pinned_mb += _module_size_mb(blk)

    # 4. Pin as many single-stream blocks as fit; offload the rest.
    single_blocks = transformer.single_transformer_blocks
    n_single = len(single_blocks)
    single_mb = _module_size_mb(single_blocks[0]) if n_single else 0.0
    free_b, _ = torch.cuda.mem_get_info(cuda)
    free_gb = free_b / 1e9
    budget_gb = free_gb - vram_margin_gb
    n_pin = max(0, min(n_single, int(budget_gb * 1e3 / single_mb))) if single_mb else n_single
    n_off = n_single - n_pin
    log.info(
        "partial-pin: VRAM free=%.2f GB, margin=%.1f GB -> pin %d/%d single blocks (%.0f MB), "
        "offload %d (~%.0f MB PCIe/step each). Total pinned ~= %.2f GB.",
        free_gb, vram_margin_gb, n_pin, n_single, n_pin * single_mb, n_off, single_mb, pinned_mb / 1e3,
    )

    offloaded = []
    for i, blk in enumerate(single_blocks):
        if i < n_pin:
            blk.to(cuda)
            pinned_mb += _module_size_mb(blk)
        else:
            offloaded.append(blk)

    # 5. Hook each offloaded block: move to cuda before forward, back to cpu after.
    #    Inputs (latent) are already on cuda; we only move the block's own weights. The output
    #    tensor stays on cuda (post-hook returns it unchanged; only the module moves to cpu).
    for blk in offloaded:
        def _pre(module, _args, _cuda=cuda):
            module.to(_cuda)
            return None

        def _post(module, _args, output, _cuda=cuda):
            module.to("cpu")
            return output

        blk.register_forward_pre_hook(_pre)
        blk.register_forward_hook(_post)

    torch.cuda.empty_cache()
    return n_pin, n_off


def build_pipeline(
    model_id: str, dtype: torch.dtype, tokenizer, pin_policy: str, vram_margin: float,
):
    """Flux2KleinPipeline with an fp16 transformer (from --model's `transformer/` subfolder).
    Lossless.

    pin_policy='partial'    -> apply_partial_pin (default; ~3x less PCIe traffic than sequential).
    pin_policy='sequential' -> enable_sequential_cpu_offload (one module on GPU at a time, slow)."""
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

    # Dummy stand-in for the text encoder (Qwen3 replaced by precomputed embeds). Must expose
    # `.dtype` so pipeline-level `.to()` / enable_sequential_cpu_offload (which reads
    # module.dtype on every submodule) doesn't raise AttributeError like a plain nn.Linear would.
    class _DummyTextEncoder(torch.nn.Module):
        dtype = torch.float32  # only used for the dtype guard in pipeline.to(); no real params
        def forward(self, *args, **kwargs):
            return None

    dummy_te = _DummyTextEncoder()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    log.info("Loading Flux2Transformer2DModel fp16 (unquantized, ~8 GB RAM)...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype,
    )
    vae = AutoencoderKLFlux2.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    )

    pipe = Flux2KleinPipelineFixed(
        scheduler=scheduler, vae=vae, text_encoder=dummy_te,
        tokenizer=tokenizer, transformer=transformer, is_distilled=True,
    )

    if pin_policy == "partial":
        apply_partial_pin(pipe, vram_margin_gb=vram_margin)
    else:
        log.info("enable_sequential_cpu_offload (one layer on GPU at a time, slow)...")
        pipe.enable_sequential_cpu_offload(device=torch.device("cuda"))
    torch.cuda.empty_cache()

    return pipe


def find_completed(out_dir: Path, n: int) -> set[int]:
    """Return indices i in [0, n) whose `img_{i:04d}.png` exists in out_dir."""
    done = set()
    for i in range(n):
        if (out_dir / f"img_{i:04d}.png").exists():
            done.add(i)
    return done


def backfill_manifest(out_dir: Path, prompts: list[str], done: set[int], manifest: Path):
    """Ensure every done image has a manifest entry (append any that are missing). Returns the
    set of indices that already had manifest entries."""
    have = set()
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    have.add(int(rec["index"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue  # skip corrupt/partial lines
    missing = sorted(done - have)
    if missing:
        with manifest.open("a", encoding="utf-8") as mf:
            for i in missing:
                path = out_dir / f"img_{i:04d}.png"
                mf.write(json.dumps({"index": i, "image": str(path), "prompt": prompts[i]}) + "\n")
        log.info("Backfilled %d missing manifest entries.", len(missing))
    return have


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

    embeds_mm, tok = load_or_compute_embeds(a.model, a.out, prompts, dtype)
    if embeds_mm.shape[0] != len(prompts):
        log.error("embed cache has %d rows but %d prompts were loaded", embeds_mm.shape[0], len(prompts))
        return 2
    log.info("Embeddings memmap: shape=%s dtype=%s (read-only, paged per image)", embeds_mm.shape, embeds_mm.dtype)

    # --- resume: skip images whose PNG already exists ---
    done = set() if a.restart else find_completed(out_dir, len(prompts))
    manifest = out_dir / "manifest.jsonl"
    if done:
        backfill_manifest(out_dir, prompts, done, manifest)
        log.info("Resume: %d/%d images already present -> skipping them.", len(done), len(prompts))
        if len(done) == len(prompts):
            log.info("All %d images already generated. Nothing to do.", len(prompts))
            return 0
    else:
        log.info("Starting from scratch (no existing images in %s).", out_dir)

    pipe = build_pipeline(
        a.model, dtype, tok,
        pin_policy=a.pin_policy, vram_margin=a.vram_margin,
    )

    with manifest.open("a", encoding="utf-8") as mf:
        todo = [i for i in range(len(prompts)) if i not in done]
        for i in tqdm(todo, total=len(todo), desc="generate", unit="img"):
            prompt = prompts[i]
            # Independent per-image generator (seed + i) so a resumed run reproduces the same
            # noise per image as a fresh run — resume is bit-identical to never having stopped.
            gen = torch.Generator(device="cuda").manual_seed(a.seed + i)
            # Index row i of the memmap -> only that ~6 MB page is faulted into RAM, then
            # copied to cuda. The other rows stay on disk; pinned transformer blocks stay on
            # the GPU; only the offloaded single blocks shuttle across PCIe.
            pe_cuda = _row_to_torch(embeds_mm[i:i + 1], dtype).to("cuda")
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
            dt = time.time() - t0
            tqdm.write(f"  saved {path.name} ({dt:.1f}s)")
            del pe_cuda, gen, img
            torch.cuda.empty_cache()

    log.info("Done. %d images in %s (manifest: %s)", len(prompts), out_dir, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
