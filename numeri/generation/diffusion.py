"""Diffusion-based generation backend (text -> image) using `diffusers`.

Registered as `generation.backend: diffusion`. Model name comes from config
(`generation.model.name`), so any HF diffusers pipeline can be swapped in without code
changes. Designed to run small / quantized models on ~6 GB VRAM.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..registry import generation_backends
from .base import GenerationBackend

log = logging.getLogger(__name__)


@generation_backends.register("diffusion")
class DiffusionBackend(GenerationBackend):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.model_name = self.gen_cfg["model"]["name"]
        if self.model_name in ("", "blank", None):
            raise ValueError(
                "generation.model.name is placeholder. Set a real diffusion model id in "
                "config/local.yaml (e.g. stabilityai/sd-turbo)."
            )
        self._pipe = None  # lazy load

    @property
    def device(self) -> str:
        d = self.gen_cfg.get("device", "auto")
        return "cuda" if d in ("auto", None) else d

    @property
    def dtype(self) -> str:
        return self.gen_cfg.get("dtype", "fp16")

    def setup(self) -> None:
        """Pre-load the diffusion pipeline to populate the local model cache. TODO: implement.

        Suggested:
          from diffusers import AutoPipelineForText2Image
          AutoPipelineForText2Image.from_pretrained(self.model_name,
              torch_dtype=..., variant="fp16").to(self.device)
        """
        raise NotImplementedError("Diffusion setup not implemented yet")

    def generate(
        self, prompts: list[str], out_dir: Path, num_images: int
    ) -> list[Path]:
        """Run text2image inference, save PNGs. TODO: implement."""
        raise NotImplementedError("Diffusion generation not implemented yet")