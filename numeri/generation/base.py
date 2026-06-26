"""Abstract base for generation backends (text -> image).

Concrete backends (e.g. diffusion) live in sibling modules and register via
`generation_backends`. The base defines the minimal contract so downstream stages
(labeling, mixing) are decoupled from the generation mechanism.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class GenerationBackend(ABC):
    """Generate images from prompts. Output paths are returned for downstream labeling."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.gen_cfg = cfg["generation"]
        self.paths = cfg["paths"]

    @abstractmethod
    def setup(self) -> None:
        """Pre-download / cache the model so `start` is offline-friendly."""

    @abstractmethod
    def generate(self, prompts: list[str], out_dir: Path, num_images: int) -> list[Path]:
        """Produce images for `prompts` under `out_dir`; return their paths in order."""