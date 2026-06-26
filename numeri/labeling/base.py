"""Abstract base for labeling backends (image -> YOLO detections).

Concrete VLM/grounding backends register via `labeling_backends` and are selected from
config by name. The base defines only the contract; the COCO/YOLO class list and output
format are shared across backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    cls: int
    x: float        # normalized center x (0..1)
    y: float        # normalized center y
    w: float        # normalized width
    h: float        # normalized height
    score: float = 1.0


class LabelingBackend(ABC):
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.lab_cfg = cfg["labeling"]
        self.paths = cfg["paths"]
        self.classes = self.lab_cfg["classes"]

    @abstractmethod
    def setup(self) -> None:
        """Pre-download / cache the labeling model."""

    @abstractmethod
    def label(self, images: list[Path]) -> dict[Path, list[Detection]]:
        """Return YOLO detections per image. Modes (bbox / region grounding / VQA) decided
        by the concrete backend based on its own config kwargs."""