"""SVHN dataset adapter (full-number, bounding-box version) → YOLO format.

Registered as `dataset.name: svhn` in config. Functions are called by the dataset CLI
actions, not imported directly, so a different use case just supplies its own adapter
module and sets `dataset.name`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..registry import dataset_adapters

log = logging.getLogger(__name__)


@dataset_adapters.register("svhn")
class SVHNAdapter:
    """Download + parse SVHN (full numbers) into YOLO layout.

    Layout produced under `real_yolo/`:
        images/train/*.png, images/test/*.png
        labels/train/*.txt, labels/test/*.txt   (YOLO x y w h, normalized)
        data.yaml
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.opts = cfg["dataset"]

    # -- public API (called by numeri dataset-* CLI actions) -------------------
    def download(self) -> None:
        """Fetch raw SVHN archives into paths.raw. TODO: implement (urllib + tar extract)."""
        raise NotImplementedError("SVHN download not implemented yet")

    def parse(self) -> None:
        """Parse downloaded SVHN into YOLO format under paths.real_yolo. TODO: implement.

        SVHN provides `digitStruct.mat` per split. Conversion outline:
          - read digitStruct (per-image bbox + per-digit labels)
          - write one <image>.txt with one line per digit:
              cls cx cy w h  (normalized 0..1)
          - emit data.yaml with train/val/test paths + class names.
        """
        raise NotImplementedError("SVHN → YOLO conversion not implemented yet")