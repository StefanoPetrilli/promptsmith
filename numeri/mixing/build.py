"""Mix real + synthetic splits into a unified YOLO dataset.

Strategy: take all real (train) + draw synthetic at `mix.ratio_synthetic` of the combined
count. The **test split is always real-only** (config mix.test_real_only, default true) so
evaluation reflects real-world distribution and not generator artifacts. Splits are
materialized on disk under `paths.mixed` as symlinks/copies + a `data.yaml`.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..config import get

log = logging.getLogger(__name__)


def build_mix(cfg: dict) -> Path:
    """Materialize real+synthetic mix under paths.mixed. TODO: implement.

    Outline:
      - read data/real_yolo/data.yaml for train/test image sets
      - read data/synthetic/labels/* for synthetic (already YOLO format)
      - sample synthetic so that len(synthetic) / len(combined) == ratio_synthetic
      - create data/mixed/{images,labels}/{train,test} via copy or symlink
      - write data/mixed/data.yaml with class names from real_yolo
    """
    out = Path(get(cfg, "paths.mixed", "data/mixed"))
    out.mkdir(parents=True, exist_ok=True)
    log.warning("mix.build_mix is a stub — wrote nothing yet")
    return out