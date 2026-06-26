"""YOLO fine-tuning wrapper (ultralytics).

`train_baseline` uses data/real_yolo/data.yaml; `train_run` uses data/mixed/data.yaml.
Both share yolo.* hyperparameters from config. Results land under data/runs/<name>.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import get

log = logging.getLogger(__name__)


def train(
    cfg: dict, data_yaml: Path, name: str
) -> None:
    """Fine-tune YOLO from `yolo.version` weights on `data_yaml`. TODO: implement.

    Suggested:
      from ultralytics import YOLO
      YOLO(get(cfg, "yolo.version")).train(
          data=str(data_yaml), epochs=..., imgsz=..., batch=..., device=..., name=name)
    """
    raise NotImplementedError("YOLO training not implemented yet")


def train_baseline(cfg: dict) -> None:
    """Train on real-only data."""
    real_yaml = Path(get(cfg, "paths.real_yolo", "data/real_yolo")) / "data.yaml"
    train(cfg, real_yaml, "baseline")


def train_run(cfg: dict) -> None:
    """Train on real + synthetic mix."""
    mixed_yaml = Path(get(cfg, "paths.mixed", "data/mixed")) / "data.yaml"
    train(cfg, mixed_yaml, "augmented")