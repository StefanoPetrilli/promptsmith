"""Evaluation + comparison of trained YOLO runs on the (real) test split."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import get

log = logging.getLogger(__name__)


def evaluate(cfg: dict, run_name: str) -> dict:
    """Run inference + compute mAP@0.5 on the real test split for a named run. TODO."""
    raise NotImplementedError("YOLO eval not implemented yet")


def compare(cfg: dict) -> None:
    """Evaluate baseline and augmented, then print/save a comparison table + plots.

    TODO: produce a CSV/Markdown table under data/runs/comparison.{csv,md} with
    mAP@0.5, mAP@0.5:0.95 deltas, and a bar chart image.
    """
    b = evaluate(cfg, "baseline")
    a = evaluate(cfg, "augmented")
    log.warning("eval.compare stub: got baseline=%s augmented=%s", b, a)