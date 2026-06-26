"""Vision-language / grounding labeling backend (image -> YOLO boxes).

Registered as `labeling.backend: vlm`. The actual VLM is selected via
`labeling.model.name` (any HF transformers VLM / grounding model that fits the device).
Default mode is region/phrase grounding — the backend maps its grounded regions to YOLO
boxes for `labeling.classes`.

Optional quality filter: if `labeling.quality_filter.enabled` is true, a second VLM pass
verifies each detection looks like the target concept; low-confidence detections are dropped.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..registry import labeling_backends
from .base import Detection, LabelingBackend

log = logging.getLogger(__name__)


@labeling_backends.register("vlm")
class VLMBackend(LabelingBackend):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.model_name = self.lab_cfg["model"]["name"]
        if self.model_name in ("", "blank", None):
            raise ValueError(
                "labeling.model.name is placeholder. Set a real VLM/grounding model id in "
                "config/local.yaml."
            )
        self._model = None

    @property
    def device(self) -> str:
        d = self.lab_cfg.get("device", "auto")
        return "cuda" if d in ("auto", None) else d

    def setup(self) -> None:
        """Pre-load + cache the VLM/grounding model. TODO: implement.

        Suggested: AutoModelForVision2Seq / AutoModelForCausalLM +
        AutoProcessor.from_pretrained(self.model_name, ...).to(self.device).
        """
        raise NotImplementedError("VLM labeling setup not implemented yet")

    def label(self, images: list[Path]) -> dict[Path, list[Detection]]:
        """Run grounding on each image, map regions to YOLO boxes for configured classes,
        optionally apply the quality filter, then return detections per image.

        The caller (numeri labeling-start) writes the YOLO .txt files.
        """
        raise NotImplementedError("VLM labeling not implemented yet")