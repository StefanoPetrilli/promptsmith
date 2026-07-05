#!/usr/bin/env python3
"""Run the wrench detector on the laptop webcam.

Usage:
  python manual_test.py
  python manual_test.py --model best.pt --source 0 --conf 0.25

Press 'q' or ESC to quit.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("manual-test")

WINDOW_NAME = "YOLO Webcam"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time YOLO webcam inference.")
    p.add_argument("--model", default="best.pt", help="path to .pt weights")
    p.add_argument("--source", default="0", help="webcam index or video file path")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference size")
    p.add_argument("--device", default="0", help="cuda device or 'cpu'")
    return p.parse_args()


def _source_to_int(source: str) -> int | str:
    try:
        return int(source)
    except ValueError:
        return source


def main() -> int:
    a = parse_args()

    model_path = Path(a.model)
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        return 2

    from ultralytics import YOLO

    log.info("Loading model %s", model_path)
    model = YOLO(str(model_path))

    source = _source_to_int(a.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", a.source)
        return 2

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    log.info("Streaming... Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to grab frame")
            break

        results = model.predict(
            source=frame,
            conf=a.conf,
            imgsz=a.imgsz,
            device=a.device,
            verbose=False,
        )[0]

        annotated = results.plot(line_width=2, font_size=0.5)
        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), ord("Q"), 27}:
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Stream closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
