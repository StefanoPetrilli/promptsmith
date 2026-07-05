#!/usr/bin/env python3
"""Run the wrench detector on the laptop webcam."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("manual-test")

WINDOW_NAME = "YOLO Webcam"
QUIT_KEYS = {ord("q"), ord("Q"), 27}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time YOLO webcam inference.")
    p.add_argument("--model", default="best.pt")
    p.add_argument("--source", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    return p.parse_args()


def parse_source(source: str) -> int | str:
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
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(parse_source(a.source))
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

        if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Stream closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
