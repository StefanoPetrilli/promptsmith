"""IO helpers shared across stages."""
from __future__ import annotations

from pathlib import Path

from ..labeling.base import Detection


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_yolo_labels(label_path: Path, detections: list[Detection]) -> None:
    """Write one YOLO .txt file: `cls cx cy w h` per line, normalized 0..1.

    Values are clipped to [0, 1]; classes are written as 0-indexed ints.
    """
    ensure_dir(label_path.parent)
    lines = []
    for d in detections:
        cls = int(d.cls)
        cx = _clip(d.x)
        cy = _clip(d.y)
        w = _clip(d.w)
        h = _clip(d.h)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _clip(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def make_data_yaml(
    path: Path,
    train: str,
    test: str,
    classes: list[str],
    val: str | None = None,
) -> None:
    """Write a YOLO `data.yaml`. `val` defaults to `train` if omitted."""
    ensure_dir(path.parent)
    txt = [
        f"train: {train}",
        f"val: {val or train}",
        f"test: {test}",
        f"nc: {len(classes)}",
        f"names: {list(classes)}",
    ]
    path.write_text("\n".join(txt) + "\n")