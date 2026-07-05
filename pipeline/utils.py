"""Small helpers shared across pipeline scripts."""
from __future__ import annotations

import colorsys


def palette(n: int) -> list[tuple[int, int, int]]:
    """Distinct RGB colors spread around the HSV wheel."""
    n = max(n, 1)
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i / n) % 1.0, 0.85, 0.95))
        for i in range(n)
    ]
