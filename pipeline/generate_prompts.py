#!/usr/bin/env python3
"""Step 1 — generate synthetic wrench prompts across labelled categories."""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.combinations import (
    ARRANGEMENTS,
    CONFUSERS,
    DENSE_ARRANGEMENTS,
    DENSE_COUNTS,
    DENSE_FINISHES,
    DENSE_WRENCH_SETS,
    DISTANCES,
    ENVIRONMENTS,
    FINISHES,
    LIGHTINGS,
    ORIENTATIONS,
    PLAIN_BACKGROUNDS,
    SURFACES,
    WRENCH_TYPES,
)

POSITIVE_MODES = ("clean_positive", "hard_positive", "dense_positive", "asset")
NEGATIVE_MODES = ("hard_negative",)
ALL_MODES = POSITIVE_MODES + NEGATIVE_MODES

_DIST_WEIGHTS = {
    "clean_positive": [3, 4, 2, 1, 1, 1],
    "hard_positive": [1, 1, 2, 3, 3, 4],
    "dense_positive": [1, 1, 2, 3, 3, 4],
    "asset": [2, 4, 2, 1, 1, 1],
}

_QUALITY_SUFFIX = "no text, no watermark, no logo, no caption, no letters, photorealistic"


def _weighted_choice(rng: random.Random, values: list[str], weights: list[float]) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _article(noun: str) -> str:
    # "a" / "an" by the first sound; bare for already-plural nouns ending in "s".
    if noun.endswith("s"):
        return ""
    return "an " if noun[:1].lower() in "aeiou" else "a "


def _join(parts: list[str]) -> str:
    return ", ".join(p for p in parts if p)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _common_axes(rng: random.Random, mode: str) -> dict:
    return {
        "scene": rng.choice(ENVIRONMENTS),
        "surface": rng.choice(SURFACES),
        "arrangement": rng.choice(ARRANGEMENTS),
        "orientation": rng.choice(ORIENTATIONS),
        "distance": _weighted_choice(rng, DISTANCES, _DIST_WEIGHTS[mode]),
        "lighting": rng.choice(LIGHTINGS),
    }


def _object_phrase(subtype: str, finish: str) -> str:
    return f"{_article(finish)}{finish} {subtype}"


def _positive_text(framing: str, subtype: str, finish: str, ax: dict) -> str:
    return _cap(_join([
        _object_phrase(subtype, finish),
        framing,
        f"{ax['arrangement']} on {ax['surface']} in {ax['scene']}",
        ax["orientation"], ax["distance"], ax["lighting"],
        _QUALITY_SUFFIX,
    ]))


def build_clean_positive(rng: random.Random) -> str:
    subtype = rng.choice(WRENCH_TYPES)
    finish = rng.choice(FINISHES)
    ax = _common_axes(rng, "clean_positive")
    return _positive_text("fully in frame, clearly separated", subtype, finish, ax)


def build_hard_positive(rng: random.Random) -> str:
    subtype = rng.choice(WRENCH_TYPES)
    finish = rng.choice(FINISHES)
    ax = _common_axes(rng, "hard_positive")
    framing = rng.choice([
        "small, near the frame edge",
        "partly occluded by another object",
        "distant in a cluttered scene",
        "partially cut off by the frame",
        "small and partially in frame",
    ])
    return _positive_text(framing, subtype, finish, ax)


def build_dense_positive(rng: random.Random) -> str:
    count = rng.choice(DENSE_COUNTS)
    finish = rng.choice(DENSE_FINISHES)
    tool_set = rng.choice(DENSE_WRENCH_SETS)
    arrangement = rng.choice(DENSE_ARRANGEMENTS)
    scene = rng.choice(ENVIRONMENTS)
    lighting = rng.choice(LIGHTINGS)
    distance = _weighted_choice(rng, DISTANCES, _DIST_WEIGHTS["dense_positive"])
    return _cap(_join([
        f"{count} {finish} {tool_set}",
        f"{arrangement} in {scene}",
        distance, lighting,
        _QUALITY_SUFFIX,
    ]))


def build_asset(rng: random.Random) -> str:
    subtype = rng.choice(WRENCH_TYPES)
    finish = rng.choice(FINISHES)
    bg = rng.choice(PLAIN_BACKGROUNDS)
    ax = _common_axes(rng, "asset")
    return _cap(_join([
        _object_phrase(subtype, finish),
        "fully in frame, clearly separated",
        f"on {bg} background",
        ax["orientation"], ax["distance"], ax["lighting"],
        f"{_QUALITY_SUFFIX}, studio product shot",
    ]))


def _confuser_phrase(confuser: str) -> str:
    return f"{_article(confuser)}{confuser}"


def build_hard_negative(rng: random.Random) -> str:
    confuser = rng.choice(CONFUSERS)
    ax = _common_axes(rng, "hard_positive")
    return _cap(_join([
        _confuser_phrase(confuser),
        f"{ax['arrangement']} on {ax['surface']} in {ax['scene']}",
        ax["distance"], ax["lighting"],
        _QUALITY_SUFFIX,
    ]))


BUILDERS = {
    "clean_positive": build_clean_positive,
    "hard_positive": build_hard_positive,
    "dense_positive": build_dense_positive,
    "asset": build_asset,
    "hard_negative": build_hard_negative,
}


def leaf_seed(base_seed: int, mode: str) -> int:
    h = hashlib.sha1(mode.encode()).hexdigest()
    return base_seed + (int(h, 16) % 1_000_000)


def generate(mode: str, n: int, seed: int) -> list[str]:
    rng = random.Random(leaf_seed(seed, mode))
    builder = BUILDERS[mode]
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < n * 20:
        p = builder(rng)
        if p not in seen:
            seen.add(p)
            out.append(p)
        tries += 1
    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)} unique {mode} prompts after {tries} tries")
    return out


def write(prompts: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / "prompts.txt"
    txt.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return txt


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic-image prompts (step 1).")
    ap.add_argument("--mode", required=True, choices=ALL_MODES)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/prompts")
    args = ap.parse_args()

    prompts = generate(args.mode, args.n, args.seed)
    txt = write(prompts, Path(args.out) / args.mode)
    print(f"wrote {len(prompts)} {args.mode} prompts -> {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())