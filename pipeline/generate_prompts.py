#!/usr/bin/env python3
"""Step 1 — build prompts for synthetic wrench photos across labelled categories.

A deterministic weighted-axis generator. Each invocation targets one *mode* (and, for the
positive modes, one wrench *count*) and writes its own leaf folder:

    data/prompts/<mode>[/<count>]/prompts.{jsonl,txt}

Modes (each with its own axis-weight bias + mode-aware negatives):
  * clean_positive  — single/few isolated, fully-framed wrenches on a simple background.
  * hard_positive   — small / distant / partially occluded / frame-edge / cluttered wrench.
  * hard_negative   — confusers (pliers, breaker bars, tire irons, pry bars, screwdrivers),
                      no wrench; negatives suppress `--no wrench`.
  * pure_negative   — realistic workshop scenes, no wrench, no confusers.
  * asset           — isolated wrench on plain / white / transparent background.

Positive leaves are further split by wrench count (1/, 2/, n/); negative leaves are count-less.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# All combination lists (wrench subtypes, finishes, scenes, surfaces, ...) and the
# mode-aware negative strings live in a sibling module so this file stays focused on
# rendering logic.
# This file is invoked both as a script (`python pipeline/generate_prompts.py`) and as a
# module; support both by ensuring the repo root is importable.
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.combinations import (
    ARRANGEMENTS,
    CONFUSERS,
    COUNT_WORDS,
    DISTANCES,
    ENVIRONMENTS,
    FINISHES,
    LIGHTINGS,
    NEG_ASSET,
    NEG_CLEAN,
    NEG_HARD_NEG,
    NEG_HARD_POS,
    NEG_PURE_NEG,
    ORIENTATIONS,
    PLAIN_BACKGROUNDS,
    SHOP_ITEMS,
    SURFACES,
    WRENCH_TYPES,
)

POSITIVE_MODES = ("clean_positive", "hard_positive", "asset")
NEGATIVE_MODES = ("hard_negative", "pure_negative")
ALL_MODES = POSITIVE_MODES + NEGATIVE_MODES


def _weighted_choice(rng: random.Random, values: list[str], weights: list[float]) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _plural(noun: str) -> str:
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def _article(noun: str) -> str:
    return "" if noun.endswith("s") else "a "


def _join(parts: list[str]) -> str:
    return ", ".join(p for p in parts if p)


def _cap(text: str) -> str:
    """Capitalize only the first character (leaves the rest, e.g. brand names, untouched)."""
    return text[:1].upper() + text[1:] if text else text

_DIST_WEIGHTS = {
    "clean_positive": [3, 4, 2, 1, 1, 1],
    "hard_positive":  [1, 1, 2, 3, 3, 4],
    "asset":          [2, 4, 2, 1, 1, 1],
}


def _common_axes(rng: random.Random, mode: str) -> dict:
    scene = rng.choice(ENVIRONMENTS)
    surface = rng.choice(SURFACES)
    arrangement = rng.choice(ARRANGEMENTS)
    orientation = rng.choice(ORIENTATIONS)
    distance = _weighted_choice(rng, DISTANCES, _DIST_WEIGHTS[mode])
    lighting = rng.choice(LIGHTINGS)
    return {
        "scene": scene, "surface": surface, "arrangement": arrangement,
        "orientation": orientation, "distance": distance, "lighting": lighting,
    }


def _object_phrase(count: int, subtype: str, finish: str) -> str:
    """`a chrome combination wrench` / `two chrome combination wrenches`."""
    noun = subtype if count == 1 else _plural(subtype)
    count_word = COUNT_WORDS.get(count, str(count))
    return f"{count_word} {finish} {noun}"


def _positive_text(count: int, framing: str, subtype: str, finish: str,
                   ax: dict, neg: str) -> str:
    parts = [
        _object_phrase(count, subtype, finish),
        framing,
        f"{ax['arrangement']} on {ax['surface']} in {ax['scene']}",
        ax["orientation"], ax["distance"], ax["lighting"],
        "photorealistic",
        neg,
    ]
    return _cap(_join(parts))


def build_clean_positive(rng: random.Random, count: int) -> dict:
    subtype = rng.choice(WRENCH_TYPES)
    finish = rng.choice(FINISHES)
    ax = _common_axes(rng, "clean_positive")
    framing = "fully in frame, clearly separated" if count > 1 else "fully in frame"
    text = _positive_text(count, framing, subtype, finish, ax, NEG_CLEAN)
    return {"text": text, "mode": "clean_positive", "count": count,
            "subtype": subtype, "finish": finish, **ax}


def build_hard_positive(rng: random.Random, count: int) -> dict:
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
    text = _positive_text(count, framing, subtype, finish, ax, NEG_HARD_POS)
    return {"text": text, "mode": "hard_positive", "count": count,
            "subtype": subtype, "finish": finish,
            "framing": framing, **ax}


def build_asset(rng: random.Random, count: int) -> dict:
    subtype = rng.choice(WRENCH_TYPES)
    finish = rng.choice(FINISHES)
    bg = rng.choice(PLAIN_BACKGROUNDS)
    ax = _common_axes(rng, "asset")
    framing = "fully in frame, clearly separated" if count > 1 else "fully in frame"
    parts = [
        _object_phrase(count, subtype, finish),
        framing, f"on {bg} background",
        ax["orientation"], ax["distance"], ax["lighting"],
        "photorealistic, studio product shot", NEG_ASSET,
    ]
    return {"text": _cap(_join(parts)), "mode": "asset", "count": count,
            "subtype": subtype, "finish": finish, "background": bg, **ax}


def build_hard_negative(rng: random.Random) -> dict:
    count = rng.choice([1, 1, 1, 2, 2, 3])
    confuser = rng.choice(CONFUSERS)
    ax = _common_axes(rng, "hard_positive")  # reuse a cluttered regime
    if count == 1:
        obj = f"{_article(confuser)}{confuser}"  # "a breaker bar" / "pliers"
    else:
        obj = f"{COUNT_WORDS[count]} {_plural(confuser)}"
    parts = [
        obj,
        f"{ax['arrangement']} on {ax['surface']} in {ax['scene']}",
        ax["distance"], ax["lighting"],
        "photorealistic", NEG_HARD_NEG,
    ]
    return {"text": _cap(_join(parts)), "mode": "hard_negative", "count": 0,
            "confuser": confuser, "confuser_count": count, **ax}


def build_pure_negative(rng: random.Random) -> dict:
    item = rng.choice(SHOP_ITEMS)
    ax = _common_axes(rng, "clean_positive")
    parts = [
        item, f"in {ax['scene']}", f"on {ax['surface']}",
        ax["distance"], ax["lighting"],
        "photorealistic", NEG_PURE_NEG,
    ]
    return {"text": _cap(_join(parts)), "mode": "pure_negative", "count": 0,
            "shop_item": item, **ax}


BUILDERS = {
    "clean_positive": lambda rng, c: build_clean_positive(rng, c),
    "hard_positive": lambda rng, c: build_hard_positive(rng, c),
    "asset": lambda rng, c: build_asset(rng, c),
    "hard_negative": lambda rng, _c: build_hard_negative(rng),
    "pure_negative": lambda rng, _c: build_pure_negative(rng),
}


def leaf_seed(base_seed: int, mode: str, count: int) -> int:
    """Stable per-leaf seed so each (mode,count) invocation is independently reproducible
    and does not overlap another leaf's RNG stream."""
    h = hashlib.sha1(f"{mode}:{count}".encode()).hexdigest()
    return base_seed + (int(h, 16) % 1_000_000)


def leaf_dir(out: Path, mode: str, count: int) -> Path:
    d = out / mode
    if mode in POSITIVE_MODES:
        d = d / str(count)
    return d


def generate(mode: str, count: int, n: int, seed: int) -> list[dict]:
    builder = BUILDERS[mode]
    rng = random.Random(leaf_seed(seed, mode, count))
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < n * 20:
        p = builder(rng, count)
        key = p["text"]
        if key not in seen:
            seen.add(key)
            out.append(p)
        tries += 1
    if len(out) < n:
        raise RuntimeError(
            f"could only generate {len(out)} unique {mode}/{count} prompts after {tries} tries"
        )
    return out


def write(prompts: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "prompts.jsonl"
    txt = out_dir / "prompts.txt"
    with jsonl.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with txt.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p["text"] + "\n")
    return jsonl, txt


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic-image prompts (step 1).")
    ap.add_argument("--mode", required=True, choices=ALL_MODES,
                    help="prompt category; writes to <out>/<mode>[/<count>]/")
    ap.add_argument("--count", type=int, default=1,
                    help="wrenches per prompt (positive modes only; ignored for negatives)")
    ap.add_argument("--n", type=int, default=1000, help="number of prompts to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/prompts", help="parent output directory")
    args = ap.parse_args()

    if args.mode in POSITIVE_MODES and args.count < 1:
        ap.error(f"--count must be >=1 for positive mode {args.mode}")

    prompts = generate(args.mode, args.count, args.n, args.seed)
    out_dir = leaf_dir(Path(args.out), args.mode, args.count)
    jsonl, txt = write(prompts, out_dir)
    rel = out_dir.relative_to(Path(args.out)).as_posix()
    print(f"wrote {len(prompts)} {rel} prompts -> {jsonl}, {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
