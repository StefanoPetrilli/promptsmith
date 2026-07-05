#!/usr/bin/env python3
"""Step 1 — generate synthetic wrench prompts across labelled categories."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

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

_DIST_WEIGHTS = {
    "clean_positive": [3, 4, 2, 1, 1, 1],
    "hard_positive": [1, 1, 2, 3, 3, 4],
    "asset": [2, 4, 2, 1, 1, 1],
}


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
    return f"some {finish} {_plural(subtype)}"


def _positive_text(framing: str, subtype: str, finish: str, ax: dict, neg: str) -> str:
    return _cap(_join([
        _object_phrase(subtype, finish),
        framing,
        f"{ax['arrangement']} on {ax['surface']} in {ax['scene']}",
        ax["orientation"], ax["distance"], ax["lighting"],
        "photorealistic",
        neg,
    ]))


def build_clean_positive(rng: random.Random) -> dict:
    subtype = rng.choice(WRENCH_TYPES)
    finish = rng.choice(FINISHES)
    ax = _common_axes(rng, "clean_positive")
    return {
        "text": _positive_text("fully in frame, clearly separated", subtype, finish, ax, NEG_CLEAN),
        "mode": "clean_positive",
        "subtype": subtype,
        "finish": finish,
        **ax,
    }


def build_hard_positive(rng: random.Random) -> dict:
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
    return {
        "text": _positive_text(framing, subtype, finish, ax, NEG_HARD_POS),
        "mode": "hard_positive",
        "subtype": subtype,
        "finish": finish,
        "framing": framing,
        **ax,
    }


def build_asset(rng: random.Random) -> dict:
    subtype = rng.choice(WRENCH_TYPES)
    finish = rng.choice(FINISHES)
    bg = rng.choice(PLAIN_BACKGROUNDS)
    ax = _common_axes(rng, "asset")
    return {
        "text": _cap(_join([
            _object_phrase(subtype, finish),
            "fully in frame, clearly separated",
            f"on {bg} background",
            ax["orientation"], ax["distance"], ax["lighting"],
            "photorealistic, studio product shot",
            NEG_ASSET,
        ])),
        "mode": "asset",
        "subtype": subtype,
        "finish": finish,
        "background": bg,
        **ax,
    }


def _confuser_phrase(rng: random.Random, confuser: str) -> str:
    count = rng.choice([1, 1, 1, 2, 2, 3])
    if count == 1:
        return f"{_article(confuser)}{confuser}"
    return f"{COUNT_WORDS[count]} {_plural(confuser)}"


def build_hard_negative(rng: random.Random) -> dict:
    confuser = rng.choice(CONFUSERS)
    ax = _common_axes(rng, "hard_positive")
    return {
        "text": _cap(_join([
            _confuser_phrase(rng, confuser),
            f"{ax['arrangement']} on {ax['surface']} in {ax['scene']}",
            ax["distance"], ax["lighting"],
            "photorealistic",
            NEG_HARD_NEG,
        ])),
        "mode": "hard_negative",
        "count": 0,
        "confuser": confuser,
        **ax,
    }


def build_pure_negative(rng: random.Random) -> dict:
    ax = _common_axes(rng, "clean_positive")
    return {
        "text": _cap(_join([
            rng.choice(SHOP_ITEMS),
            f"in {ax['scene']}",
            f"on {ax['surface']}",
            ax["distance"], ax["lighting"],
            "photorealistic",
            NEG_PURE_NEG,
        ])),
        "mode": "pure_negative",
        "count": 0,
        "shop_item": rng.choice(SHOP_ITEMS),
        **ax,
    }


BUILDERS = {
    "clean_positive": build_clean_positive,
    "hard_positive": build_hard_positive,
    "asset": build_asset,
    "hard_negative": build_hard_negative,
    "pure_negative": build_pure_negative,
}


def leaf_seed(base_seed: int, mode: str) -> int:
    h = hashlib.sha1(mode.encode()).hexdigest()
    return base_seed + (int(h, 16) % 1_000_000)


def generate(mode: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(leaf_seed(seed, mode))
    builder = BUILDERS[mode]
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < n * 20:
        p = builder(rng)
        if p["text"] not in seen:
            seen.add(p["text"])
            out.append(p)
        tries += 1
    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)} unique {mode} prompts after {tries} tries")
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
    ap.add_argument("--mode", required=True, choices=ALL_MODES)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/prompts")
    args = ap.parse_args()

    prompts = generate(args.mode, args.n, args.seed)
    out_dir = Path(args.out) / args.mode
    jsonl, txt = write(prompts, out_dir)
    print(f"wrote {len(prompts)} {args.mode} prompts -> {jsonl}, {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
