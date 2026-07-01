#!/usr/bin/env python3
"""Step 1 — build prompts for synthetic photos of *exactly one wrench*.

Varies four axes (environment, wrench type, device, perspective) plus surface / finish /
arrangement. Writes `<out>/prompts.{jsonl,txt}` (txt = one prompt per line, fed to step 2).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --- axes --------------------------------------------------------------------

ENVIRONMENTS = [
    "a home garage workshop with a wooden workbench",
    "a professional mechanic's shop with tool chests",
    "a kitchen drawer partially pulled open",
    "a truck bed with an open metal toolbox",
    "a construction site with concrete and rebar",
    "a basement workshop with pegboard walls",
    "a bicycle repair stand in a driveway",
    "a factory floor with steel equipment",
    "a hardware store shelf full of packaged tools",
    "a garden shed with a dusty worktable",
    "an automotive engine bay with the hood up",
    "a motorcycle garage with parts on a bench",
    "a maker-space workbench with soldering gear",
    "a plumbing van interior with shelves",
    "an aircraft maintenance hangar floor",
    "a farm equipment barn with a workbench",
    "a rooftop HVAC unit with a service kit beside it",
    "a cluttered office desk with a small toolkit",
    "a roadside breakdown scene with the car hood up",
    "a workshop bench under a carport",
]

WRENCH_TYPES = [
    "combination wrench (open-end on one side, box-end on the other)",
    "adjustable crescent wrench",
    "socket wrench with a ratchet handle",
    "torque wrench with a long handle and gauge",
    "heavy pipe wrench with serrated jaws",
    "box-end wrench",
    "ratcheting combination wrench",
    "stubby open-end wrench",
    "flare-nut wrench",
    "Allen (hex) key set laid out",
    "spanner wrench with holes",
    "crowfoot wrench",
]

SURFACES = [
    "a scarred wooden workbench",
    "a concrete floor",
    "a red metal toolbox tray",
    "a perforated pegboard",
    "a greasy engine block",
    "a stainless steel bench",
    "a tiled work surface",
    "an open cloth tool roll",
    "a cardboard mat",
    "a plastic parts tray",
    "a workbench with a rubber mat",
    "a dirty shop rag",
]

FINISHES = [
    "polished chrome", "chrome", "matte black oxide",
    "brushed steel", "rusty and weathered", "greasy and well-used",
    "brand new and shiny", "painted blue", "two-tone with red handle",
]

ARRANGEMENTS = [
    "lying flat, neatly laid out",
    "scattered casually",
    "in use, fitted onto a bolt",
    "held in a gloved hand mid-turn",
    "hanging from a pegboard hook",
    "resting inside an open toolbox",
    "crossed over another wrench",
    "standing on its box-end",
]

DEVICES = [
    "a smartphone camera",
    "a DSLR with a 50mm lens",
    "a cheap phone camera with slight motion blur",
    "a dashcam",
    "a CCTV security camera",
    "a doorbell camera",
    "a drone looking down",
    "a 35mm film camera with grain",
    "a polaroid instant camera",
    "a wide-angle action camera",
    "an old analog camcorder",
    "a high-end mirrorless camera with sharp focus",
]

PERSPECTIVES = [
    "straight-on eye-level",
    "slightly low angle looking up",
    "slightly high angle looking down",
    "a side angle from the left",
    "a side angle from the right",
    "a three-quarter view",
    "a bird's-eye view from directly above",
    "a worm's-eye view from near the surface",
    "a tilted dutch-angle composition",
    "a close-up cropped tightly on the wrench",
    "a moderately cropped medium shot",
]


def build_prompt(rng: random.Random) -> dict:
    wtype = rng.choice(WRENCH_TYPES)
    finish = rng.choice(FINISHES)
    env = rng.choice(ENVIRONMENTS)
    surface = rng.choice(SURFACES)
    arrangement = rng.choice(ARRANGEMENTS)
    dev = rng.choice(DEVICES)
    pers = rng.choice(PERSPECTIVES)

    text = (
        f"{dev.capitalize()} photo of a {finish} {wtype} {arrangement} on {surface}, "
        f"in {env}, {pers}, photorealistic, sharp focus on the wrench"
    )
    return {
        "text": text,
        "wrench_count": 1,
        "wrench_type": wtype,
        "finish": finish,
        "environment": env,
        "surface": surface,
        "arrangement": arrangement,
        "device": dev,
        "perspective": pers,
    }


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen = set()
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        p = build_prompt(rng)
        key = p["text"]
        if key not in seen:
            seen.add(key)
            out.append(p)
        tries += 1
    if len(out) < n:
        raise RuntimeError(f"could only generate {len(out)} unique prompts after {tries} tries")
    return out


def write(prompts: list[dict], out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{stem}.jsonl"
    txt = out_dir / f"{stem}.txt"
    with jsonl.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with txt.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p["text"] + "\n")
    return jsonl, txt


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic-image prompts (step 1).")
    ap.add_argument("--n", type=int, default=1000, help="number of prompts to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data", help="output directory")
    args = ap.parse_args()

    prompts = generate(args.n, args.seed)
    jsonl, txt = write(prompts, Path(args.out), "prompts")
    print(f"wrote {len(prompts)} prompts -> {jsonl}, {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
