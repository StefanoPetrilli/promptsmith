"""Combination lists and constants for the prompt generator.

All the Cartesian-product axes (wrench subtypes, finishes, scenes, surfaces, ...) and the
mode-aware negative strings live here so `generate_prompts.py` stays focused on rendering
logic. Import these as `from pipeline.combinations import ...`.
"""
from __future__ import annotations

# Scope note: the target object is an *adjustable spanner* and tools close to it
WRENCH_TYPES = [
    "combination wrench",
    "crescent adjustable wrench",
    "box-end wrench",
    "stubby wrench",
    "flare-nut wrench",
    "open-end wrench"
]

FINISHES = [
    "polished chrome", "chrome", "matte black oxide", "brushed steel",
    "rusty", "greasy", "brand new", "painted blue", "red-handled",
    "painted red", "chrome-vanadium", "nickel-plated", "oxidized",
    "scuffed", "mirror finish", "salt-stained",
]

ENVIRONMENTS = [
    "a home garage",
    "a mechanic's shop",
    "a kitchen drawer",
    "a truck bed",
    "a construction site",
    "a basement workshop",
    "a driveway bike stand",
    "a factory floor",
    "a hardware store shelf",
    "a garden shed",
    "an engine bay",
    "a motorcycle garage",
    "a maker-space bench",
    "a plumbing van",
    "an aircraft hangar",
    "a farm barn",
    "a rooftop HVAC unit",
    "an office desk",
    "a roadside breakdown",
    "a carport bench",
    "a service pit",
    "a tool crib",
    "a marine engine room",
    "a lawn-mower bench",
    "a heavy-equipment cab",
    "a paint booth",
    "a tire-shop bay",
    "a welding table",
    "a shop bench",
    "a garage attic",
    "a carport workbench",
    "an outdoor gravel mat",
]

SURFACES = [
    "a wooden workbench", "a concrete floor", "a toolbox tray", "a pegboard",
    "an engine block", "a stainless bench", "a tiled surface", "a cloth tool roll",
    "a cardboard mat", "a parts tray", "a rubber mat", "a shop rag",
    "a steel shelf", "a vise anvil", "a leather tool mat", "a tailgate",
    "a cart top", "a masonite cover", "a composite surface", "an enamel tray",
]

ARRANGEMENTS = [
    "lying flat", "scattered", "fitted on a bolt", "held in a hand",
    "hanging on a hook", "in a toolbox",
    "standing on its box-end", "on a bolt head", "tucked in a tool roll",
    "balanced on a pipe", "across two sockets", "clipped to a belt",
    "propped on a toolbox", "on a shop rag", "in a pile of fasteners",
]

ORIENTATIONS = [
    "horizontal", "vertical", "diagonal", "pointing toward camera",
    "pointing away from camera", "flat to the viewer", "tilted forty-five degrees",
    "propped upright", "curved across the frame",
]

# Distance / framing — swept heavily for scale variety.
DISTANCES = [
    "extreme close-up", "close-up", "medium shot", "mid-distance",
    "far shot", "distant",
]

LIGHTINGS = [
    "soft shop light", "fluorescent light", "tungsten lamp", "window daylight",
    "low light", "backlight", "mixed light", "direct sunlight",
    "overcast light", "swing-arm lamp", "LED strip", "ring light",
]

# --- negatives / non-wrench content ------------------------------------------

# Confusers for hard_negative (no wrench).
CONFUSERS = [
    "pliers", "slip-joint pliers", "breaker bar", "tire iron",
    "pry bar", "flathead screwdriver", "Phillips screwdriver",
    "socket extension bar", "hex-bit driver", "locking pliers",
    "diagonal cutter", "needle-nose pliers",
]

# --- dense / multi-instance positive prompts ------------------------------

DENSE_COUNTS = ["three", "four", "five", "six", "eight"]
DENSE_ARRANGEMENTS = [
    "laid out in a row",
    "hanging on a pegboard",
    "arranged in an open toolbox drawer",
    "scattered on a workbench",
    "lined up on a cloth tool roll",
    "stacked on a steel shelf",
    "mounted on a wall rack",
    "placed in a parts tray",
]
DENSE_FINISHES = [
    "chrome", "polished chrome", "chrome-vanadium", "brushed steel",
    "rusty", "greasy", "matte black oxide", "nickel-plated",
]
DENSE_WRENCH_SETS = [
    "combination wrenches",
    "open-end wrenches",
    "box-end wrenches",
    "crescent adjustable wrenches",
    "wrenches of different sizes",
    "combination and open-end wrenches",
]

# Plain backgrounds for asset mode.
PLAIN_BACKGROUNDS = [
    "a white seamless", "a gray seamless",
    "an off-white sweep", "a pure black", "a white cyclorama",
]
