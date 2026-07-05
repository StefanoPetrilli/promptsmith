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
    "pipe wrench",
    "box-end wrench",
    "ratcheting wrench",
    "stubby wrench",
    "flare-nut wrench",
    "open-end wrench",
    "adjustable pipe wrench",
    "flex combination wrench"
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
    "hanging on a hook", "in a toolbox", "crossed over another wrench",
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

COUNT_WORDS = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five"}

# --- negatives / non-wrench content ------------------------------------------

# Confusers for hard_negative (no wrench).
CONFUSERS = [
    "pliers", "slip-joint pliers", "breaker bar", "tire iron",
    "pry bar", "flathead screwdriver", "Phillips screwdriver",
    "socket extension bar", "hex-bit driver", "locking pliers",
    "diagonal cutter", "needle-nose pliers",
]

# Plain backgrounds for asset mode.
PLAIN_BACKGROUNDS = [
    "a white seamless", "a gray seamless",
    "an off-white sweep", "a pure black", "a white cyclorama",
]

# Non-confuser shop items for pure_negative (no wrench, no confusers).
SHOP_ITEMS = [
    "a stack of bolts", "a coil of wire", "a roll of duct tape",
    "a parts bin", "a shop vacuum", "a battery charger",
    "jumper cables", "a pile of shop rags", "a fluid funnel",
    "a measuring tape", "a flashlight", "a shop press",
    "a jack stand", "a creeper seat", "an air hose reel",
]

# Short negative tails (FLUX `--no ...` syntax). Kept compact; the suppressed list still
# covers the failure modes that matter (extra wrenches, look-alikes, wrench-in-negative).
NEG_CLEAN = "--no other wrenches, no wrench-like tools"
NEG_HARD_POS = "--no extra wrenches, no wrench-like tools"
NEG_ASSET = "--no background objects, no other wrenches"
NEG_HARD_NEG = "--no wrench"
NEG_PURE_NEG = "--no wrench, no pliers, no screwdrivers, no pry bars"
