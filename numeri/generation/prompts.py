"""Prompt builders for synthetic generation.

A "template" is a callable producing a list of text prompts from a config-driven spec.
Templates are registered by name (matching `generation.prompt_template` in config) so the
prompting strategy stays decoupled from any particular model or use case.
"""
from __future__ import annotations

import random
from typing import Callable

PromptFn = Callable[[dict], list[str]]
_templates: dict[str, PromptFn] = {}


def register(name: str) -> Callable[[PromptFn], PromptFn]:
    def deco(fn: PromptFn) -> PromptFn:
        _templates[name] = fn
        return fn

    return deco


def build(cfg: dict) -> list[str]:
    """Return the list of prompts for the configured template + num_images."""
    name = cfg["generation"]["prompt_template"]
    if name not in _templates:
        raise KeyError(
            f"Unknown prompt template '{name}'. Available: {sorted(_templates)}"
        )
    rng = random.Random(cfg.get("seed", 0))
    _set_rng(rng)
    return _templates[name](cfg)


_active_rng = None


def _set_rng(rng: random.Random) -> None:
    global _active_rng
    _active_rng = rng


@register("svhn_house_number")
def svhn_house_number(cfg: dict) -> list[str]:
    """Templated prompts describing street-level house-number scenes (SVHN validation).

    TODO: implement richer templating (digit count, font, color, background, lighting,
    perspective). For now a placeholder shape so the pipeline wiring is testable.
    """
    n = cfg["generation"].get("num_images", 1)
    # TODO: replace with a real prompt combinator
    return [
        "a street-level photograph of a house number plaque on a residential facade, "
        "crisp digits, varied font and color, daytime, natural perspective"
    ] * n


@register("generic")
def generic(cfg: dict) -> list[str]:
    """Use-case-agnostic placeholder: just repeat a configurable base prompt."""
    base = cfg["generation"].get("prompt", "a photo of a scene")
    n = cfg["generation"].get("num_images", 1)
    return [base] * n