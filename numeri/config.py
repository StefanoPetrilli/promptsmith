"""Config loading + minimal validation.

Config is a plain dict; backends resolve their own entries by name via
`numeri.registry`. No hard-coded model assumptions live here.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def _resolve_refs(value: Any) -> Any:
    """Resolve dotted `${a.b.c}` references against the parent config.

    A *copy* of the in-progress dict is used and parameters are resolved in a single pass,
    so references should point to keys that appear earlier or are scalars. Keep it simple.
    """
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            ref = value[2:-1]
            # resolved against the config after a full load (see load() second pass)
            return value
        return value
    if isinstance(value, dict):
        return {k: _resolve_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v) for v in value]
    return value


def _apply_refs(cfg: dict[str, Any], root: dict[str, Any]) -> Any:
    if isinstance(cfg, str) and cfg.startswith("${") and cfg.endswith("}"):
        ref = cfg[2:-1]
        cur: Any = root
        for part in ref.split("."):
            cur = cur[part]
        return cur
    if isinstance(cfg, dict):
        return {k: _apply_refs(v, root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_apply_refs(v, root) for v in cfg]
    return cfg


def load(config: str | Path | None = None) -> dict[str, Any]:
    """Load config. Resolution order:
    1. explicit `config` path if given, or NUMERI_CONFIG env var,
    2. config/local.yaml (user override),
    3. config/default.yaml (fallback).

    Deep-merge local over default — ensures users only override what they need.
    """
    picked = (
        Path(config)
        if config
        else Path(os.environ.get("NUMERI_CONFIG", "config/local.yaml"))
    )
    base = _load_yaml(DEFAULT_CONFIG_PATH)
    if picked.exists() and picked != DEFAULT_CONFIG_PATH:
        local = _load_yaml(picked)
        base = _deep_merge(base, local)
        log.info("Loaded config from default + %s", picked)
    else:
        log.info("Loaded config from %s", DEFAULT_CONFIG_PATH)
    return _apply_refs(base, base)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)} in {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def setup_logging(cfg: dict[str, Any]) -> None:
    logging.basicConfig(
        level=get(cfg, "log_level", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )