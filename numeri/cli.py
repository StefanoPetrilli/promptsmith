"""CLI entry: `python -m numeri <action>` / `numeri <action>`.

Mirrors the go-task taskfiles: each action instantiates the configured backend via
`numeri.registry` and calls a stage-specific function. Keeping it thin means tasks and
the CLI never duplicate logic.
"""
from __future__ import annotations

import sys

from . import config as cfg_mod

# Importing these modules triggers backend registration with the registries.
from .datasets import svhn as _svhn  # noqa: F401
from .generation import diffusion as _diffusion  # noqa: F401
from .labeling import vlm as _vlm  # noqa: F401
from .registry import dataset_adapters, generation_backends, labeling_backends


def _load() -> dict:
    c = cfg_mod.load()  # picks up NUMERI_CONFIG or config/local.yaml over default
    cfg_mod.setup_logging(c)
    return c


# --- actions -----------------------------------------------------------------

def env_check() -> int:
    cfg = _load()
    import torch
    print("numeri env check")
    print(f"  python    : {sys.version.split()[0]}")
    print(f"  torch     : {torch.__version__}")
    print(f"  cuda avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu       : {torch.cuda.get_device_name(0)}")
    print(f"  gen backend : {generation_backends.names()}")
    print(f"  lab backend : {labeling_backends.names()}")
    print(f"  datasets    : {dataset_adapters.names()}")
    return 0


def dataset_download() -> int:
    cfg = _load()
    adapter = dataset_adapters.get(cfg["dataset"]["name"])(cfg)
    adapter.download()
    return 0


def dataset_parse() -> int:
    cfg = _load()
    adapter = dataset_adapters.get(cfg["dataset"]["name"])(cfg)
    adapter.parse()
    return 0


def generation_setup() -> int:
    cfg = _load()
    backend = generation_backends.get(cfg["generation"]["backend"])(cfg)
    backend.setup()
    return 0


def generation_start() -> int:
    cfg = _load()
    from .generation.prompts import build
    from .utils.io import ensure_dir
    out = ensure_dir(cfg["paths"]["synthetic"] + "/images")
    prompts = build(cfg)
    backend = generation_backends.get(cfg["generation"]["backend"])(cfg)
    backend.generate(prompts, out, cfg["generation"].get("num_images", 1))
    return 0


def labeling_setup() -> int:
    cfg = _load()
    backend = labeling_backends.get(cfg["labeling"]["backend"])(cfg)
    backend.setup()
    return 0


def labeling_start() -> int:
    cfg = _load()
    from pathlib import Path
    backend = labeling_backends.get(cfg["labeling"]["backend"])(cfg)
    imgs = sorted(Path(cfg["paths"]["synthetic"]).rglob("*.png"))
    backend.label(imgs)
    return 0


def mix_build() -> int:
    cfg = _load()
    from .mixing.build import build_mix
    build_mix(cfg)
    return 0


def train_baseline() -> int:
    cfg = _load()
    from .train.yolo import train_baseline
    train_baseline(cfg)
    return 0


def train_run() -> int:
    cfg = _load()
    from .train.yolo import train_run
    train_run(cfg)
    return 0


def eval_run() -> int:
    cfg = _load()
    from .eval.metrics import evaluate
    evaluate(cfg, "augmented")
    return 0


def eval_compare() -> int:
    cfg = _load()
    from .eval.metrics import compare
    compare(cfg)
    return 0


_ACTIONS = {
    "env-check": env_check,
    "dataset-download": dataset_download,
    "dataset-parse": dataset_parse,
    "generation-setup": generation_setup,
    "generation-start": generation_start,
    "labeling-setup": labeling_setup,
    "labeling-start": labeling_start,
    "mix-build": mix_build,
    "train-baseline": train_baseline,
    "train-run": train_run,
    "eval-run": eval_run,
    "eval-compare": eval_compare,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m numeri <action>")
        print("actions:")
        for a in _ACTIONS:
            print(f"  {a}")
        return 0
    action = argv[0]
    if action not in _ACTIONS:
        print(f"unknown action: {action}", file=sys.stderr)
        print(f"available: {', '.join(_ACTIONS)}", file=sys.stderr)
        return 2
    return _ACTIONS[action]()


if __name__ == "__main__":
    sys.exit(main())