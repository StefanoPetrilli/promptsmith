# Numeri — Specifications

A reusable, **model-agnostic** pipeline that: (1) generates a synthetic image dataset
using small models running **locally**, (2) labels that dataset using small models running
**locally**, (3) mixes it with a pre-existing real dataset, and (4) fine-tunes a small YOLO
detector. Quality of the synthetic-augmented training is validated against a real-only
baseline on the **Street View House Numbers (SVHN)** dataset.

---

## Hardware target (reference)
- NVIDIA RTX 2060, 6 GB VRAM · 12 CPU threads · 15 GB RAM · Python 3.10.
- All models must run locally and fit this profile (small / quantized). The pipeline must
  not hard-code any model — only constraints (max params, fp/bits, device) are configured.

## Guiding principle: pluggable & reusable
- **No model is hard-coded.** Generation and labeling are interfaces backed by swappable
  backends selected via config (diffusion for generation, VLM for labeling).
- **No use case is hard-coded.** SVHN is the *validation* scenario; the same stages can be
  retargeted to any object-detection task by changing config + a dataset adapter.
- Two distinct model roles (do not conflate):
  - **Generation backend** → produces images from prompts (typically a diffusion model).
  - **Labeling backend** → produces per-image annotations incl. YOLO bounding boxes
    (typically a vision-language / grounding model).

## Pipeline stages (each = a task namespace)

| Namespace      | Task(s)                                | Responsibility                                                       |
|----------------|----------------------------------------|----------------------------------------------------------------------|
| `dataset`      | `dataset:download`, `dataset:parse`   | Fetch & convert real dataset → YOLO format (images + labels + yaml). |
| `image-generation` | `image-generation:setup`, `image-generation:start`           | Generate synthetic images from templated prompts (seeded, reproducible). |
| `labeling`     | `labeling:setup`, `labeling:start`     | Auto-annotate synthetic images → YOLO boxes + classes; optional VLM quality filter. |
| `mix`          | `mix:build`                            | Combine real + synthetic at a configurable ratio; never place synthetic in test. |
| `train`        | `train:run`, `train:baseline`          | Fine-tune YOLO (real-only vs real+synthetic).                        |
| `eval`         | `eval:run`, `eval:compare`             | mAP metrics + comparison table/plots.                                |
| `env`          | `env:setup`, `env:check`               | Python venv, dependencies, device sanity check.                     |

## Scaffolding to create

```
numeri/
├─ SPECIFICATIONS.md            # this file
├─ README.md                    # quickstart: install go-task, env:setup, then run pipeline
├─ Taskfile.yml                 # go-task entry — delegates to namespace Taskfiles
├─ .gitignore
├─ pyproject.toml               # deps: torch, ultralytics, diffusers, transformers, PIL, pyyaml, numpy; project pkg `numeri`
├─ config/
│  └─ default.yaml              # dataset, generation/labeling backends (selected by name), mix ratio, yolo, paths
├─ numeri/                      # importable Python package
│  ├─ __init__.py
│  ├─ cli.py                    # `python -m numeri <stage>` entry, mirrors tasks
│  ├─ config.py                 # load/validate config.yaml
│  ├─ registry.py               # backend registry: register/lookup generation & labeling backends by name
│  ├─ datasets/
│  │  ├─ __init__.py
│  │  └─ svhn.py                 # SVHN download + → YOLO convert (an adapter; one adapter per use case)
│  ├─ generation/
│  │  ├─ __init__.py
│  │  ├─ base.py                 # GenerationBackend ABC: generate(prompts) -> image paths
│  │  ├─ prompts.py              # templated prompt builder (schema-driven, use-case agnostic)
│  │  └─ diffusion.py            # example diffusion backend (stub: load via diffusers with config kwargs)
│  ├─ labeling/
│  │  ├─ __init__.py
│  │  ├─ base.py                 # LabelingBackend ABC: label(images) -> YOLO annotations
│  │  └─ vlm.py                  # example VLM/grounding backend (stub)
│  ├─ mixing/
│  │  └─ build.py                # assemble real+synthetic splits with ratio + seed
│  ├─ train/
│  │  └─ yolo.py                  # ultralytics YOLOv8n/v11n fine-tune wrapper
│  ├─ eval/
│  │  └─ metrics.py               # run inference, compute mAP, compare runs
│  └─ utils/
│     ├─ io.py                    # YOLO read/write, image utils
│     └─ logging.py
├─ data/                        # gitignored: raw/, real_yolo/, synthetic/, mixed/, runs/
└─ tasks/                       # namespace Taskfiles included by root Taskfile.yml
   ├─ env.yml
   ├─ dataset.yml
   ├─ image-generation.yml
   ├─ labeling.yml
   ├─ mix.yml
   ├─ train.yml
   └─ eval.yml
```

## Build tooling
- Task runner: **go-task** (`task <namespace>:<action>`). Must be installed (not present yet).
- Python: venv at `.venv/` created by `env:setup`; package installed editable (`pip install -e .`).
- Config-driven: `config/default.yaml` references backends **by name**; swapping models =
  changing config + (optionally) registering a new backend class in `registry.py`.

## Open decisions (to confirm before implementation)
1. SVHN variant: full-number bbox version (not 32×32 crops)? .mat or folder images?
2. Generation backend default (must fit 6 GB VRAM).
3. Labeling backend default (must output bboxes; e.g. grounding VLM).
4. Mix ratio synthetic:real and whether test set is real-only (assumed: yes).
5. YOLO version: YOLOv8n vs YOLOv11n.
6. Include optional VLM quality-filter pass on generated images?

## Out of scope for now
- Cloud / API models (everything runs locally).
- Multi-GPU or distributed generation.
- Non-detection tasks (classification/segmentation) — architecture leaves room but not built.