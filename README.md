# numeri

A fully-local pipeline that builds a small **single-class YOLO detector** (target object:
*"wrench"*) from scratch — synthetic images → auto-labels → fine-tune — runnable on a
6 GB NVIDIA GPU.

## Stages

| # | Task | What it does |
|---|------|--------------|
| 1 | `pipeline:generate` (`--test`) | Write N varied *"one wrench"* prompts → `data/prompts.{jsonl,txt}` |
| 2 | `pipeline:images` (`images-test`) | Render prompts with **FLUX.2-klein-4B** (NF4, Qwen3 embeds on CPU) → `data/images/` |
| 3 | `pipeline:label` (`label-test`) | Text-grounded boxes/masks with **SAM 3** ("wrench") → YOLO `.txt` + `labels.jsonl` |
| 4 | `pipeline:visualize` (`visualize-test`) | Overlay SAM 3 segmentation + boxes on the images → `data/visuals/` |
| 5 | `pipeline:train` (`train-test`) | Assemble train/val split + fine-tune **YOLOv8n** → `data/dataset/runs/` |

Each `*-test` variant runs on a handful of images for a quick smoke check.

## Requirements
- Python 3.10+, NVIDIA GPU (~6 GB VRAM). FLUX.2 Qwen3 encoder runs on CPU; SAM 3 + VAE + NF4 transformer on GPU.
- [go-task](https://taskfile.dev) 3.x.

## Quickstart
```bash
task env:setup          # venv + deps + fetch SAM 3 ckpt and CLIP BPE vocab
task env:check         # show torch / CUDA availability

# full pipeline (1000 prompts)
task pipeline:generate
task pipeline:images
task pipeline:label
task pipeline:train
task pipeline:visualize

# or smoke-test each step
task pipeline:test
task pipeline:images-test
task pipeline:label-test
task pipeline:train-test
task pipeline:visualize-test
```

## Layout
- `pipeline/` — the five stage scripts (importable, run directly via `python pipeline/<step>.py`).
- `tasks/` — go-task definitions (`Taskfile.yml` → `env`, `pipeline`).
- `data/`, `models/` — gitignored outputs and downloaded checkpoints.