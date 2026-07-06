# numeri

A fully-local pipeline that builds a small **single-class YOLO detector** (target object:
*"wrench"*) from scratch — synthetic images → auto-labels → fine-tune — runnable on a
6 GB NVIDIA GPU.

## Stages

| # | Task | What it does |
|---|------|--------------|
| 1 | `pipeline:prompts` (`--test`) | Write N varied wrench prompts → `data/prompts/<mode>/prompts.txt` |
| 2 | `pipeline:images` (`images-test`) | Render prompts with **FLUX.2-klein-4B** (fp16 + partial GPU pinning; Qwen3 embeds precomputed + cached) → `data/synthetic/images/` |
| 3 | `pipeline:label` (`label-test`) | Text-grounded boxes/masks with **SAM 3** ("wrench") → YOLO `.txt` + `labels.jsonl` + mask PNGs in `data/synthetic/labels/` |
| 4 | `pipeline:visualize` (`visualize-test`) | Overlay segmentation + boxes on the images → `data/synthetic/visuals/` |
| 5 | `pipeline:train` (`train-test`) | Assemble train/val split + fine-tune **YOLOv8n** → `data/dataset/runs/` |
|   | `pipeline:download-openimages` | Download real **Open Images** wrench images for validation → `data/openimages/` |
|   | `pipeline:train-real-val` | Train on synthetic images, validate on real Open Images |

Each `*-test` variant runs on a handful of images for a quick smoke check.

## Requirements
- Python 3.10+, NVIDIA GPU (~6 GB VRAM) + ~8 GB free RAM. The FLUX.2-klein-4B transformer + VAE run fp16 (lossless) via partial GPU pinning (most blocks on GPU, rest offloaded per-block); Qwen3 text embeddings are precomputed + cached to disk, then released. SAM 3 runs on GPU.
- [go-task](https://taskfile.dev) 3.x.

## Quickstart
```bash
task env:setup          # venv + deps + fetch SAM 3 ckpt and CLIP BPE vocab
task env:check         # show torch / CUDA availability

# full pipeline (1000 prompts)
task pipeline:prompts
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

## Real-world validation with Open Images

The synthetic validation split can be replaced by real wrench images from Open Images:

```bash
task pipeline:download-openimages   # ~200 real wrench images -> data/openimages/
task pipeline:train-real-val        # train on synthetic, validate on real images
```

You can also run the downloader directly:

```bash
python pipeline/download_openimages.py --class-name Wrench --split train \
  --out data/openimages --limit 500
```

Then point training at the real validation set:

```bash
python pipeline/train_yolo.py --images data/synthetic/images --labels data/synthetic/labels \
  --val-images data/openimages/images --val-labels data/openimages/labels \
  --out data/dataset --epochs 500 --patience 15
```

## Layout
- `pipeline/` — the five stage scripts (importable, run directly via `python pipeline/<step>.py`).
- `tasks/` — go-task definitions (`Taskfile.yml` → `env`, `pipeline`).
- `data/`, `models/` — gitignored outputs and downloaded checkpoints.

### `data/` layout
```
data/
├── prompts/        # stage 1 seed prompts (<mode>/prompts.{jsonl,txt})
├── synthetic/      # stage 2-4 intermediates
│   ├── images/      #   raw FLUX renders
│   ├── images_pp/   #   post-processed (Albumentations)
│   ├── labels/     #   SAM 3 labels + masks + labels.jsonl
│   └── visuals/     #   overlay visualisations
├── openimages/     # real-world validation set + Open Images metadata cache
│   ├── images/  labels/  visuals/  cache/
├── dataset/        # stage 5 assembled YOLO dataset (data.yaml, images/, labels/) + runs/
└── inference/      # model inference outputs (train/, val/, openimages/)
```
