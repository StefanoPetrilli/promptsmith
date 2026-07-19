# numeri

A fully-local pipeline that builds a small **single-class YOLO detector** (target object:
*"wrench"*) from scratch — synthetic images → auto-labels → fine-tune — runnable on a
6 GB NVIDIA GPU.

## Stages

| # | Task | What it does |
|---|------|--------------|
| 1 | `pipeline:prompts` (`--test`) | Write N varied wrench prompts → `data/prompts/<mode>/prompts.txt` (positives + synthetic **hard-negative** confusers) **and** sample N real wrench-free negatives from COCO val2017 → `data/synthetic/images/pure_negative/` |
| 2 | `pipeline:images` (`images-test`) | Render prompts with **FLUX.2-klein-9B** (fp16 + partial GPU pinning; Qwen3 embeds precomputed + cached) → `data/synthetic/images/` (skips the COCO leaf — it has no `prompts.txt`) |
| 3 | `pipeline:label` (`label-test`) | Text-grounded boxes/masks with **SAM 3** ("wrench") → YOLO `.txt` + `labels.jsonl` + mask PNGs in `data/synthetic/labels/` |
| 4 | `pipeline:visualize` (`visualize-test`) | Overlay segmentation + boxes on the images → `data/synthetic/visuals/` |
| 5 | `pipeline:handpick` (`handpick-test`) | Human approve/discard of segmentations → `data/synthetic/approved/` (+discarded) |
| 6 | `pipeline:postprocess` (`postprocess-test`) | Albumentations degradation (approved only) → `data/synthetic/images_pp/` |
| 6b | `pipeline:visualize-pp` (`visualize-pp-test`) | Overlay YOLO boxes on degraded images → `data/synthetic/visuals_pp/` |
| 7 | `pipeline:train` (`train-test`) | Assemble train/val split + fine-tune **YOLOv8n** → `data/dataset/runs/` |
|   | `pipeline:assemble-verification` | Stage + version the real verification dataset → `verification_dataset/assembled/` |
|   | `pipeline:train-real-val` | Train on synthetic images, validate on the real verification dataset |
|   | `pipeline:inference-verification` | Run the trained model on the verification dataset vs. GT |

Each `*-test` variant runs on a handful of images for a quick smoke check.

Each `*-test` variant runs on a handful of images for a quick smoke check.

## Requirements
- Python 3.10+, NVIDIA GPU (~6 GB VRAM) + ~8 GB free RAM. The FLUX.2-klein-9B transformer + VAE run fp16 (lossless) via partial GPU pinning (most blocks on GPU, rest offloaded per-block); Qwen3 text embeddings are precomputed + cached to disk, then released. SAM 3 runs on GPU.
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

# and on the post-processed images (after postprocess)
task pipeline:visualize-pp

# or smoke-test each step
task pipeline:test
task pipeline:images-test
task pipeline:label-test
task pipeline:train-test
task pipeline:visualize-test
task pipeline:visualize-pp-test
```

## Real-world validation with the verification dataset

The synthetic validation split can be replaced by a hand-curated real hold-out set under
`verification_dataset/` (positives with YOLO boxes; hard-negatives and negatives as
background to probe false positives). Assemble + version it, then train with it as the
validation split:

```bash
task pipeline:assemble-verification   # stage flat YOLO set -> verification_dataset/assembled/
task pipeline:visualize-verification   # overlay GT boxes for QC
task pipeline:train-real-val           # train on synthetic, validate on real images
task pipeline:inference-verification   # run trained model on the verification set vs GT
```

You can also run the assembler directly:

```bash
python pipeline/assemble_verification.py --root verification_dataset \
  --out verification_dataset/assembled
```

Then point training at it:

```bash
python pipeline/train_yolo.py --images data/synthetic/approved/images \
  --labels data/synthetic/approved/labels \
  --val-images verification_dataset/assembled/images \
  --val-labels verification_dataset/assembled/labels \
  --out data/dataset --epochs 500 --patience 15
```

The assembled dir is self-contained and versioned: `manifest.json` (per-file sha256,
dims, box counts, split) + `VERSION` (content-addressed). Commit it to git so every
run pins a specific dataset snapshot.

## Layout
- `pipeline/` — the five stage scripts (importable, run directly via `python pipeline/<step>.py`).
- `tasks/` — go-task definitions (`Taskfile.yml` → `env`, `pipeline`).
- `data/`, `models/` — gitignored outputs and downloaded checkpoints.

### `data/` layout
```
data/
├── prompts/        # stage 1 seed prompts (<mode>/prompts.{jsonl,txt})
├── synthetic/      # stage 2-6 intermediates
│   ├── images/      #   raw FLUX renders
│   ├── images_pp/   #   post-processed (Albumentations)
│   ├── labels/     #   SAM 3 labels + masks + labels.jsonl
│   ├── visuals/     #   SAM 3 overlay visualisations (for handpick review)
│   ├── visuals_pp/ #   YOLO overlays on post-processed images (final QC)
│   ├── approved/   #   hand-picked images + labels (symlinks)
│   └── discarded/  #   rejected images + labels (symlinks)
├── dataset/        # stage 5 assembled YOLO dataset (data.yaml, images/, labels/) + runs/
└── inference/      # model inference outputs (train/, val/, verification/)

verification_dataset/   # real hold-out set (committed, versioned)
├── positive/ hard_negatives/ negatives/   # source images (by split)
├── positive_labels/                       # YOLO .txt for positive images
└── assembled/                             # flat versioned YOLO set consumed by training
    ├── images/ labels/  visuals/
    ├── manifest.json   VERSION
```
