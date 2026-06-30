# Numeri — Synthetic-wrench pipeline plan

Goal: fine-tune a small YOLO detector that recognizes **wrenches** in the wild, trained
entirely on **synthetic** data. The pipeline has four steps; each has a full task and a
`*-test` task that runs on a handful of images so the step can be verified in isolation
before committing to a full run.

All pipeline code lives in [`pipeline/`](pipeline) and is driven by go-task
(`task pipeline:<step>`). Exploratory probes that led to this plan live in
[`experiments/`](experiments) and are kept for reference only.

```
prompts (1000)  ──step 1──▶  images (1000)  ──step 2──▶  labels (YOLO)  ──step 3──▶  YOLO model ──step 4──▶
                                 (FLUX.2-klein)          (SAM 3 boxes)       (single class: wrench)
                                       │                     │
                                       └────────step 5────────▶  visuals (mask + box overlays)
```

> **Direction change (2026-06).** The dataset used to target house-number digits (0-9)
> labeled by a remote Qwen 3.7 Plus VLM. It now targets a **single object class —
> wrenches** — labeled locally by **Segment Anything 3 (SAM 3)** used as an
> open-vocabulary text-grounded segmenter (prompt: `"wrench"`). No remote API is involved
> anymore; the whole pipeline runs offline on the 6 GB Turing GPU.

## Data layout (all gitignored, regenerable)

| Path | Produced by | Contents |
|------|-------------|----------|
| `data/prompts.jsonl`, `data/prompts.txt` | step 1 | 1000 wrench prompts + per-prompt metadata |
| `data/images/img_XXXX.png`, `data/images/manifest.jsonl` | step 2 | rendered images + prompt→image map |
| `data/labels/img_XXXX.txt`, `data/labels/labels.jsonl`, `data/labels/masks/img_XXXX.png` | step 3 | YOLO labels (class 0=wrench) + SAM 3 audit log + instance-mask PNGs |
| `data/dataset/{images,labels}/{train,val}`, `data/dataset/data.yaml` | step 4 | assembled YOLO dataset |
| `data/dataset/runs/train/weights/best.pt` | step 4 | fine-tuned detector |
| `data/visuals/img_XXXX.png`, `data/visuals/visuals.jsonl` | step 5 | image copies with mask overlay + boxes drawn |
| `models/yolov8n.pt` | step 4 (cached) | base YOLO weights |
| `models/sam3/sam3.pt` | step 3 (cached) | SAM 3 checkpoint |

The `*_test` variants write to `data/<stage>_test/`. The step-5 test variant writes to
`data/visuals_test/` and consumes `data/labels_test/`.

---

## Step 1 — Generate prompts  (`pipeline/generate_prompts.py`)

A seeded RNG iteratively samples a wrench scene along four axes, plus secondary
attributes, until it has N unique prompts:

- **environment** — garage workbench, mechanic's shop, kitchen drawer, truck-bed toolbox,
  construction site, basement pegboard, bicycle repair stand, factory floor, hardware
  shelf, garden shed, engine bay, ...
- **wrench type** — combination, adjustable/crescent, socket+ratchet, torque, pipe,
  box-end, ratcheting, stubby, flare-nut, hex/Allen, spanner, crowfoot ...
- **device** — smartphone, DSLR, dashcam, CCTV, doorbell cam, drone, 35mm film, polaroid, ...
- **perspective** — eye-level, low/high angle, side, three-quarter, bird's-eye, worm's-eye, dutch, close-up, ...

Secondary: surface (workbench, concrete, toolbox tray, pegboard, engine block, ...),
wrench count (1-4, weighted toward 1-2), arrangement (laid out, scattered, in use on a
bolt, held in hand, hanging, ...), finish/condition (chrome, black oxide, rusty, greasy,
brand-new, painted), time of day, lighting, weather.

Output: `prompts.jsonl` (metadata, for auditing variety) and `prompts.txt` (one prompt per
line, fed to step 2). Each prompt explicitly mentions the wrench(es) so FLUX.2 renders
them, and ends with "sharp focus on the wrench".

| Task | Command | Output |
|------|---------|--------|
| `task pipeline:generate` | `--n 1000` | `data/prompts.{jsonl,txt}` |
| `task pipeline:test` | `--test` (12 prompts, printed to stdout) | `data/prompts_test.{jsonl,txt}` |

**Verified:** `task pipeline:test` produces 12 wrench prompts covering all four primary axes
and a spread of wrench counts/types.

---

## Step 2 — Generate images  (`pipeline/generate_images.py`)

Unchanged from the digit pipeline. Renders each prompt with **FLUX.2-klein-4B** on the
6 GB Turing GPU, using the loader validated in `experiments/try_flux2.py`:

- Qwen3 text encoder runs on **CPU**; prompt embeddings are precomputed and cached to disk
  (`_embeds_<hash>.pt`), then Qwen3 is released — it never goes on the GPU.
- The 4B transformer is loaded as **bitsandbytes NF4** on cuda (~2 GB); VAE is fp16 on cuda.
- No `enable_model_cpu_offload` (it would pull Qwen3 → GPU and OOM).

Outputs `img_XXXX.png` and a `manifest.jsonl` (prompt→image mapping, consumed by step 3).

| Task | Command | Output |
|------|---------|--------|
| `task pipeline:images` | all prompts, 768px, 8 steps | `data/images/` |
| `task pipeline:images-test` | first 3 test prompts | `data/images_test/` |

**Verified:** `task pipeline:images-test` rendered 3 images in ~26 s (~8.5 s/image after the
one-time Qwen3 embedding precompute). Embeddings cached for reruns.

> First full run downloads ~16 GB (transformer 7.75 GB + Qwen3 8 GB + VAE/scheduler);
> everything is cached afterwards. Budget for 1000 images: ~9 min Qwen3 embeds (once) +
> ~8 s/image → roughly 2.5 h of GPU time.

---

## Step 3 — Label images  (`pipeline/label_images.py`)

Per-wrench bounding boxes from **Segment Anything 3 (SAM 3)**, run locally as an
open-vocabulary text-grounded segmenter. SAM 3 is given the text prompt `"wrench"` and
returns, for every image, instance masks + bounding boxes + confidence scores for the
wrenches it finds. We keep the boxes (single class, id 0 = `"wrench"`) and write them in
YOLO format (`0 x_center y_center width height`, normalized) as one `.txt` per image. A
`labels.jsonl` audit log records the raw pixel boxes and scores per image.

Why local SAM 3 instead of a remote VLM:
- **No API key / network** — the whole pipeline runs offline on the 6 GB GPU.
- SAM 3 text grounding gives tight, instance-level boxes for an open vocabulary, which is
  exactly what a single-class "wrench" detector needs.

Instance **masks** are also persisted (one `masks/<stem>.png` per image, a 16-bit PNG
where pixel value = instance id, 0 = background) so step 5 can overlay the segmentation
without reloading SAM 3.

Model weights: `facebook/sam3` is **gated**; we use the open mirror `1038lab/sam3`
(`sam3.pt`, ~3.45 GB), cached at `models/sam3/sam3.pt` by `task env:fetch-assets`. The
`sam3` pip package's text tokenizer also needs the CLIP `bpe_simple_vocab_16e6.txt.gz`
file, which the package does not ship — `task env:fetch-assets` drops it into
`.venv/.../assets/`.

GPU budget on the RTX 2060 (6 GB): SAM 3 ViT-L backbone @ resolution 1008 fits in ~3.5 GB
of weights; inference activations are modest (verified end-to-end). `--resolution 768` can
be used to be safe.

Configuration via CLI flags (no secrets anywhere):
- `--model-path` — default `models/sam3/sam3.pt`
- `--prompt` — default `wrench`
- `--confidence` — default `0.55` (boxes below this are dropped, both here and in step 5)
- `--resolution` — default `1008`

| Task | Command | Output |
|------|---------|--------|
| `task pipeline:label` | all images | `data/labels/` |
| `task pipeline:label-test` | first 3 test images | `data/labels_test/` |

**Verified:** SAM 3 loads in ~10 s (3.46 GB VRAM) and, on a sample image, returns tight,
well-scored boxes for the text prompt. `task pipeline:label-test` produces YOLO `.txt`
files + `labels.jsonl` for 3 images with no API key.

> TODO before labeling all 1000: eyeball a few `labels.jsonl` entries + draw the boxes (see
> `experiments/scripts/draw_yolo_boxes.py`) and tune `--confidence` (default 0.55) if boxes are
> too noisy/missing.

---

## Step 4 — Fine-tune YOLO  (`pipeline/train_yolo.py`)

Assembles a YOLO dataset from `data/images` + `data/labels` (paired by stem, 80/20 train/val
split, at least 1 val image), writes `data.yaml` with a **single class** `0: wrench`, and
fine-tunes `yolov8n.pt` via `ultralytics` to recognize only wrenches. Images/labels are
symlinked into the dataset dir (no copies).

| Task | Command | Output |
|------|---------|--------|
| `task pipeline:train` | 50 epochs, 640px, batch 16 | `data/dataset/runs/train/weights/best.pt` |
| `task pipeline:train-test` | 1 epoch, 320px, batch 1, `--allow-dummy-labels` | `data/dataset_test/runs/train/weights/best.pt` |

The `--allow-dummy-labels` flag writes a placeholder centered box (class 0) for any image
with no real label, **purely so the smoke test can run before step 3 has produced labels**.
Never use it for a real run. (With SAM 3 labeling in place, the smoke test runs on real
labels; the flag is a harmless fallback.)

**Verified:** `task pipeline:train-test` assembled a 3-image dataset (2 train / 1 val) and
completed 1 epoch on the GPU, producing `best.pt` / `last.pt` under
`data/dataset_test/runs/train/`.

---

## Step 5 — Visualize labels  (`pipeline/visualize_labels.py`)

A pure post-processing step: for each image it renders a copy with the **SAM 3
segmentation overlaid** (semi-transparent colored regions, one color per instance) and the
**bounding boxes drawn** on top with the confidence score annotated. No model is loaded —
nothing leaves the CPU — so it is cheap to run on the whole set.

Inputs are the step-2 image + the step-3 `labels.jsonl` (which carries `boxes_xyxy` +
`scores`) + the instance-mask PNG written by step 3 under `masks/`. Outputs go to
`data/visuals/img_XXXX.png` plus a `visuals.jsonl` index.

Which boxes are drawn? By default the same boxes written to the YOLO `.txt` labels (SAM 3's
`pred_boxes` — the actual training targets), so the visualization shows exactly what the
detector will learn. Pass `--boxes-from masks` to instead draw the tight axis-aligned bbox
computed from each segmentation mask's pixels (a literal "box-based-on-the-segmentation"
view), which is useful for spotting over/under-segmentation.

Boxes below **0.55 confidence** are discarded (`--min-confidence`, default 0.55) — the same
floor step 3 applies — and their corresponding mask instances are removed from the overlay
so the segmentation and the drawn boxes always agree.

| Task | Command | Output |
|------|---------|--------|
| `task pipeline:visualize` | all images | `data/visuals/` |
| `task pipeline:visualize-test` | first 3 test images | `data/visuals_test/` |

**Verified:** `task pipeline:visualize-test` renders 3 overlaid PNGs (mask regions +
boxes + scores) referencing the instance masks saved by step 3.

---

## Running the whole pipeline

```bash
task env:setup                       # one-time: create .venv, install numeri editable,
                                     # fetch SAM 3 checkpoint + CLIP BPE vocab
# Step 1
task pipeline:generate               # 1000 wrench prompts -> data/prompts.txt
# Step 2
task pipeline:images                 # FLUX.2-klein -> data/images/      (~2.5 h GPU)
# Step 3
task pipeline:label                  # local SAM 3 -> data/labels/
# Step 4
task pipeline:train                  # fine-tune single-class YOLO -> data/dataset/runs/train/weights/best.pt
```

Run each `*-test` task first to verify a step before the full run:

```bash
task pipeline:test
task pipeline:images-test
task pipeline:label-test
task pipeline:train-test
task pipeline:visualize-test
```

## Repository layout

```
pipeline/                 # the active 4-step pipeline
  generate_prompts.py     # step 1  (wrench-scene prompts)
  generate_images.py      # step 2  (FLUX.2-klein-4B, NF4, 6 GB GPU)
  label_images.py         # step 3  (SAM 3 text grounding -> single-class boxes + masks)
  train_yolo.py           # step 4  (assemble dataset + ultralytics fine-tune, class 0=wrench)
  visualize_labels.py     # step 5  (overlay segmentation + boxes -> image copies)
tasks/pipeline.yml        # go-task definitions for all 8 tasks
tasks/env.yml             # env:setup + env:fetch-assets (SAM 3 ckpt + BPE vocab)
PLAN.md                   # this file
data/                     # gitignored — all pipeline outputs
models/                   # gitignored — cached base weights (yolov8n.pt, sam3/sam3.pt)
experiments/              # all exploratory probes live here (try_flux2.py, try_qwen25vl.py, ...)
  scripts/                #   helper scripts (draw_grid, draw_yolo_boxes, auto_digit_boxes)
  weights/                #   experiment weights (yolov8-worldv2, clip)
  notes/                  #   notes (manual_bbox_grid.md)
  out/                    #   gitignored experiment outputs
numeri/                   # generic, model-agnostic framework scaffold (kept for reuse; see SPECIFICATIONS.md)
```

## Open items

- Tune step 3 `--confidence` on a few rendered images before labeling all 1000.
- After step 3, optionally split a held-out real test set of wrench photos to report honest
  mAP; the `numeri/eval` scaffold + `task eval:*` can be reused for this.
- Tune step 4 hyperparameters (epochs, imgsz, augmentation) once real labels exist.
