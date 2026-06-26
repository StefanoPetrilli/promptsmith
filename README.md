# Numeri

A reusable, **model-agnostic**, fully-local pipeline that:

1. **Generates** a synthetic image dataset with small models (typically diffusion),
2. **Auto-labels** it incl. YOLO bounding boxes with small vision-language models,
3. **Mixes** it with a real dataset, and
4. **Fine-tunes** a small YOLO detector and compares against a real-only baseline.

Validated on the **Street View House Numbers (SVHN)** dataset. See
[`SPECIFICATIONS.md`](SPECIFICATIONS.md) for the full design.

## Requirements
- Python 3.10+
- NVIDIA GPU recommended (~6 GB VRAM enough for small backends). All backends fall back to
  CPU but slowly.
- [go-task](https://taskfile.dev) 3.x — install:
  ```bash
  sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
  ```

## Quickstart
```bash
task env:setup          # create .venv, install numeri editable, sanity-check device
task dataset:download   # fetch SVHN
task dataset:parse      # convert to YOLO format under data/real_yolo/

# Configure backends: copy config/default.yaml -> config/local.yaml
# and set generation.model.name / labeling.model.name (kept out of defaults).
task image-generation:setup
task image-generation:start

task labeling:setup
task labeling:start

task mix:build
task train:baseline     # real-only
task train:run          # real + synthetic
task eval:compare
```

Each task is a thin wrapper around the `numeri` Python package; you can also run
`python -m numeri <stage> [--config config/local.yaml]`.

## Configuration
All behavior is driven by `config/*.yaml`. Backends are referenced **by name** and resolved
through `numeri.registry`, so switching models or use cases requires no code changes —
just config (and, for a brand-new backend, registering one class).

| Area | Key | What it selects |
|------|-----|-----------------|
| dataset | `dataset.name` | `numeri.datasets.<name>` adapter |
| generation | `generation.backend` + `generation.prompt_template` | `numeri.generation.<backend>`, prompt schema |
| labeling | `labeling.backend` | `numeri.labeling.<backend>` |
| yolo | `yolo.version` | base weights / nano model |

Files under `data/` and `models/` are gitignored (regenerable).

## Project layout
See the diagram in [`SPECIFICATIONS.md`](SPECIFICATIONS.md). Code lives in `numeri/`; task
definitions in `tasks/*.yml`; entry root in `Taskfile.yml`.