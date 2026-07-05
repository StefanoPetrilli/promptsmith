# Manual webcam test

Self-contained folder to run the trained wrench detector on a laptop webcam.

## Contents

- `best.pt` — trained YOLO weights (copy this from the server).
- `manual_test.py` — webcam inference script.
- `requirements.txt` — minimal Python dependencies.
- `Taskfile.yml` — task runner definitions.

## Usage

1. Copy this whole folder to your laptop.
2. Install [go-task](https://taskfile.dev) if you don't have it.
3. Run:

   ```bash
   cd manual_test
   task run
   ```

The first run creates a `.venv` and installs dependencies. Subsequent runs reuse it.

Press `q` or `ESC` to close the webcam window.

## Options

Override defaults via the task:

```bash
task run -- --conf 0.5 --source 1 --device cpu
```

Or call the script directly after setting up the venv:

```bash
.venv/bin/python manual_test.py --model best.pt --conf 0.25
```
