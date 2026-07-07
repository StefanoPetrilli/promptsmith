#!/usr/bin/env python3
"""Step 5 — hand-pick review of segmented images.

Some FLUX renders hallucinate wrenches that SAM 3 then segments poorly. This
stage opens a small browser interface that shows each segmentation overlay
produced in step 4 and asks a human to APPROVE or DISCARD it. Approved originals
(plus their labels and instance masks) are symlinked into the approved tree;
discarded ones into the discarded tree. A decisions file makes the review
resumable. Downstream stages (postprocess, train) read only from the approved
tree.

data layout produced:
  <out-root>/images/<leaf>/<stem>.png        (symlink -> original render)
  <out-root>/images/<leaf>/manifest.jsonl    (filtered, for run_leaves/postprocess)
  <out-root>/labels/<leaf>/<stem>.txt        (symlink -> YOLO label)
  <out-root>/labels/<leaf>/masks/<stem>.png  (symlink -> SAM instance mask)
  <out-root>/visuals/<leaf>/<stem>.png       (symlink -> overlay, for reference)
  <out-root>/decisions.jsonl                 (image -> approved|discarded, resume log)

The UI is a tiny local web server (no extra deps beyond Pillow). Point a
browser at the printed URL. Controls:  ← / Backspace  discard,
→ / Enter  approve,  q  quit.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("handpick")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand-pick review of segmented images (step 5).")
    p.add_argument("--visuals-root", default="data/synthetic/visuals",
                   help="where step 4 wrote visuals.jsonl + overlay PNGs")
    p.add_argument("--images-root", default="data/synthetic/images",
                   help="original render tree (for manifest index/prompt + image symlinks)")
    p.add_argument("--labels-root", default="data/synthetic/labels",
                   help="SAM label tree (for .txt + masks symlinks)")
    p.add_argument("--out-root", default="data/synthetic/approved",
                   help="destination for approved images+labels")
    p.add_argument("--discard-root", default="data/synthetic/discarded",
                   help="destination for discarded images+labels")
    p.add_argument("--limit", type=int, default=0,
                   help="review only the first N visuals per leaf (test mode)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true",
                   help="do not try to open a browser automatically")
    return p.parse_args()


# --- collection --------------------------------------------------------------


def collect_visual_entries(visuals_root: Path, limit: int) -> list[dict]:
    """Gather visuals.jsonl entries, grouped by leaf, with a per-leaf limit."""
    by_leaf: dict[str, list[dict]] = {}
    for vf in sorted(visuals_root.rglob("visuals.jsonl")):
        leaf = vf.parent.relative_to(visuals_root).as_posix()
        rows = []
        for line in vf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        by_leaf[leaf] = rows
    entries = []
    for leaf, rows in sorted(by_leaf.items()):
        for r in (rows[:limit] if limit > 0 else rows):
            entries.append({**r, "leaf": leaf})
    return entries


def load_decisions(out_root: Path) -> dict[str, str]:
    """image_path -> decision, so already-reviewed images are skipped."""
    f = out_root / "decisions.jsonl"
    out: dict[str, str] = {}
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[d["image"]] = d.get("decision", "")
    return out


def append_decision(out_root: Path, image: str, visual: str, decision: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    f = out_root / "decisions.jsonl"
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"image": image, "visual": visual, "decision": decision}) + "\n")


def load_manifest_cache(images_root: Path) -> dict[str, dict]:
    """image_path -> {index, prompt} from the original render manifests."""
    cache: dict[str, dict] = {}
    if not images_root.is_dir():
        return cache
    for mf in images_root.rglob("manifest.jsonl"):
        for line in mf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "image" in d:
                cache[d["image"]] = {"index": d.get("index"), "prompt": d.get("prompt", "")}
    return cache


# --- sorting -----------------------------------------------------------------


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def manifest_has(manifest: Path, image_path: str) -> bool:
    if not manifest.exists():
        return False
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line).get("image") == image_path:
                return True
        except json.JSONDecodeError:
            continue
    return False


def sort_entry(
    entry: dict,
    images_root: Path,
    labels_root: Path,
    dest_root: Path,
    manifest_cache: dict[str, dict],
) -> None:
    """Symlink the original image + label + mask + visual into dest_root."""
    img = Path(entry["image"])
    leaf = entry["leaf"]
    stem = img.stem
    try:
        rel_img = img.relative_to(images_root)
    except ValueError:
        # absolute path or different root — fall back to <leaf>/<name>
        rel_img = Path(leaf) / img.name

    # original image
    dst_img = dest_root / "images" / rel_img
    if img.exists():
        safe_symlink(img, dst_img)
    else:
        log.warning("source image missing: %s", img)

    # YOLO label (.txt)
    rel_label = rel_img.with_suffix(".txt")
    src_label = labels_root / rel_label
    if src_label.exists():
        safe_symlink(src_label, dest_root / "labels" / rel_label)

    # SAM instance mask
    src_mask = labels_root / rel_img.parent / "masks" / f"{stem}.png"
    if src_mask.exists():
        safe_symlink(src_mask, dest_root / "labels" / rel_img.parent / "masks" / f"{stem}.png")

    # the segmented overlay (what the reviewer actually judged)
    if entry.get("visual"):
        vpath = Path(entry["visual"])
        if vpath.exists():
            safe_symlink(vpath, dest_root / "visuals" / leaf / f"{stem}.png")

    # filtered manifest entry for downstream run_leaves/postprocess
    manifest = dest_root / "images" / leaf / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_has(manifest, str(dst_img)):
        info = manifest_cache.get(str(img), {})
        rec = {
            "index": info.get("index"),
            "image": str(dst_img),
            "prompt": info.get("prompt", ""),
        }
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


# --- web UI ------------------------------------------------------------------


PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Hand-pick review</title>
<style>
  body {{ font-family: system-ui, sans-serif; text-align: center; margin: 0; background:#1e1e1e; color:#eee }}
  #info {{ font-size: 15px; font-weight:600; padding: 10px; }}
  #wrap {{ display:flex; justify-content:center; align-items:center; min-height:70vh }}
  img {{ max-width: 92vw; max-height: 80vh; border:2px solid #444; image-rendering:pixelated }}
  .bar {{ margin: 16px 0 40px; }}
  button {{ font-size: 17px; padding: 10px 26px; margin: 0 8px; border:none; border-radius:6px;
            cursor:pointer; font-weight:700 }}
  .discard {{ background:#c0392b; color:#fff }}
  .approve {{ background:#27ae60; color:#fff }}
  .quit {{ background:#555; color:#fff }}
  .hint {{ color:#888; font-size:12px; margin-top:10px }}
</style></head>
<body>
<div id="info">{info}</div>
<div id="wrap"><img id="img" src="{imgsrc}"></div>
<div class="bar">
  <button class="discard" onclick="decide('discarded')">Discard (←)</button>
  <button class="approve" onclick="decide('approved')">Approve (→)</button>
  <button class="quit" onclick="fetch('/quit',{{method:'POST'}}).then(()=>location.href='/')">Quit</button>
</div>
<div class="hint">{hint}</div>
<script>
  function decide(d) {{
    fetch('/decide?d='+d, {{method:'POST'}}).then(()=>location.reload());
  }}
  document.addEventListener('keydown', (e) => {{
    if (e.repeat) return;
    if (e.key === 'ArrowLeft' || e.key === 'Backspace') decide('discarded');
    else if (e.key === 'ArrowRight' || e.key === 'Enter') decide('approved');
    else if (e.key === 'q' || e.key === 'Escape') fetch('/quit',{{method:'POST'}}).then(()=>location.reload());
  }});
</script>
</body></html>
"""

DONE_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Done</title>
<style>body{{font-family:system-ui,sans-serif;text-align:center;margin-top:80px;background:#1e1e1e;color:#eee}}</style>
</head><body><h1>All reviewed.</h1><p>{msg}</p><p>You can close this tab.</p></body></html>
"""


class Reviewer:
    def __init__(
        self,
        entries: list[dict],
        out_root: Path,
        discard_root: Path,
        images_root: Path,
        labels_root: Path,
        manifest_cache: dict[str, dict],
        host: str,
        port: int,
        open_browser: bool,
    ) -> None:
        self.entries = entries
        self.idx = 0
        self.out_root = out_root
        self.discard_root = discard_root
        self.images_root = images_root
        self.labels_root = labels_root
        self.manifest_cache = manifest_cache
        self.reviewed = 0
        self.lock = threading.Lock()
        self.quit_flag = threading.Event()
        self.host = host
        self.port = port
        self.open_browser = open_browser

    # -- decision logic --

    def _current(self) -> dict | None:
        with self.lock:
            if self.idx >= len(self.entries):
                return None
            return self.entries[self.idx]

    def _decide(self, decision: str) -> None:
        with self.lock:
            if self.idx >= len(self.entries):
                return
            e = self.entries[self.idx]
            dest = self.out_root if decision == "approved" else self.discard_root
            dest.mkdir(parents=True, exist_ok=True)
            sort_entry(e, self.images_root, self.labels_root, dest, self.manifest_cache)
            append_decision(self.out_root, e["image"], e.get("visual", ""), decision)
            log.info("%s  %s  ->  %s", decision, e["image"], dest)
            self.reviewed += 1
            self.idx += 1

    def request_quit(self) -> None:
        self.quit_flag.set()

    # -- http --

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        reviewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # quieter
                pass

            def _send_html(self, code: int, html: str) -> None:
                b = html.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)

            def _send_image(self, path: Path) -> None:
                if not path.exists():
                    self._send_html(404, f"missing: {path}")
                    return
                # re-encode so huge PNGs display fast and we control size
                try:
                    buf = Image.open(path).convert("RGB")
                    buf.thumbnail((1400, 1400))
                    import io
                    out = io.BytesIO()
                    buf.save(out, format="JPEG", quality=88)
                    data = out.getvalue()
                    ct = "image/jpeg"
                except Exception as exc:  # fall back to raw bytes
                    log.warning("could not re-encode %s: %s", path, exc)
                    data = path.read_bytes()
                    ct = "image/png"
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    e = reviewer._current()
                    if e is None:
                        self._send_html(
                            200,
                            DONE_HTML.format(msg=f"Reviewed {reviewer.reviewed}/{len(reviewer.entries)} "
                                                 f"this session."),
                        )
                        return
                    info = (f"[{reviewer.idx + 1}/{len(reviewer.entries)}]  "
                            f"{e['leaf']}/{Path(e['image']).name}")
                    hint = (f"prompt: {reviewer.manifest_cache.get(str(Path(e['image'])), {}).get('prompt','')}"
                            .replace("{", "").replace("}", ""))
                    self._send_html(
                        200,
                        PAGE_HTML.format(
                            info=info,
                            imgsrc=f"/img?id={reviewer.idx}",
                            hint=hint,
                        ),
                    )
                    return
                if parsed.path == "/img":
                    qs = parse_qs(parsed.query)
                    try:
                        i = int(qs["id"][0])
                        e = reviewer.entries[i]
                    except (KeyError, IndexError, ValueError):
                        self._send_html(404, "bad id")
                        return
                    self._send_image(Path(e["visual"]))
                    return
                self._send_html(404, "not found")

            def do_POST(self):  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/decide":
                    qs = parse_qs(parsed.query)
                    d = (qs.get("d") or [""])[0]
                    if d in ("approved", "discarded"):
                        reviewer._decide(d)
                    self.send_response(204)
                    self.end_headers()
                    return
                if parsed.path == "/quit":
                    reviewer.request_quit()
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_response(404)
                self.end_headers()

        return Handler

    def run(self) -> None:
        server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        url = f"http://{self.host}:{self.port}"
        log.info("Hand-pick UI ready at  %s  (%d images queued)", url, len(self.entries))
        if self.open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            while not self.quit_flag.is_set():
                server.handle_request()
        finally:
            server.server_close()


# --- main --------------------------------------------------------------------


def main() -> int:
    a = parse_args()
    visuals_root = Path(a.visuals_root)
    images_root = Path(a.images_root)
    labels_root = Path(a.labels_root)
    out_root = Path(a.out_root)
    discard_root = Path(a.discard_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not visuals_root.is_dir():
        log.error("visuals root %s does not exist (run visualize first)", visuals_root)
        return 2

    entries = collect_visual_entries(visuals_root, a.limit)
    if not entries:
        log.error("no visuals.jsonl entries under %s", visuals_root)
        return 2

    decisions = load_decisions(out_root)
    todo: list[dict] = []
    for e in entries:
        if e["image"] in decisions:
            continue
        if not e.get("visual") or not Path(e["visual"]).exists():
            log.warning("missing overlay for %s — skipping", e.get("image"))
            continue
        todo.append(e)

    n_done = len(entries) - len(todo)
    log.info("visuals: %d total, %d already decided, %d to review", len(entries), n_done, len(todo))
    if not todo:
        log.info("nothing new to review — all decided.")
        return 0

    manifest_cache = load_manifest_cache(images_root)
    reviewer = Reviewer(
        todo, out_root, discard_root,
        images_root, labels_root, manifest_cache,
        host="127.0.0.1", port=a.port, open_browser=not a.no_browser,
    )
    try:
        reviewer.run()
    except KeyboardInterrupt:
        pass
    log.info("Reviewed %d/%d this session. Decisions log: %s",
             reviewer.reviewed, len(todo), out_root / "decisions.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
