"""Download public tyre datasets and merge them into data/<task>/<class>/*.jpg.

Run from the training/ directory:
    python -m src.prepare_data --task condition
    python -m src.prepare_data --task tread

Sources are listed in sources.json. Images are converted to RGB JPEG (max side 512),
and near-identical duplicates across sources are dropped (perceptual hash) so the
same photo cannot leak into both train and test splits.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import IMG_EXTS, TASKS

MAX_SIDE = 512
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# Imagenette classes that often contain tyres/vehicles: garbage truck, gas pump.
IMAGENETTE_EXCLUDE = {"n03417042", "n03425413"}


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p


def class_from_path(path: Path, root: Path, mapping: dict):
    """Match folder names (deepest first) against the mapping keys."""
    parts = [s.lower() for s in path.relative_to(root).parts[:-1]]
    for part in reversed(parts):
        if part in mapping:
            return mapping[part]
    for part in reversed(parts):
        for key, target in mapping.items():
            if key in part:
                return target
    return None


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(g, dtype=np.int16).ravel().tolist()
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


class Writer:
    """Writes normalised images into data/<task>/<class>/ and skips duplicates."""

    def __init__(self, out_dir: Path, classes: list):
        self.out_dir = out_dir
        self.classes = classes
        self.seen = set()
        self.counts = Counter()
        self.dupes = 0
        for c in classes:
            (out_dir / c).mkdir(parents=True, exist_ok=True)

    def add(self, img: Image.Image, target: str, source: str):
        if target not in self.classes:
            return
        img = ImageOps.exif_transpose(img).convert("RGB")
        if min(img.size) < 32:
            return
        h = dhash(img)
        if h in self.seen:
            self.dupes += 1
            return
        self.seen.add(h)
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(self.out_dir / target / f"{source}_{h:016x}.jpg", quality=90)
        self.counts[target] += 1


def _open(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None


def load_folder_tree(root: Path, mapping: dict, writer: Writer, name: str):
    for p in iter_images(root):
        target = class_from_path(p, root, mapping)
        img = _open(p) if target else None
        if img is not None:
            writer.add(img, target, name)


def src_kaggle(cfg, writer, raw):
    dest = raw / cfg["name"]
    if not dest.exists():
        subprocess.run(["kaggle", "datasets", "download", "-d", cfg["slug"],
                        "-p", str(dest), "--unzip"], check=True)
    load_folder_tree(dest, cfg["map"], writer, cfg["name"])


def src_folder(cfg, writer, raw):
    root = Path(cfg["path"])
    if not root.exists():
        print(f"  (skipping: {root} does not exist)")
        return
    load_folder_tree(root, {c: c for c in writer.classes}, writer, cfg["name"])


def src_imagenette(cfg, writer, raw):
    tgz = raw / "imagenette2-160.tgz"
    dest = raw / "imagenette2-160"
    if not dest.exists():
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(raw)
    files = [p for p in iter_images(dest) if p.parent.name not in IMAGENETTE_EXCLUDE]
    random.Random(0).shuffle(files)
    for p in files[: cfg["count"]]:
        img = _open(p)
        if img is not None:
            writer.add(img, cfg["target"], cfg["name"])


def src_huggingface(cfg, writer, raw):
    from datasets import load_dataset

    ds = load_dataset(cfg["repo"], split=cfg.get("split", "train"))
    label_feat = ds.features[cfg["label_col"]]
    for row in ds:
        label = row[cfg["label_col"]]
        if hasattr(label_feat, "int2str"):
            label = label_feat.int2str(label)
        target = cfg["map"].get(str(label).lower())
        if target:
            writer.add(row[cfg["image_col"]], target, cfg["name"])


def src_roboflow(cfg, writer, raw):
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("  (skipping: set ROBOFLOW_API_KEY to use Roboflow sources)")
        return
    dest = raw / cfg["name"]
    project = Roboflow(api_key=key).workspace(cfg["workspace"]).project(cfg["project"])
    fmt = "folder" if cfg["kind"] == "classification" else "yolov8"
    if not dest.exists():
        project.version(cfg["version"]).download(fmt, location=str(dest))
    if cfg["kind"] == "classification":
        load_folder_tree(dest, cfg["map"], writer, cfg["name"])
    else:
        crop_yolo_boxes(dest, cfg["map"], writer, cfg["name"])


def crop_yolo_boxes(root: Path, mapping: dict, writer: Writer, name: str, pad: float = 0.1):
    """Turn a YOLO detection export into classification crops (one crop per box)."""
    import yaml

    meta = yaml.safe_load((root / "data.yaml").read_text())
    names = [n.lower() for n in meta["names"]]
    for img_path in iter_images(root):
        label_path = Path(str(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
        img = _open(img_path) if label_path.exists() else None
        if img is None:
            continue
        img = ImageOps.exif_transpose(img).convert("RGB")
        W, H = img.size
        for line in label_path.read_text().splitlines():
            vals = line.split()
            if len(vals) != 5:  # skip polygon labels
                continue
            cls, cx, cy, bw, bh = int(vals[0]), *map(float, vals[1:])
            target = mapping.get(names[cls])
            if not target:
                continue
            bw, bh = bw * (1 + pad), bh * (1 + pad)
            box = (max(0, (cx - bw / 2) * W), max(0, (cy - bh / 2) * H),
                   min(W, (cx + bw / 2) * W), min(H, (cy + bh / 2) * H))
            if min(box[2] - box[0], box[3] - box[1]) >= 48:
                writer.add(img.crop(box), target, name)


SOURCES = {
    "kaggle": src_kaggle,
    "folder": src_folder,
    "imagenette": src_imagenette,
    "huggingface": src_huggingface,
    "roboflow": src_roboflow,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--sources", default="sources.json")
    ap.add_argument("--out", default="data")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--clean", action="store_true", help="delete existing data/<task> first")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) / args.task
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)

    writer = Writer(out_dir, TASKS[args.task])
    for cfg in json.loads(Path(args.sources).read_text())[args.task]:
        if not cfg.get("enabled", True):
            continue
        print(f"[{cfg['type']}] {cfg['name']}")
        before = sum(writer.counts.values())
        try:
            SOURCES[cfg["type"]](cfg, writer, raw)
        except Exception as e:  # one broken source should not kill the whole run
            print(f"  FAILED: {e}")
        print(f"  +{sum(writer.counts.values()) - before} images")

    (out_dir / "splits.json").unlink(missing_ok=True)  # data changed -> re-split on next train
    summary = {"task": args.task, "counts": dict(writer.counts), "duplicates_skipped": writer.dupes}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    empty = [c for c in TASKS[args.task] if writer.counts[c] == 0]
    if empty:
        print(f"WARNING: no images for {empty}. These classes will be dropped from training "
              f"unless you add photos under data/own/{args.task}/<class>/.")


if __name__ == "__main__":
    main()
