"""Folder dataset, stratified split and the train/eval transforms.

The eval preprocessing (centre square crop -> 224x224 -> ImageNet normalise) is
mirrored exactly by web/js/preprocess.js.
"""
import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import IMG_EXTS, IMG_SIZE, MEAN, STD


def scan(data_dir: Path, classes: list):
    """Return (paths, labels, used_classes); classes with no images are dropped."""
    used = [c for c in classes
            if (data_dir / c).is_dir() and any(p.suffix.lower() in IMG_EXTS for p in (data_dir / c).iterdir())]
    paths, labels = [], []
    for idx, c in enumerate(used):
        for p in sorted((data_dir / c).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(idx)
    return paths, labels, used


def make_splits(data_dir: Path, classes: list, seed: int = 42, val: float = 0.15, test: float = 0.15):
    """Stratified 70/15/15 split, cached to data_dir/splits.json so every script agrees.

    Pass classes=None to only read the cached split (used by evaluate/export).
    """
    cache = data_dir / "splits.json"
    if cache.exists():
        s = json.loads(cache.read_text())
        if classes is None or s["classes_requested"] == classes:
            return s
    if classes is None:
        raise SystemExit(f"No {cache}; run training first.")
    paths, labels, used = scan(data_dir, classes)
    if len(used) < 2:
        raise SystemExit(f"Need images for at least 2 classes in {data_dir}, found {used}")
    tr_p, rest_p, tr_y, rest_y = train_test_split(
        paths, labels, test_size=val + test, stratify=labels, random_state=seed)
    va_p, te_p, va_y, te_y = train_test_split(
        rest_p, rest_y, test_size=test / (val + test), stratify=rest_y, random_state=seed)
    s = {"classes_requested": classes, "classes": used,
         "train": [tr_p, tr_y], "val": [va_p, va_y], "test": [te_p, te_y]}
    cache.write_text(json.dumps(s))
    return s


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    return img.crop((left, top, left + m, top + m))


def load_rgb(path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def train_transform():
    return A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        A.HorizontalFlip(),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=25, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02, p=0.7),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
        A.GaussNoise(p=0.1),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def eval_tensor(img: Image.Image) -> torch.Tensor:
    """Deterministic preprocessing, identical to the browser."""
    img = center_square(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class TyreDataset(Dataset):
    def __init__(self, paths, labels, train: bool):
        self.paths, self.labels, self.train = paths, labels, train
        self.tf = train_transform() if train else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        if self.train:
            x = self.tf(image=np.asarray(img))["image"]
        else:
            x = eval_tensor(img)
        return x, self.labels[i]
