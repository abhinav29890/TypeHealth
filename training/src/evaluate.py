"""Evaluate a checkpoint on the held-out test split and write a report.

    python -m src.evaluate --ckpt checkpoints/condition/best.pt --data data/condition
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import eval_tensor, load_rgb, make_splits
from .model import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def predict(model, paths, batch=32, device="cpu"):
    model.to(device)
    probs = []
    for i in range(0, len(paths), batch):
        x = torch.stack([eval_tensor(load_rgb(p)) for p in paths[i:i + batch]]).to(device)
        probs.append(model(x).softmax(1).cpu())
    return torch.cat(probs).numpy()


def save_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(4 + len(classes), 3.5 + len(classes) * 0.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args(argv)

    model, ckpt = load_checkpoint(args.ckpt)
    classes = ckpt["classes"]
    splits = make_splits(Path(args.data), None)
    paths, labels = splits[args.split]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs = predict(model, paths, device=device)
    preds = probs.argmax(1)

    report = classification_report(labels, preds, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    print(classification_report(labels, preds, labels=range(len(classes)), target_names=classes, zero_division=0))

    out = Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    task = ckpt["task"]
    (out / f"{task}_metrics.json").write_text(json.dumps(
        {"task": task, "split": args.split, "n": len(labels), "classes": classes,
         "report": report, "confusion_matrix": cm.tolist()}, indent=2))
    save_confusion(cm, classes, out / f"{task}_confusion.png")
    print(f"macro-F1 {report['macro avg']['f1-score']:.3f}  -> {out}/{task}_metrics.json")


if __name__ == "__main__":
    main()
