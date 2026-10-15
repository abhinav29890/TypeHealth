"""Fine-tune EfficientNet-B0 on one task.

    python -m src.train --task condition --data data/condition --epochs 25
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .config import ARCH, IMG_SIZE, TASKS
from .dataset import TyreDataset, make_splits
from .model import build_model, classifier_params


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        total_loss += loss.item() * len(y)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / len(targets),
        "acc": float((preds == targets).mean()),
        "f1": float(f1_score(targets, preds, average="macro")),
    }


def cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--freeze-epochs", type=int, default=2, help="train only the head first")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    splits = make_splits(Path(args.data), TASKS[args.task], seed=args.seed)
    classes = splits["classes"]
    print(f"classes: {classes}")
    for name in ("train", "val", "test"):
        counts = np.bincount(splits[name][1], minlength=len(classes))
        print(f"  {name:5s}: {dict(zip(classes, counts.tolist()))}")

    train_dl = DataLoader(TyreDataset(*splits["train"], train=True), batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    val_dl = DataLoader(TyreDataset(*splits["val"], train=False), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    counts = np.bincount(splits["train"][1], minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(classes) * counts), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path(args.out) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, bad_epochs, history = -1.0, 0, []

    phases = [("head", args.freeze_epochs, 1e-3), ("full", args.epochs - args.freeze_epochs, args.lr)]
    epoch = 0
    for phase, n_epochs, lr in phases:
        if n_epochs <= 0:
            continue
        for p in model.parameters():
            p.requires_grad = phase == "full"
        for p in classifier_params(model):
            p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        steps = n_epochs * len(train_dl)
        scheduler = cosine_with_warmup(optimizer, steps, warmup_steps=min(len(train_dl), steps // 10 + 1))
        bad_epochs = 0

        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            tr = run_epoch(model, train_dl, device, criterion, optimizer, scheduler, scaler)
            va = run_epoch(model, val_dl, device, criterion)
            history.append({"epoch": epoch, "phase": phase, "train": tr, "val": va})
            print(f"[{phase}] epoch {epoch:2d}  train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
                  f"val loss {va['loss']:.3f} acc {va['acc']:.3f} f1 {va['f1']:.3f}  ({time.time() - t0:.0f}s)")
            if va["f1"] > best_f1:
                best_f1, bad_epochs = va["f1"], 0
                torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": ARCH,
                            "img_size": IMG_SIZE, "task": args.task, "val_f1": best_f1},
                           out_dir / "best.pt")
            else:
                bad_epochs += 1
                if phase == "full" and bad_epochs >= args.patience:
                    print("early stopping")
                    break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val macro-F1: {best_f1:.3f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
