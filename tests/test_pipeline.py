"""End-to-end smoke test on a tiny synthetic dataset:
folder data -> train (random init, 2 epochs) -> evaluate -> export + quantise."""
import json

import numpy as np
from PIL import Image

from src import evaluate, export_onnx, train
from src.prepare_data import Writer, class_from_path


def synth(kind, rng):
    if kind == "good":  # regular tread-like stripes
        a = np.zeros((160, 200, 3), np.uint8)
        a[::12] = 200
        a[1::12] = 200
    elif kind == "defective":  # noise
        a = rng.integers(0, 255, (160, 200, 3), dtype=np.uint8)
    else:  # flat colour
        a = np.full((160, 200, 3), rng.integers(0, 255, 3), np.uint8)
    return Image.fromarray(a)


def test_class_from_path(tmp_path):
    m = {"defective": "defective", "good": "good"}
    root = tmp_path
    assert class_from_path(root / "Digital images of defective and good tyres/good/x.jpg", root, m) == "good"
    assert class_from_path(root / "set/Defective/x.jpg", root, m) == "defective"
    assert class_from_path(root / "other/x.jpg", root, m) is None


def test_writer_skips_duplicates(tmp_path):
    w = Writer(tmp_path, ["good", "defective"])
    img = synth("defective", np.random.default_rng(1))
    w.add(img, "good", "a")
    w.add(img.copy(), "good", "b")
    w.add(img, "not_a_class", "c")
    assert w.counts["good"] == 1 and w.dupes == 1


def test_train_evaluate_export(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "condition"
    for cls in ("good", "defective", "not_tyre"):
        (data / cls).mkdir(parents=True)
        for i in range(14):
            synth(cls, rng).save(data / cls / f"{i}.jpg")

    ckpts, reports, web = tmp_path / "ckpt", tmp_path / "reports", tmp_path / "web"
    train.main(["--task", "condition", "--data", str(data), "--out", str(ckpts), "--epochs", "2",
                "--freeze-epochs", "1", "--batch", "8", "--workers", "0", "--no-pretrained"])
    ckpt = ckpts / "condition" / "best.pt"
    assert ckpt.exists()

    evaluate.main(["--ckpt", str(ckpt), "--data", str(data), "--reports", str(reports)])
    metrics = json.loads((reports / "condition_metrics.json").read_text())
    assert metrics["classes"] == ["good", "defective", "not_tyre"]
    assert (reports / "condition_confusion.png").exists()

    export_onnx.main(["--ckpt", str(ckpt), "--data", str(data), "--out", str(web)])
    meta = json.loads((web / "condition.json").read_text())
    assert (web / meta["file"]).exists()
    assert meta["placeholder"] is False
    assert not list(web.glob("*.fp32.onnx"))  # only the shipped model remains
