"""Export a checkpoint to ONNX for the browser, quantise to INT8, and write the
model metadata the web app reads (web/models/<task>.json).

    python -m src.export_onnx --ckpt checkpoints/condition/best.pt --data data/condition

Steps: fp32 export (outputs: logits, cam) -> parity check vs PyTorch -> static INT8
(QDQ, per-channel) quantisation calibrated on validation images -> top-1 agreement
check on the test split. The INT8 model is shipped only if agreement >= --min-agree,
otherwise the fp32 model is shipped.

--placeholder exports an untrained model so the web app can be tested end to end
before real training has happened.
"""
import argparse
import hashlib
import json
import random
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .config import ARCH, IMG_SIZE, MEAN, STD, TASKS
from .dataset import eval_tensor, load_rgb, make_splits
from .model import ExportWrapper, build_model, load_checkpoint

OPSET = 17


def export_fp32(wrapper, path: Path):
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with warnings.catch_warnings():
        # The TorchScript exporter is deprecated but produces the simplest graph for ORT-web.
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            wrapper, dummy, str(path), opset_version=OPSET, dynamo=False,
            input_names=["input"], output_names=["logits", "cam"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}},
        )


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def check_parity(wrapper, onnx_path, batches, atol=1e-3):
    sess = session(onnx_path)
    worst = 0.0
    with torch.no_grad():
        for x in batches:
            t_logits, t_cam = wrapper(torch.from_numpy(x))
            o_logits, o_cam = sess.run(None, {"input": x})
            worst = max(worst, np.abs(t_logits.numpy() - o_logits).max(), np.abs(t_cam.numpy() - o_cam).max())
    if worst > atol:
        raise SystemExit(f"ONNX parity check failed: max abs diff {worst:.2e} > {atol}")
    print(f"parity OK (max abs diff {worst:.2e})")


class Reader(CalibrationDataReader):
    def __init__(self, batches):
        self.it = iter([{"input": x} for x in batches])

    def get_next(self):
        return next(self.it, None)


def quantize(fp32_path: Path, int8_path: Path, calib_batches):
    pre = fp32_path.with_suffix(".preproc.onnx")
    quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
    quantize_static(
        str(pre), str(int8_path), Reader(calib_batches),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
    )
    pre.unlink(missing_ok=True)


def top1_agreement(path_a, path_b, batches):
    a, b = session(path_a), session(path_b)
    same = total = 0
    for x in batches:
        pa = a.run(["logits"], {"input": x})[0].argmax(1)
        pb = b.run(["logits"], {"input": x})[0].argmax(1)
        same += int((pa == pb).sum())
        total += len(pa)
    return same / total


def image_batches(paths, bs=16):
    for i in range(0, len(paths), bs):
        yield torch.stack([eval_tensor(load_rgb(p)) for p in paths[i:i + bs]]).numpy()


def random_batches(n, bs=8, seed=0):
    g = np.random.default_rng(seed)
    return [g.standard_normal((bs, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32) for _ in range(n)]


def write_meta(path: Path, entry: dict):
    path.write_text(json.dumps(entry, indent=2) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint from src.train")
    ap.add_argument("--data", help="data/<task> dir (for calibration + agreement check)")
    ap.add_argument("--out", default="../web/models")
    ap.add_argument("--min-agree", type=float, default=0.98)
    ap.add_argument("--calib", type=int, default=200, help="number of calibration images")
    ap.add_argument("--placeholder", choices=TASKS, help="export an UNTRAINED model for this task")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.placeholder:
        task, classes = args.placeholder, TASKS[args.placeholder]
        torch.manual_seed(0)
        model = build_model(len(classes), pretrained=False).eval()
        calib, check = random_batches(4), random_batches(4, seed=1)
        metrics = None
    else:
        model, ckpt = load_checkpoint(args.ckpt)
        task, classes = ckpt["task"], ckpt["classes"]
        splits = make_splits(Path(args.data), None)
        val_paths = list(splits["val"][0])
        random.Random(0).shuffle(val_paths)
        calib = list(image_batches(val_paths[: args.calib]))
        check = list(image_batches(splits["test"][0]))
        metrics = {"val_f1": round(ckpt.get("val_f1", 0.0), 4)}

    wrapper = ExportWrapper(model).eval()
    fp32 = out / f"{task}.fp32.onnx"
    int8 = out / f"{task}.onnx"
    export_fp32(wrapper, fp32)
    check_parity(wrapper, fp32, check[:4])
    quantize(fp32, int8, calib)
    agree = top1_agreement(fp32, int8, check)
    print(f"INT8 vs fp32 top-1 agreement: {agree:.3f}")

    quantized = agree >= args.min_agree or bool(args.placeholder)
    if quantized:
        fp32.unlink()
    else:
        print(f"agreement < {args.min_agree}: shipping fp32 model instead")
        int8.unlink()
        fp32.rename(int8)
    shipped = int8

    write_meta(out / f"{task}.json", {
        "task": task,
        "file": shipped.name,
        "sha256": hashlib.sha256(shipped.read_bytes()).hexdigest()[:16],
        "labels": classes,
        "input_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "arch": ARCH,
        "quantized": quantized,
        "int8_agreement": round(agree, 4),
        "placeholder": bool(args.placeholder),
        "metrics": metrics,
    })
    print(f"wrote {shipped} ({shipped.stat().st_size / 1e6:.1f} MB) and {task}.json")


if __name__ == "__main__":
    main()
