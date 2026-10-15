import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.config import IMG_SIZE
from src.export_onnx import export_fp32
from src.model import ExportWrapper, build_model


@pytest.fixture(scope="module")
def wrapper():
    torch.manual_seed(0)
    model = build_model(3, pretrained=False).eval()
    with torch.no_grad():  # make the head non-trivial so differences would show
        model.get_classifier().weight.normal_(0, 0.5)
        model.get_classifier().bias.normal_(0, 0.5)
    return ExportWrapper(model).eval()


def test_wrapper_logits_match_model(wrapper):
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        logits, cam = wrapper(x)
        expected = wrapper.model(x)
    assert torch.allclose(logits, expected, atol=1e-4)
    assert cam.shape == (2, 3, IMG_SIZE // 32, IMG_SIZE // 32)
    # CAM is the head applied per position: its spatial mean + bias equals the logits.
    assert torch.allclose(cam.mean(dim=(2, 3)) + wrapper.b, logits, atol=1e-4)


def test_onnx_matches_pytorch(wrapper, tmp_path):
    path = tmp_path / "m.onnx"
    export_fp32(wrapper, path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input"]
    assert [o.name for o in sess.get_outputs()] == ["logits", "cam"]

    x = np.random.default_rng(0).standard_normal((3, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    o_logits, o_cam = sess.run(None, {"input": x})
    with torch.no_grad():
        t_logits, t_cam = wrapper(torch.from_numpy(x))
    np.testing.assert_allclose(o_logits, t_logits.numpy(), atol=1e-3)
    np.testing.assert_allclose(o_cam, t_cam.numpy(), atol=1e-3)


def test_shipped_models_json_is_consistent():
    from pathlib import Path

    models_dir = Path(__file__).resolve().parents[1] / "web" / "models"
    for task in ("condition", "tread"):
        entry = json.loads((models_dir / f"{task}.json").read_text())
        assert entry["task"] == task
        assert (models_dir / entry["file"]).exists(), task
        assert entry["input_size"] == IMG_SIZE
        assert len(entry["labels"]) >= 2
        if task == "condition":
            assert "not_tyre" in entry["labels"]  # scoring.js relies on it
