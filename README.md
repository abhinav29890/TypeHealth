# TyreCheck: on-device tyre health from a photo

A web app for mechanics: take a photo of a tyre and get a health report covering **condition**
(good / defective, e.g. cracks, cuts, bulges) and **tread wear** (good / moderate / worn), with a
0–100 score, a heatmap of where the model looked, and plain-language recommendations.

It's **edge AI**: the neural networks run inside the browser (ONNX Runtime Web + WebAssembly).
Photos never leave the device, it works offline after the first visit, and it costs nothing to host.

```
 phone/PC browser ─────────────────────────────────────────────────────────────┐
 │ photo → quality check (blur, light) → centre crop 224² → condition model ─┐  │
 │                                                       → tread model ─────┤  │
 │                     health score + status + advice + CAM heatmap ◄────────┘  │
 └──────────────────────────────────────────────────────────────────────────────┘
 training (free Colab GPU): public datasets → EfficientNet-B0 fine-tune → ONNX → INT8
```

## Project layout

| Path | What |
|---|---|
| `web/` | The static app (no build step). `js/scoring.js` holds the health-score rules |
| `web/models/` | `<task>.onnx` + `<task>.json` metadata. **Ships with untrained placeholders** (the app shows "Demo mode") |
| `training/src/` | `prepare_data` → `train` → `evaluate` → `export_onnx` |
| `training/sources.json` | Which datasets to download and how their folders map to our classes |
| `training/notebooks/` | Colab notebooks: `02_train_condition`, `03_train_tread` (`01_explore_data` is optional) |
| `tests/` | Python (export parity, full pipeline smoke test) and JS (scoring/preprocessing) tests |

## Quick start: run the app locally

```bash
python -m http.server 8000 -d web
# open http://localhost:8000
```

To test on a phone on the same Wi-Fi, open `http://<pc-ip>:8000`. Analysis works over plain HTTP, but
offline mode and "install to home screen" need HTTPS (or localhost). The GitHub Pages deployment gives you HTTPS for free.

## Train the real models (free GPU)

1. Push this repo to GitHub.
2. Open `training/notebooks/02_train_condition.ipynb` in Google Colab
   (File → Open notebook → GitHub), pick **Runtime → T4 GPU**, set `REPO_URL`, add your free Kaggle
   API key as Colab secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`), and run all cells. That's about 15–25 minutes.
3. Download the zip it produces and unzip it at the project root. This replaces `web/models/condition.*`
   and adds `training/reports/condition_*`.
4. Repeat with `03_train_tread.ipynb`. It also needs a free Roboflow API key (`ROBOFLOW_API_KEY`).
   **Strongly recommended:** upload 100–200 of your own tread photos per class (see the notebook).
5. Commit and push. The app picks up the new models automatically (cache-busted by hash).

### Data sources

| Model | Source | Notes |
|---|---|---|
| condition | [Kaggle: Tyre Quality Classification](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification) | ~1.8k images, good / defective |
| condition | [Kaggle: Tire Texture Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition) | ~1k images, cracked / normal |
| condition | [Imagenette](https://github.com/fastai/imagenette) | ~1k random non-tyre photos for the `not_tyre` class |
| tread | [Roboflow: tyre-tread-detection](https://universe.roboflow.com/multichart/tyre-tread-detection-ulhub) | boxes cropped: `tread`→good, `tread-shallow`→worn. Check the version number and class names |
| both | your own photos | `training/data/own/<task>/<class>/*.jpg`, picked up automatically |

Add more sources (Kaggle, Roboflow, Hugging Face, local folders) by editing `training/sources.json`.
Duplicates across sources are removed with a perceptual hash so they can't leak between train and test.

### Training locally (CPU, slow but works)

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows (use bin/activate on Linux/macOS)
pip install -r training/requirements.txt
cd training
python -m src.prepare_data --task condition --clean
python -m src.train       --task condition --data data/condition --epochs 15 --workers 0
python -m src.evaluate    --ckpt checkpoints/condition/best.pt --data data/condition
python -m src.export_onnx --ckpt checkpoints/condition/best.pt --data data/condition
```

## How it works

- **Models:** two EfficientNet-B0 classifiers (ImageNet-pretrained, fine-tuned with class-weighted
  cross-entropy and label smoothing, AdamW + cosine LR, head-only warm-up, early stopping on val macro-F1).
- **Export:** ONNX with two outputs, `logits` and `cam` (the classifier applied at every 7×7 position,
  which gives class activation maps for free). It's then statically quantised to INT8 (QDQ, per-channel)
  using validation images, **only if** INT8 agrees with fp32 on ≥98% of the test set. About 5 MB per model.
- **Browser:** the same preprocessing as training (centre square crop → 224×224 → ImageNet normalise),
  so what you see in the app's square preview is exactly what the model analysed.
- **Score** (`web/js/scoring.js`): 60% condition + 40% expected tread level, capped at 40 if a defect
  is likely, 35 if the tread is likely worn, and 65 if it's likely moderate. Green ≥ 70, amber ≥ 45, red below.
  Photos that are probably not a tyre are rejected. Blurry or dark photos are flagged.
- **PWA:** a service worker caches the app, the models and the ONNX runtime, so it works offline and
  can be installed to the home screen. History is kept in the browser's local storage on that device only.

## Deploy (free)

Push to GitHub, then go to **Settings → Pages → Source: GitHub Actions**. `.github/workflows/pages.yml`
publishes `web/` on every push to `main`. `tests.yml` runs the test suites.

## Tests

```bash
node --test tests/scoring.test.mjs         # scoring, preprocessing, quality checks
pytest tests                               # ONNX parity + train→evaluate→export smoke test
```

## Limitations

- This is a **visual screening aid**, not a replacement for a physical inspection and a tread-depth gauge.
- Public tread-wear data is scarce, so expect the tread model to be the weaker of the two until it's
  trained on the mechanic's own photos. Re-train whenever you have collected more labelled photos.
- Score thresholds in `scoring.js` and quality thresholds in `quality.js` are starting points. Tune them on
  real photos from the workshop.
