# TyreCheck: on-device tyre health from a photo

TyreCheck is a web app for mechanics and drivers. Take a photo of a tyre and it gives you a health report:

- **Condition:** good or defective (cracks, cuts, bulges, punctures)
- **Tread wear:** good, moderate or worn
- a **0–100 health score** with a green / amber / red status
- **heatmaps** showing where on the tyre the model looked
- plain-language **recommendations** on what to do next

It is **edge AI**: the neural networks run inside your browser (ONNX Runtime Web + WebAssembly).
Photos never leave the device, it works offline after the first visit, and it costs nothing to host.
There is no server, no account and no build step.

```
 phone/PC browser ─────────────────────────────────────────────────────────────┐
 │ photo → quality check (blur, light) → centre crop 224² → condition model ─┐  │
 │                                                       → tread model ─────┤  │
 │                     health score + status + advice + CAM heatmap ◄────────┘  │
 └──────────────────────────────────────────────────────────────────────────────┘
 training (free Colab GPU): public datasets → EfficientNet-B0 fine-tune → ONNX
```

---

## Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Set up on another laptop](#set-up-on-another-laptop)
4. [How to use the app](#how-to-use-the-app)
5. [Reading the report](#reading-the-report)
6. [Technical details](#technical-details)
7. [Project layout](#project-layout)
8. [Model performance](#model-performance)
9. [Retraining the models](#retraining-the-models)
10. [Deploying to GitHub Pages](#deploying-to-github-pages)
11. [Tests](#tests)
12. [Troubleshooting](#troubleshooting)
13. [Limitations](#limitations)

---

## Features

| Feature | What it does |
|---|---|
| **Take photo / upload / drag & drop** | Use the phone camera directly (rear camera opens by default), pick an existing image, or drag one onto the page on a PC. |
| **Condition check** | Classifies the tyre as *good* or *defective* (cracks, cuts, bulges, exposed cords). |
| **Tread-wear check** | Classifies tread as *good*, *moderate* or *worn*. Marked **Beta** in the app because its accuracy is lower (see [Model performance](#model-performance)). |
| **Not-a-tyre rejection** | If the photo probably isn't a tyre, the app asks you to retake it instead of guessing. |
| **Photo quality check** | Measures blur and brightness before analysis and warns if the photo is blurry, too dark or over-exposed. |
| **Health score (0–100)** | Combines both models into one number with a status: **Good to go** (≥ 70), **Monitor / inspect soon** (45–69), **Repair or replace** (< 45). |
| **Confidence bars** | Shows how confident each model is. Low confidence (< 60%) is flagged and you're asked to retake from another angle. |
| **Damage map / Wear map** | Class-activation heatmaps over the photo showing which area drove the decision. |
| **Recommendations** | Practical next steps, e.g. measure with a depth gauge, check alignment, replace immediately if there's a bulge. |
| **Print / save PDF** | Prints a clean report (buttons and history are hidden in print). Use "Save as PDF" in the print dialog to keep a copy. |
| **Recent checks** | The last 30 checks (thumbnail, score, status, date) are kept in the browser on that device only. A **Clear** button wipes them. |
| **Works offline** | After the first visit, the app, models and runtime are cached by a service worker. |
| **Installable (PWA)** | "Add to home screen" on a phone or "Install app" in Chrome/Edge on a PC. It then opens like a normal app. |
| **Sample photos** | `samples/` has 10 freely licensed tyre photos (new, worn, bald, cracked, punctured, cut) to try the app with. |
| **Private by design** | No uploads, no analytics, no backend. Everything runs on your device. |

---

## Requirements

### To run the app (most people only need this)

| Need | Details |
|---|---|
| **Git** | To clone the repo. [git-scm.com](https://git-scm.com/downloads). Or download the ZIP from GitHub instead. |
| **Python 3.8+** | Only used as a tiny local web server (`python -m http.server`). Any other static file server also works. |
| **A modern browser** | Chrome, Edge, Firefox or Safari (recent versions). WebAssembly must be enabled (it is by default). |
| **Internet on first load** | The ONNX runtime is loaded from the jsDelivr CDN the first time. After that it's cached and works offline. |
| **Disk space** | About 40 MB (the two models are ~16 MB each). |

No GPU, no `pip install`, no Node.js and no build step are needed just to use the app.

### To retrain the models (optional)

- A free **Google Colab** account (recommended, gives you a free GPU), **or** Python 3.10–3.12 locally
- A free **Kaggle** account and API token (to download the datasets)
- Python packages from `training/requirements.txt` (PyTorch, timm, albumentations, ONNX, onnxruntime, …)

### To run the tests (optional)

- **Node.js 18+** (CI uses 22) for the JavaScript tests
- The training Python packages + `pytest` for the Python tests

---

## Set up on another laptop

### 1. Clone the repository

```bash
git clone https://github.com/abhinav29890/TyreHealth.git
cd TyreHealth
```

No Git? On the GitHub page click **Code → Download ZIP**, unzip it and open a terminal in that folder.

The trained models (`web/models/*.onnx`) are committed to the repo, so there is nothing extra to download.

### 2. Start a local web server

**Windows (PowerShell or Command Prompt):**
```powershell
python -m http.server 8000 -d web
```
If `python` isn't found, try `py -m http.server 8000 -d web`.

**macOS / Linux:**
```bash
python3 -m http.server 8000 -d web
```

> Opening `web/index.html` by double-clicking it will **not** work. Browsers block loading
> the models and modules from `file://` URLs. Always go through a web server.

### 3. Open the app

Go to **http://localhost:8000** in your browser. The pill in the top-right corner shows
*Loading models…* and then **Ready · on-device** when it's ready to use.

### 4. (Optional) Use it from a phone on the same Wi-Fi

1. Find the laptop's local IP address:
   - Windows: `ipconfig` → *IPv4 Address* (e.g. `192.168.1.23`)
   - macOS: `ipconfig getifaddr en0` / Linux: `hostname -I`
2. On the phone, open `http://<that-ip>:8000`.
3. If it doesn't load, allow Python through the laptop's firewall for private networks.

Analysis works over plain HTTP, but **offline mode and "install to home screen" need HTTPS**
(or `localhost`). The easiest way to get HTTPS for free is the
[GitHub Pages deployment](#deploying-to-github-pages).

### 5. (Optional) Set up the training environment

Only needed if you want to retrain models or run the Python tests.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r training/requirements.txt
```

On a machine without an NVIDIA GPU, install the smaller CPU build of PyTorch first:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

---

## How to use the app

1. **Open the app** and wait for the status pill to show **Ready · on-device**.
2. **Get a good photo:**
   - Fill the frame with the tread or sidewall. **Only the centre square is analysed**, and the
     preview shows exactly what the model saw.
   - Use good, even light. Avoid glare and direct sunlight.
   - Hold the camera steady. Wipe off heavy mud first if you can.
3. **Load it:** tap **Take photo** (phone camera), **Upload image**, or drag & drop an image onto the box.
4. **Wait a second or two** while it shows *Analysing…*.
5. **Read the report:** score, status, findings with confidence, and recommendations
   (see [Reading the report](#reading-the-report)).
6. **Check the heatmaps:** switch between **Photo**, **Damage map** and **Wear map** to see which
   part of the tyre the model focused on.
7. **Save it (optional):** **Print / save PDF** → choose "Save as PDF" as the printer.
8. **Next tyre:** tap **New check**. Past results appear under **Recent checks**.

Don't have a tyre handy? Open **Sample photos to test with** at the bottom of the page
(`/samples/`), save an image, and upload it.

---

## Reading the report

| Score | Status | Meaning |
|---|---|---|
| 70–100 | 🟢 **Good to go** | No defect found and tread looks fine. Routine checks only. |
| 45–69 | 🟠 **Monitor / inspect soon** | Partially worn tread or a borderline result. Plan a closer look. |
| 0–44 | 🔴 **Repair or replace** | A likely defect or worn tread. Inspect it now. |

- **Possible defect** caps the score at 40, **worn tread** caps it at 35, and **moderate wear** caps it at 65.
  So one serious problem can't be hidden by a good result in the other check.
- **"% confident"** is the model's probability for its answer. Under 60% is marked uncertain.
- A **Beta** badge on tread wear means that model is less reliable. Always confirm tread depth
  with a gauge.
- **Quality warnings** (blur, dark, over-exposed) mean the result may be unreliable. Retake the photo.
- **"Not a tyre?"** means the photo was rejected. Retake it with the tyre filling most of the frame.

---

## Technical details

### Tech stack

| Layer | Technology |
|---|---|
| Frontend | Plain HTML, CSS and JavaScript ES modules. No framework and no build step. |
| Inference | [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.30 (WebAssembly backend, multi-threaded when cross-origin isolated) |
| Offline / install | Service worker + Web App Manifest (PWA) |
| Storage | `localStorage` for history (max 30 entries, this device only) |
| Models | EfficientNet-B0 (via `timm`), fine-tuned in PyTorch, exported to ONNX |
| Training | PyTorch, timm, albumentations, scikit-learn, on Google Colab (free T4 GPU) |
| Hosting | GitHub Pages via GitHub Actions |
| CI | GitHub Actions: Node test runner + pytest |

### Processing pipeline (in the browser)

1. **Decode** the image with `createImageBitmap`.
2. **Quality gate** (`web/js/quality.js`): downscale to 256×256, then compute
   - sharpness as the variance of the Laplacian (very blurry < 25, slightly blurry < 60)
   - mean brightness (too dark < 40, dark < 70, over-exposed > 220)
3. **Preprocess** (`web/js/preprocess.js`): centre square crop → 224×224 → ImageNet mean/std
   normalisation → NCHW float32 tensor. This is the same transform used in training.
4. **Inference** (`web/js/inference.js`): run both ONNX models. Each returns
   - `logits` → softmax → class probabilities
   - `cam` → a 7×7 class activation map per class
5. **Scoring** (`web/js/scoring.js`):
   - reject if P(not_tyre) ≥ 0.5
   - condition probabilities are renormalised without the not-tyre class
   - expected tread level = 1·P(good) + 0.5·P(moderate) + 0·P(worn)
   - score = 100 × (0.6 · P(good condition) + 0.4 · tread level). When the tread model is
     Beta, the weights are 0.85 / 0.15.
   - caps: defect → 40, worn → 35, moderate → 65. Status thresholds: 70 / 45.
6. **Heatmap** (`web/js/cam.js`): the CAM for the predicted class is upscaled and drawn as a
   translucent overlay.
7. **History** (`web/js/history.js`): thumbnail + score saved to `localStorage`.

### Models

| | Condition | Tread |
|---|---|---|
| Classes | `good`, `defective`, `not_tyre` | `good`, `moderate`, `worn` |
| Architecture | EfficientNet-B0, ImageNet-pretrained | EfficientNet-B0, ImageNet-pretrained |
| Input | 224×224 RGB | 224×224 RGB |
| File | `web/models/condition.onnx` (~16 MB, fp32) | `web/models/tread.onnx` (~16 MB, fp32) |
| Status | Production | Beta |

Each model has a metadata file (`web/models/<task>.json`) with its labels, input size,
normalisation, a hash (used to cache-bust the `.onnx` file), test metrics, and the
`beta` / `placeholder` flags that the app reads.

**Training recipe:** class-weighted cross-entropy with label smoothing, AdamW, cosine learning-rate
schedule, a 2-epoch head-only warm-up, augmentation with albumentations, and early stopping on
validation macro-F1. Near-duplicate images across datasets are removed with a perceptual hash so
they can't leak between the train and test splits.

**Export:** ONNX with two outputs, `logits` and `cam` (the classifier applied at every 7×7 position,
which gives class activation maps for free). The exporter also tries static INT8 quantisation
(QDQ, per-channel, calibrated on validation images) and keeps it **only if** INT8 agrees with fp32
on ≥ 98% of the test set. The current models didn't pass that check, so they ship as fp32.
A model is automatically marked **beta** when its test macro-F1 is below 0.85.

### Offline caching (`web/sw.js`)

| Resource | Strategy |
|---|---|
| App shell (HTML, CSS, JS, icons) | stale-while-revalidate |
| `models/<task>.json` | network-first, so retrained models are picked up |
| `models/*.onnx?v=<hash>` | cache-first; older versions are evicted |
| ONNX Runtime from jsDelivr | cache-first (version-pinned URL) |

### Data sources

| Model | Source | Notes |
|---|---|---|
| condition | [Kaggle: Tyre Quality Classification](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification) | ~1.8k images, good / defective |
| condition | [Kaggle: Tire Texture Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition) | ~1k images, cracked / normal |
| condition | [Imagenette](https://github.com/fastai/imagenette) | ~1k random non-tyre photos for the `not_tyre` class |
| tread | [Kaggle: Tyre Condition Classification Dataset](https://www.kaggle.com/datasets/sameersambhare1/tyre-condition-classification-dataset) | ~550 workshop photos: NEW→good, SERVICEABLE→moderate, UNUSABLE→worn |
| tread | [Kaggle: Tire Tread Photos](https://www.kaggle.com/datasets/numberfive/tire-tread-photos) | ~370 images: GoodTireTread→good, BadTireTread→worn |
| both | your own photos | `training/data/own/<task>/<class>/*.jpg`, picked up automatically |

Add more sources (Kaggle, Roboflow, Hugging Face, local folders) by editing `training/sources.json`.

---

## Project layout

```
TyreHealth/
├── web/                         The app (served as-is, no build step)
│   ├── index.html               Page layout
│   ├── style.css                Styles (including print styles)
│   ├── manifest.json            PWA manifest
│   ├── sw.js                    Service worker (offline caching)
│   ├── icons/                   App icons
│   ├── js/
│   │   ├── app.js               UI wiring: capture, analyse, render report, history
│   │   ├── inference.js         Loads ONNX models and runs them
│   │   ├── preprocess.js        Crop / resize / normalise into a tensor
│   │   ├── quality.js           Blur and exposure checks
│   │   ├── scoring.js           Health score, status and recommendations
│   │   ├── cam.js               Heatmap overlay
│   │   └── history.js           Recent checks in localStorage
│   ├── models/                  condition.onnx/.json, tread.onnx/.json
│   └── samples/                 10 test photos + credits page
├── training/
│   ├── src/
│   │   ├── config.py            Shared constants (image size, class order, normalisation)
│   │   ├── prepare_data.py      Download, map, de-duplicate and split datasets
│   │   ├── dataset.py           PyTorch dataset + augmentations
│   │   ├── model.py             EfficientNet-B0 with a CAM-friendly head
│   │   ├── train.py             Training loop
│   │   ├── evaluate.py          Test-set metrics + confusion matrix
│   │   └── export_onnx.py       ONNX export, optional INT8, metadata JSON
│   ├── notebooks/               Colab notebooks (01 explore, 02 condition, 03 tread)
│   ├── reports/                 Test metrics of the shipped models
│   ├── sources.json             Dataset list and folder→class mapping
│   └── requirements.txt         Python dependencies for training
├── tests/                       JS (scoring/preprocessing) and Python (export parity, pipeline) tests
└── .github/workflows/           pages.yml (deploy), tests.yml (CI)
```

---

## Model performance

Measured on held-out test sets (`training/reports/*.json`):

| Model | Test images | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| Condition | 445 | **97.3%** | 0.97 | good 98% recall, defective 96%, not-tyre 98% |
| Tread (Beta) | 134 | 76.1% | 0.71 | good 80% recall, moderate 93%, **worn only 41%** |

The tread model misses many worn tyres because the public data has only ~200 "worn" images.
This is why it's marked Beta, carries less weight in the score, and the report always asks
for a gauge check.

---

## Retraining the models

### On Google Colab (recommended, free GPU, ~15–25 min per model)

1. Open `training/notebooks/02_train_condition.ipynb` in [Google Colab](https://colab.research.google.com)
   (**File → Open notebook → GitHub**, paste the repo URL).
2. **Runtime → Change runtime type → T4 GPU**.
3. Get a Kaggle API token (kaggle.com → Settings → *Create New Token*) and add it as Colab
   secrets (🔑 icon on the left): `KAGGLE_USERNAME` and `KAGGLE_KEY`.
4. **Runtime → Run all**. At the end it downloads a zip.
5. Unzip it at the project root. This replaces `web/models/condition.*` and adds
   `training/reports/condition_*`.
6. Repeat with `03_train_tread.ipynb`. **Strongly recommended:** add 100–200 of your own tread
   photos per class (set `UPLOAD_OWN_PHOTOS = True` in the notebook).
7. Commit and push. The app picks up the new models automatically (cache-busted by hash).

### Locally (CPU works, but slowly)

```bash
cd training
python -m src.prepare_data --task condition --clean
python -m src.train       --task condition --data data/condition --epochs 15 --workers 0
python -m src.evaluate    --ckpt checkpoints/condition/best.pt --data data/condition
python -m src.export_onnx --ckpt checkpoints/condition/best.pt --data data/condition
```

Replace `condition` with `tread` for the tread model. For Kaggle downloads, put `kaggle.json` in
`~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`) or set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.

Useful options: `train --epochs --batch --lr --patience --freeze-epochs`,
`export_onnx --min-agree 0.98 --beta-below 0.85`.

---

## Deploying to GitHub Pages

1. Push the repo to GitHub.
2. Go to **Settings → Pages → Source: GitHub Actions**.
3. `.github/workflows/pages.yml` publishes the `web/` folder on every push to `main` that touches `web/`.
4. The app is then available at `https://abhinav29890.github.io/TyreHealth/` over HTTPS, so offline
   mode and install-to-home-screen work on phones.

---

## Tests

```bash
node --test tests/scoring.test.mjs   # scoring, preprocessing and quality checks
pytest tests                         # ONNX export parity + train→evaluate→export smoke test
```

Both suites run on every push and pull request via `.github/workflows/tests.yml`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Blank page or "Model error" when opening `index.html` directly | Run it through a web server (`python -m http.server 8000 -d web`), not `file://`. |
| "Could not load the AI models" | The first load needs internet for the ONNX runtime CDN. Check your connection and reload. |
| `python` not recognised (Windows) | Use `py` instead, or reinstall Python with "Add to PATH" ticked. |
| Phone can't reach `http://<ip>:8000` | Same Wi-Fi? Allow Python through the firewall for private networks. |
| No "Install app" option or no offline mode | These need HTTPS or `localhost`. Use the GitHub Pages URL. |
| Old version still showing after an update | Reload twice (the service worker updates in the background), or hard-refresh with Ctrl+Shift+R. |
| "Demo mode" banner | The models in `web/models` are untrained placeholders. Retrain or restore them from git. |
| History is empty | History is per browser and per device, and is not kept in private/incognito windows. |

---

## Limitations

- This is a **visual screening aid**, not a replacement for a physical inspection and a tread-depth gauge.
- The tread model is **Beta** (macro-F1 0.71, finds only 41% of worn tyres). Retrain it with the
  workshop's own photos to improve it.
- Only the **centre square** of the photo is analysed.
- Score thresholds in `web/js/scoring.js` and quality thresholds in `web/js/quality.js` are
  starting points. Tune them on real photos from the workshop.
- History is stored only in that browser on that device and is not synced or backed up.
