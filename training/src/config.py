"""Shared constants for training, export and the web app."""

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ARCH = "efficientnet_b0"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class order is the contract between training and the web app's scoring.js.
TASKS = {
    "condition": ["good", "defective", "not_tyre"],
    "tread": ["good", "moderate", "worn"],
}
