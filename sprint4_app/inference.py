from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "baseline_mobilenetv2.pth"
CLASS_NAMES = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"])

_MODEL: torch.nn.Module | None = None


preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def build_model() -> torch.nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
    model.eval()
    return model


def get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
        _MODEL = build_model()
    return _MODEL


def predict_image(image_bytes: bytes) -> dict:
    image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(get_model()(tensor), dim=1)[0]
        top_probs, top_idxs = torch.topk(probabilities, 5)

    top_predictions = [
        {
            "class_name": CLASS_NAMES[idx],
            "confidence": round(float(prob), 6),
        }
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist())
    ]

    return {
        "predicted_class": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["confidence"],
        "low_confidence": top_predictions[0]["confidence"] < 0.5,
        "top_predictions": top_predictions,
    }
