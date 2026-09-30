"""
Reusable inference module for ChineseFoodNet (208 classes).

Loads a fine-tuned timm model and exposes:
    - read_class_names(csv_path)
    - load_model(model_name, num_classes, ckpt_path, device)
    - predict_topk(model, image, class_names, top_k, img_size, device)

Designed to be imported by the FastAPI server (server.py) but can also be used
in scripts or notebooks directly.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Defaults (must match the model that was fine-tuned)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_NAME = "convnextv2_base.fcmae_ft_in22k_in1k"
DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_CLASSES = 208
DEFAULT_TOP_K = 5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    """One predicted class with its probability."""
    class_id: int
    label_zh: str
    label_en: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "label_zh": self.label_zh,
            "label_en": self.label_en,
            "confidence": round(self.confidence, 6),
        }


# --------------------------------------------------------------------------- #
# Class names
# --------------------------------------------------------------------------- #
def read_class_names(csv_path: str | Path,
                     num_classes: int = DEFAULT_NUM_CLASSES
                     ) -> list[tuple[str, str]]:
    """Return [(zh_name, en_name), ...] sorted by class id (0..N-1).

    csv format: ``id,zh_name,en_name`` (BOM is tolerated)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return [(f"class_{i:03d}", "") for i in range(num_classes)]

    items: list[tuple[int, str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                idx = int(row[0])
            except (TypeError, ValueError):
                continue
            zh = row[1].strip() if len(row) > 1 else str(idx)
            en = row[2].strip() if len(row) > 2 else ""
            items.append((idx, zh, en))

    if not items:
        return [(f"class_{i:03d}", "") for i in range(num_classes)]

    items.sort(key=lambda x: x[0])
    out = [(zh, en) for _, zh, en in items]

    # Pad / truncate to num_classes for safety
    if len(out) < num_classes:
        out.extend([("", "")] * (num_classes - len(out)))
    elif len(out) > num_classes:
        out = out[:num_classes]
    return out


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    suffix = ckpt_path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file
        return load_safetensors_file(str(ckpt_path), device=str(device))
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            nested = state.get(key)
            if isinstance(nested, dict):
                return nested
        return state
    raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")


def _normalize_keys(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        out[nk] = v
    return out


def load_model(model_name: str = DEFAULT_MODEL_NAME,
               num_classes: int = DEFAULT_NUM_CLASSES,
               ckpt_path: str | Path | None = None,
               device: torch.device | str | None = None) -> tuple[torch.nn.Module, dict]:
    """Build a timm model and load fine-tuned weights from a local file.

    Returns (model, info_dict).  info_dict contains keys:
        loaded, total, skipped_shape, skipped_unknown, missing, head_warning
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    info = {
        "model_name": model_name,
        "num_classes": num_classes,
        "device": str(device),
        "loaded": 0,
        "total": len(model.state_dict()),
        "skipped_shape": 0,
        "skipped_unknown": 0,
        "missing": 0,
        "head_warning": "",
        "checkpoint": str(ckpt_path) if ckpt_path else "",
    }

    if ckpt_path is None or not Path(ckpt_path).exists():
        info["head_warning"] = f"checkpoint not found: {ckpt_path} (running with RANDOM weights)"
        model.to(device).eval()
        return model, info

    raw_state = _load_state_dict(Path(ckpt_path), device)
    state = _normalize_keys(raw_state)
    model_state = model.state_dict()

    matched = {}
    skipped_shape = []
    skipped_unknown = 0
    for k, v in state.items():
        if k not in model_state:
            skipped_unknown += 1
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        matched[k] = v

    missing, _ = model.load_state_dict(matched, strict=False)

    head_warning = ""
    for k, sh_ckpt, sh_model in skipped_shape:
        if "head" in k.lower():
            head_warning = (
                f"head shape mismatch: ckpt={sh_ckpt} vs model={sh_model} "
                f"(this is probably not a fine-tuned 208-class checkpoint)"
            )

    info.update({
        "loaded": len(matched),
        "skipped_shape": len(skipped_shape),
        "skipped_unknown": skipped_unknown,
        "missing": len(missing),
        "head_warning": head_warning,
    })

    model.to(device).eval()
    return model, info


# --------------------------------------------------------------------------- #
# Preprocessing & inference
# --------------------------------------------------------------------------- #
def build_preprocess(img_size: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def predict_topk(model: torch.nn.Module,
                 image: Image.Image,
                 class_names: list[tuple[str, str]],
                 top_k: int = DEFAULT_TOP_K,
                 img_size: int = DEFAULT_IMG_SIZE,
                 device: torch.device | str | None = None,
                 preprocess: transforms.Compose | None = None,
                 ) -> list[Prediction]:
    """Run a single image through the model and return top-K predictions."""
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    if preprocess is None:
        preprocess = build_preprocess(img_size)

    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top_k = min(top_k, probs.numel())
    values, indices = probs.topk(top_k)

    results: list[Prediction] = []
    for val, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
        zh, en = class_names[idx] if idx < len(class_names) else ("", "")
        results.append(Prediction(class_id=int(idx),
                                  label_zh=zh,
                                  label_en=en,
                                  confidence=float(val)))
    return results
