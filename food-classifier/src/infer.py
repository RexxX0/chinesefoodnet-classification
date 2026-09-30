"""Inference helpers and CLI for ChineseFoodNet classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import timm
import torch
from config import DEFAULT_CONFIG, load_config
from PIL import Image
from torchvision import transforms


def build_model(model_name: str, num_classes: int, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def read_class_names(path: str) -> list[str]:
    class_file = Path(path)
    if not class_file.exists():
        return [str(i) for i in range(208)]
    if class_file.suffix.lower() == ".csv":
        names: list[tuple[int, str]] = []
        with class_file.open("r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    idx = int(row[0])
                except (TypeError, ValueError):
                    continue
                zh_name = row[1].strip() if len(row) > 1 else str(idx)
                en_name = row[2].strip() if len(row) > 2 else ""
                label = f"{idx:03d} {zh_name}"
                if en_name:
                    label = f"{label} | {en_name}"
                names.append((idx, label))
        if names:
            names.sort(key=lambda item: item[0])
            return [label for _, label in names]
    return [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_image(image: Image.Image, img_size: int) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return tfm(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict_topk(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_names: list[str],
    top_k: int,
    device: torch.device,
) -> list[tuple[str, float]]:
    logits = model(input_tensor.to(device))
    probs = torch.softmax(logits, dim=1)[0]
    values, indices = probs.topk(top_k)
    return [(class_names[idx.item()], float(val.item())) for val, idx in zip(values, indices)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image inference for ChineseFoodNet classifier.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    num_classes = int(cfg["project"]["num_classes"])
    top_k = int(args.top_k if args.top_k is not None else cfg["inference"]["top_k"])

    model = build_model(model_cfg["name"], num_classes, paths_cfg["checkpoint_path"], device)
    class_names = read_class_names(paths_cfg["class_names_file"])
    image = Image.open(args.image)
    input_tensor = preprocess_image(image, img_size=int(model_cfg["img_size"]))
    predictions = predict_topk(model, input_tensor, class_names, top_k=top_k, device=device)

    for label, score in predictions:
        print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
