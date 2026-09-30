"""Standalone evaluation script."""

from __future__ import annotations

import argparse

import torch
from config import DEFAULT_CONFIG, load_config
from dataset import build_imagefolder_loader
from infer import build_model
from train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on validation data.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    project_cfg = cfg["project"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    paths_cfg = cfg["paths"]
    data_cfg = cfg.get("data", {})
    class_limit = data_cfg.get("class_limit")
    max_samples_per_class = data_cfg.get("max_samples_per_class")
    if class_limit is not None:
        class_limit = int(class_limit)
    if max_samples_per_class is not None:
        max_samples_per_class = int(max_samples_per_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg["name"], int(project_cfg["num_classes"]), paths_cfg["checkpoint_path"], device)

    val_loader = build_imagefolder_loader(
        root_dir=paths_cfg["val_dir"],
        img_size=int(model_cfg["img_size"]),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        train=False,
        class_limit=class_limit,
        max_samples_per_class=max_samples_per_class,
    )
    val_loss, val_acc = evaluate(model, val_loader, device, use_amp=device.type == "cuda")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
