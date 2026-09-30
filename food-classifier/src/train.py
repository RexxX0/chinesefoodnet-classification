"""Training entrypoint for ChineseFoodNet classifier."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import timm
import torch
from config import DEFAULT_CONFIG, load_config
from dataset import (
    build_imagefolder_loader,
    build_loader_from_samples,
    build_stratified_folds,
    collect_imagefolder_samples,
)
from safetensors.torch import load_file as load_safetensors_file
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ChineseFoodNet classifier.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", type=str, default=None, help="Path to a training checkpoint to resume from.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def _extract_state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("state_dict", "model_state_dict", "model", "module"):
        nested = state.get(key)
        if isinstance(nested, dict):
            return nested
    return state


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        normalized[new_key] = value
    return normalized


def load_local_pretrained_weights(model: nn.Module, pretrained_checkpoint: str, device: torch.device) -> None:
    checkpoint_path = Path(pretrained_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Local pretrained checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix.lower() == ".safetensors":
        state_dict = load_safetensors_file(str(checkpoint_path), device=str(device))
    else:
        state = torch.load(checkpoint_path, map_location=device)
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
        state_dict = _extract_state_dict(state)

    state_dict = _normalize_state_dict_keys(state_dict)
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)

    print(f"Loaded local pretrained weights from: {checkpoint_path}")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} mismatched or unknown pretrained keys.")
    if missing_keys:
        print(f"Missing model keys after pretrained load: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected pretrained keys after load: {len(unexpected_keys)}")


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_acc: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)


def resume_training_state(
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(resume_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {resume_path} is not a resumable training checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_acc = float(checkpoint.get("best_acc", -1.0))
    return start_epoch, best_acc


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    steps = 0
    correct = 0
    total = 0
    amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    progress = tqdm(loader, desc="train", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(images)
            loss = criterion(logits, labels)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        steps += 1
        running_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
        avg_loss = running_loss / max(steps, 1)
        avg_acc = correct / max(total, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
    return running_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss()
    amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    progress = tqdm(loader, desc="val", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        with amp_ctx:
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
        steps += 1
        running_loss += float(loss.item())
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
        avg_loss = running_loss / max(steps, 1)
        avg_acc = correct / max(total, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
    return running_loss / max(len(loader), 1), correct / max(total, 1)


def plot_history(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_acc"]) for row in history]
    val_acc = [float(row["val_acc"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc")
    axes[1].plot(epochs, val_acc, label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", dpi=150)
    plt.close(fig)


def load_existing_history(log_csv: Path) -> list[dict[str, float]]:
    if not log_csv.exists():
        return []
    with log_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        history: list[dict[str, float]] = []
        for row in reader:
            history.append(
                {
                    "epoch": float(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "val_acc": float(row["val_acc"]),
                    "epoch_seconds": float(row["epoch_seconds"]),
                }
            )
        return history


def save_history(log_csv: Path, history: list[dict[str, float]]) -> None:
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc",
                "epoch_seconds",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(history)


def build_rotating_cv_loaders(
    merged_samples: list[tuple[str, int]],
    classes: list[str],
    class_to_idx: dict[str, int],
    folds: list[list[int]],
    active_fold: int,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    val_indices = set(folds[active_fold])
    train_samples = [sample for index, sample in enumerate(merged_samples) if index not in val_indices]
    val_samples = [sample for index, sample in enumerate(merged_samples) if index in val_indices]
    if not train_samples or not val_samples:
        raise ValueError(
            f"Cross-validation split for fold {active_fold} is empty. "
            "Try reducing fold_count or increasing the available data."
        )
    train_loader = build_loader_from_samples(
        samples=train_samples,
        classes=classes,
        class_to_idx=class_to_idx,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
    )
    val_loader = build_loader_from_samples(
        samples=val_samples,
        classes=classes,
        class_to_idx=class_to_idx,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
    )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    from torch.utils.tensorboard import SummaryWriter

    cfg = load_config(args.config)
    project_cfg = cfg["project"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    paths_cfg = cfg["paths"]
    data_cfg = cfg.get("data", {})
    cv_cfg = cfg.get("cross_validation", {})
    class_limit = data_cfg.get("class_limit")
    max_samples_per_class = data_cfg.get("max_samples_per_class")
    if class_limit is not None:
        class_limit = int(class_limit)
    if max_samples_per_class is not None:
        max_samples_per_class = int(max_samples_per_class)

    set_seed(int(project_cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    cv_enabled = bool(cv_cfg.get("enabled", False))
    cv_fold_count = int(cv_cfg.get("fold_count", 5))
    cv_fold_epochs = int(cv_cfg.get("fold_epochs", 10))

    merged_samples: list[tuple[str, int]] | None = None
    merged_classes: list[str] | None = None
    merged_class_to_idx: dict[str, int] | None = None
    cv_folds: list[list[int]] | None = None
    active_fold = -1

    if cv_enabled:
        merged_samples, merged_classes, merged_class_to_idx = collect_imagefolder_samples(
            root_dirs=[paths_cfg["train_dir"], paths_cfg["val_dir"]],
            class_limit=class_limit,
            max_samples_per_class=max_samples_per_class,
        )
        cv_folds = build_stratified_folds(merged_samples, cv_fold_count)
        train_loader, val_loader = build_rotating_cv_loaders(
            merged_samples=merged_samples,
            classes=merged_classes,
            class_to_idx=merged_class_to_idx,
            folds=cv_folds,
            active_fold=0,
            img_size=int(model_cfg["img_size"]),
            batch_size=int(train_cfg["batch_size"]),
            num_workers=int(train_cfg["num_workers"]),
        )
        print(
            f"Cross-validation enabled: merged {len(merged_samples)} samples from train/val, "
            f"{cv_fold_count} folds, rotating every {cv_fold_epochs} epochs."
        )
    else:
        train_loader = build_imagefolder_loader(
            root_dir=paths_cfg["train_dir"],
            img_size=int(model_cfg["img_size"]),
            batch_size=int(train_cfg["batch_size"]),
            num_workers=int(train_cfg["num_workers"]),
            train=True,
            class_limit=class_limit,
            max_samples_per_class=max_samples_per_class,
        )
        val_loader = build_imagefolder_loader(
            root_dir=paths_cfg["val_dir"],
            img_size=int(model_cfg["img_size"]),
            batch_size=int(train_cfg["batch_size"]),
            num_workers=int(train_cfg["num_workers"]),
            train=False,
            class_limit=class_limit,
            max_samples_per_class=max_samples_per_class,
        )

    model = build_model(
        model_name=model_cfg["name"],
        num_classes=int(project_cfg["num_classes"]),
        pretrained=bool(model_cfg["pretrained"]),
    ).to(device)
    pretrained_checkpoint = model_cfg.get("pretrained_checkpoint")
    if pretrained_checkpoint:
        load_local_pretrained_weights(model, str(pretrained_checkpoint), device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))

    output_dir = Path(paths_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_csv = logs_dir / "history.csv"
    writer = SummaryWriter(log_dir=str(logs_dir / "tensorboard"))
    best_acc = -1.0
    best_ckpt = Path(paths_cfg["checkpoint_path"])
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)
    latest_ckpt = Path(paths_cfg.get("latest_checkpoint_path", output_dir / "latest_checkpoint.pth"))
    latest_ckpt.parent.mkdir(parents=True, exist_ok=True)
    history = load_existing_history(log_csv)
    start_epoch = 0

    resume_cfg = cfg.get("resume", {})
    auto_resume = bool(resume_cfg.get("auto_resume", True))
    resume_path = args.resume or paths_cfg.get("resume_checkpoint_path")
    if resume_path is None and auto_resume and latest_ckpt.exists():
        resume_path = str(latest_ckpt)
    if resume_path:
        resume_file = Path(resume_path)
        if not resume_file.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_file}")
        start_epoch, best_acc = resume_training_state(resume_file, model, optimizer, scaler, device)
        print(f"Resumed training from epoch {start_epoch} using: {resume_file}")
        if start_epoch >= int(train_cfg["epochs"]):
            print(
                f"Resume checkpoint is already at epoch {start_epoch}, "
                f"which reaches or exceeds configured epochs={int(train_cfg['epochs'])}."
            )
            writer.close()
            return

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        if cv_enabled:
            next_active_fold = (epoch // cv_fold_epochs) % cv_fold_count
            if next_active_fold != active_fold:
                active_fold = next_active_fold
                assert merged_samples is not None
                assert merged_classes is not None
                assert merged_class_to_idx is not None
                assert cv_folds is not None
                train_loader, val_loader = build_rotating_cv_loaders(
                    merged_samples=merged_samples,
                    classes=merged_classes,
                    class_to_idx=merged_class_to_idx,
                    folds=cv_folds,
                    active_fold=active_fold,
                    img_size=int(model_cfg["img_size"]),
                    batch_size=int(train_cfg["batch_size"]),
                    num_workers=int(train_cfg["num_workers"]),
                )
                print(
                    f"Switched cross-validation fold to {active_fold + 1}/{cv_fold_count} "
                    f"for epochs {epoch + 1}-{min(epoch + cv_fold_epochs, int(train_cfg['epochs']))}."
                )
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=use_amp, scaler=scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
        epoch_time = time.time() - epoch_start

        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("loss/val", val_loss, epoch + 1)
        writer.add_scalar("acc/train", train_acc, epoch + 1)
        writer.add_scalar("acc/val", val_acc, epoch + 1)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch + 1)
        if cv_enabled:
            writer.add_scalar("cv/active_fold", active_fold + 1, epoch + 1)

        row = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "epoch_seconds": epoch_time,
        }
        history = [item for item in history if int(item["epoch"]) != epoch + 1]
        history.append(row)
        history.sort(key=lambda item: item["epoch"])
        save_history(log_csv, history)
        writer.flush()

        fold_prefix = f"fold={active_fold + 1}/{cv_fold_count} " if cv_enabled else ""
        print(
            f"Epoch {epoch + 1}: {fold_prefix}train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} epoch_s={epoch_time:.1f}"
        )
        save_training_checkpoint(latest_ckpt, model, optimizer, scaler, epoch, best_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            save_training_checkpoint(latest_ckpt, model, optimizer, scaler, epoch, best_acc)

    save_history(log_csv, history)

    plot_history(history, output_dir=output_dir)
    writer.close()

    print(f"Training finished. Best val_acc={best_acc:.4f}")
    print(f"Saved checkpoint to: {best_ckpt}")
    print(f"Saved latest training checkpoint to: {latest_ckpt}")
    print(f"Saved CSV log to: {log_csv}")
    print(f"Saved figure to: {output_dir / 'metrics.png'}")
    print(f"TensorBoard logdir: {logs_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
