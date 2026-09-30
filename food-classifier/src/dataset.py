"""Dataset and transform helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_imagefolder_loader(
    root_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    class_limit: int | None = None,
    max_samples_per_class: int | None = None,
) -> DataLoader:
    dataset = datasets.ImageFolder(Path(root_dir), transform=build_transforms(img_size, train=train))
    if class_limit is not None:
        if class_limit <= 0:
            raise ValueError("class_limit must be a positive integer.")
        selected_old_ids = set(range(class_limit))
        filtered_samples = [(path, label) for path, label in dataset.samples if label in selected_old_ids]
        if not filtered_samples:
            raise ValueError(f"No samples found for class_limit={class_limit} under {root_dir}.")
        dataset.samples = filtered_samples
        dataset.imgs = filtered_samples
        dataset.targets = [label for _, label in filtered_samples]
        dataset.classes = dataset.classes[:class_limit]
        dataset.class_to_idx = {name: idx for idx, name in enumerate(dataset.classes)}
    if max_samples_per_class is not None:
        if max_samples_per_class <= 0:
            raise ValueError("max_samples_per_class must be a positive integer.")
        per_class_count: dict[int, int] = {}
        limited_samples: list[tuple[str, int]] = []
        for path, label in dataset.samples:
            current = per_class_count.get(label, 0)
            if current >= max_samples_per_class:
                continue
            limited_samples.append((path, label))
            per_class_count[label] = current + 1
        if not limited_samples:
            raise ValueError(
                f"No samples found for max_samples_per_class={max_samples_per_class} under {root_dir}."
            )
        dataset.samples = limited_samples
        dataset.imgs = limited_samples
        dataset.targets = [label for _, label in limited_samples]
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)


class ImagePathDataset(Dataset[tuple[object, int]]):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        classes: list[str],
        class_to_idx: dict[str, int],
        img_size: int,
        train: bool,
    ) -> None:
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(img_size, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def collect_imagefolder_samples(
    root_dirs: list[str],
    class_limit: int | None = None,
    max_samples_per_class: int | None = None,
) -> tuple[list[tuple[str, int]], list[str], dict[str, int]]:
    if not root_dirs:
        raise ValueError("root_dirs must not be empty.")

    root_paths = [Path(root_dir) for root_dir in root_dirs]
    for root_path in root_paths:
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root_path}")

    class_names = sorted(
        {
            class_dir.name
            for root_path in root_paths
            for class_dir in root_path.iterdir()
            if class_dir.is_dir()
        }
    )
    if not class_names:
        raise ValueError(f"No class directories found under: {root_dirs}")

    if class_limit is not None:
        if class_limit <= 0:
            raise ValueError("class_limit must be a positive integer.")
        class_names = class_names[:class_limit]

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    per_class_count = {idx: 0 for idx in range(len(class_names))}
    samples: list[tuple[str, int]] = []

    for class_name in class_names:
        label = class_to_idx[class_name]
        for root_path in root_paths:
            class_dir = root_path / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if max_samples_per_class is not None:
                    if max_samples_per_class <= 0:
                        raise ValueError("max_samples_per_class must be a positive integer.")
                    if per_class_count[label] >= max_samples_per_class:
                        continue
                samples.append((str(image_path), label))
                per_class_count[label] += 1

    if not samples:
        raise ValueError(f"No image samples found under: {root_dirs}")

    return samples, class_names, class_to_idx


def build_loader_from_samples(
    samples: list[tuple[str, int]],
    classes: list[str],
    class_to_idx: dict[str, int],
    img_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    dataset = ImagePathDataset(
        samples=samples,
        classes=classes,
        class_to_idx=class_to_idx,
        img_size=img_size,
        train=train,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)


def build_stratified_folds(samples: list[tuple[str, int]], fold_count: int) -> list[list[int]]:
    if fold_count <= 1:
        raise ValueError("fold_count must be greater than 1.")
    label_to_indices: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(samples):
        label_to_indices.setdefault(label, []).append(index)

    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for indices in label_to_indices.values():
        for offset, index in enumerate(indices):
            folds[offset % fold_count].append(index)
    return folds
