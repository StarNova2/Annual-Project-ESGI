from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DATASET_ROOT = Path(
    r"G:\.shortcut-targets-by-id\1iAfi-pFGoqzi63RuDrtZzG3SU73_RGLr\Dataset projet annuel"
)
DEFAULT_CLASS_NAMES = ("FPS", "METROIDVANIA", "MOBA")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    class_names: tuple[str, ...]
    image_paths: list[Path]
    counts_by_class: dict[str, int]
    image_size: tuple[int, int]
    skipped_paths: list[Path]


@dataclass(frozen=True)
class SplitDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_paths: list[Path]
    test_paths: list[Path]


def load_labeled_image_dataset(
    root: Path | str = DEFAULT_DATASET_ROOT,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    image_size: tuple[int, int] = (8, 6),
    grayscale: bool = True,
) -> Dataset:
    # creation of the dataset lists
    root = Path(root)
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    image_paths: list[Path] = []
    counts_by_class: dict[str, int] = {}
    skipped_paths: list[Path] = []

    # loading images
    for class_index, class_name in enumerate(class_names):
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(f"No supported image files found in: {class_dir}")

        counts_by_class[class_name] = 0

        for path in files:
            try:
                x_rows.append(_load_image_vector(path, image_size=image_size, grayscale=grayscale))
            except Exception as exc:
                skipped_paths.append(path)
                print(
                    f"[dataset_loader] Image ignoree: {path} "
                    f"({type(exc).__name__}: {exc})"
                )
                continue

            y_rows.append(_one_vs_rest_label(class_index, len(class_names)))
            image_paths.append(path)
            counts_by_class[class_name] += 1

        if counts_by_class[class_name] == 0:
            raise FileNotFoundError(f"No readable image files found in: {class_dir}")

    # creation of the dataset bundle
    return Dataset(
        x=np.stack(x_rows).astype(np.float32),
        y=np.stack(y_rows).astype(np.float32),
        class_names=class_names,
        image_paths=image_paths,
        counts_by_class=counts_by_class,
        image_size=image_size,
        skipped_paths=skipped_paths,
    )


def stratified_split(
    bundle: Dataset,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SplitDataset:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")

    # creation of the split indices
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_index in range(len(bundle.class_names)):
        class_indices = np.where(bundle.y[:, class_index] == 1.0)[0]
        class_indices = rng.permutation(class_indices)

        split_index = int(len(class_indices) * (1.0 - test_ratio))
        split_index = max(1, split_index)
        split_index = min(split_index, len(class_indices) - 1)

        train_indices.extend(class_indices[:split_index].tolist())
        test_indices.extend(class_indices[split_index:].tolist())

    train_indices = rng.permutation(np.asarray(train_indices, dtype=np.intp))
    test_indices = rng.permutation(np.asarray(test_indices, dtype=np.intp))

    # creation of the split bundle
    return SplitDataset(
        x_train=bundle.x[train_indices],
        y_train=bundle.y[train_indices],
        x_test=bundle.x[test_indices],
        y_test=bundle.y[test_indices],
        train_paths=[bundle.image_paths[index] for index in train_indices],
        test_paths=[bundle.image_paths[index] for index in test_indices],
    )


def _load_image_vector(path: Path, image_size: tuple[int, int], grayscale: bool) -> np.ndarray:
    # loading and preprocessing one image
    with Image.open(path) as image:
        image = image.convert("L" if grayscale else "RGB")
        image = image.resize(image_size, Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)

    pixels = pixels / 127.5 - 1.0
    return pixels.reshape(-1)


def _one_vs_rest_label(class_index: int, class_count: int) -> np.ndarray:
    label = np.full((class_count,), -1.0, dtype=np.float32)
    label[class_index] = 1.0
    return label
