from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dataset_loader import DEFAULT_DATASET_ROOT, load_labeled_image_dataset, stratified_split
from rust_bridge import MLPRust, OVRLinearClassifier, TaskMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Rust linear baseline and MLP on the game screenshot dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing FPS, METROIDVANIA and MOBA subfolders.",
    )
    parser.add_argument("--width", type=int, default=int(40 * 16/9), help="Resized image width.")
    parser.add_argument("--height", type=int, default=40, help="Resized image height.")
    parser.add_argument("--grayscale", dest="grayscale", action="store_true", help="Use grayscale images.")
    parser.add_argument("--rgb", dest="grayscale", action="store_false", help="Use RGB images.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction used for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--linear-lr", type=float, default=0.01, help="Learning rate for the linear baseline.")
    parser.add_argument(
        "--linear-steps",
        "--linear-epochs",
        dest="linear_steps",
        type=int,
        default=20_000,
        help="Random training steps for the linear baseline.",
    )
    parser.add_argument("--mlp-lr", type=float, default=0.01, help="Learning rate for the MLP.")
    parser.add_argument(
        "--mlp-steps",
        "--mlp-epochs",
        dest="mlp_steps",
        type=int,
        default=50_000,
        help="Random training steps for the naive MLP.",
    )
    parser.add_argument(
        "--mlp-layers",
        type=int,
        nargs="*",
        default=[16, 8],
        help="Hidden layer sizes for the MLP. Example: --mlp-layers 16 8",
    )
    parser.set_defaults(grayscale=True)
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    # loading the dataset and creating the split
    bundle = load_labeled_image_dataset(
        root=args.root,
        image_size=(args.width, args.height),
        grayscale=args.grayscale,
    )
    split = stratified_split(bundle, test_ratio=args.test_ratio, seed=args.seed)

    # display of the dataset information
    print("=== Dataset ===")
    print(f"Chemin: {args.root}")
    print(f"Classes: {', '.join(bundle.class_names)}")
    print(f"Nb images par classe: {bundle.counts_by_class}")
    color_mode = "grayscale" if args.grayscale else "rgb"
    print(f"Format apres pretraitement: {args.width}x{args.height} {color_mode}")
    print(f"Nb features: {bundle.x.shape[1]}")
    print(f"Nb images train: {split.x_train.shape[0]}")
    print(f"Nb images test: {split.x_test.shape[0]}")
    print(f"Images ignorees: {len(bundle.skipped_paths)}")
    if bundle.skipped_paths:
        for skipped_path in bundle.skipped_paths[:5]:
            print(f"  - {skipped_path.name}")
    print()

    # training of the linear model
    linear = OVRLinearClassifier(
        input_dim=split.x_train.shape[1],
        output_dim=len(bundle.class_names),
        learning_rate=args.linear_lr,
    )
    linear.fit(split.x_train, split.y_train, epochs=args.linear_steps)
    linear_predictions = linear.predict_labels(split.x_test)
    linear.close()

    # training of the mlp model
    mlp = MLPRust(
        layer_sizes=[split.x_train.shape[1], *args.mlp_layers, len(bundle.class_names)],
        learning_rate=args.mlp_lr,
        task_mode=TaskMode.CLASSIFICATION,
    )
    mlp.fit(split.x_train, split.y_train, steps=args.mlp_steps)
    mlp_predictions = mlp.predict_labels(split.x_test)
    mlp.close()

    # display of the results
    print("=== Resultats ===")
    _print_results("Linear", split.y_test, linear_predictions, bundle.class_names)
    print()
    _print_results("MLP", split.y_test, mlp_predictions, bundle.class_names)
    print()
    print(f"Temps total: {time.perf_counter() - start_time:.2f}s")


def _print_results(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
) -> None:
    accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
    confusion = _confusion_matrix(y_true, y_pred, len(class_names))

    # display of one result block
    print(f"{title}:")
    print(f"Accuracy: {accuracy:.3f}")
    print("Matrice de confusion (lignes=reel, colonnes=predit) :")
    print("       " + " ".join(f"{name[:5]:>6}" for name in class_names))
    for row_index, row in enumerate(confusion):
        print(f"{class_names[row_index][:5]:>5} " + " ".join(f"{value:>6}" for value in row))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int32)
    true_indices = np.argmax(y_true, axis=1)
    pred_indices = np.argmax(y_pred, axis=1)

    for true_index, pred_index in zip(true_indices, pred_indices):
        matrix[true_index, pred_index] += 1

    return matrix


if __name__ == "__main__":
    main()
