from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import CLASS_NAMES, SAVE_MODEL_DIR


MODEL_FAMILY_ORDER = ("linear", "rbf", "mlp")
PREFERRED_MODEL_FILENAMES = {
    "linear": ("linear/model_linear.json", "model_linear.json"),
    "rbf": ("rbf/model_rbf.json", "model_rbf.json"),
    "mlp": ("mlp/model_mlp.json", "model_mlp.json"),
}


@dataclass(frozen=True)
class ModelInfo:
    filename: str
    model_type: str | None
    accuracy: float | None
    input_dim: int | None
    output_dim: int | None
    class_names: tuple[str, ...]
    family: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "model_type": self.model_type,
            "accuracy": self.accuracy,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "class_names": self.class_names,
            "family": self.family,
        }


def list_saved_models(save_model_dir: Path = SAVE_MODEL_DIR) -> list[ModelInfo]:
    discovered_models: list[ModelInfo] = []

    if not save_model_dir.exists():
        return discovered_models

    for path in sorted(save_model_dir.rglob("*.json")):
        try:
            payload = _read_model_payload(path)
        except (OSError, json.JSONDecodeError):
            continue

        family = _model_family(payload.get("model_type"), path)
        if family is None or not _model_payload_looks_usable(payload):
            continue

        parameters = _model_parameters(payload)
        layer_sizes = parameters.get("layer_sizes") or []

        discovered_models.append(
            ModelInfo(
                filename=path.relative_to(save_model_dir).as_posix(),
                model_type=payload.get("model_type"),
                accuracy=payload.get("accuracy") or parameters.get("accuracy"),
                input_dim=parameters.get("input_dim") or payload.get("input_dim") or _first_layer_size(layer_sizes),
                output_dim=parameters.get("output_dim") or payload.get("output_dim") or _last_layer_size(layer_sizes),
                class_names=_class_names_from_payload(payload),
                family=family,
            )
        )

    return _select_models_for_interface(discovered_models)


def predict_with_saved_model(model_filename: str, image_vector: list[float]) -> dict[str, Any]:
    path = _resolve_model_path(model_filename)
    payload = _read_model_payload(path)
    parameters = _model_parameters(payload)
    layer_sizes = parameters.get("layer_sizes") or []
    expected_input_dim = int(parameters.get("input_dim") or _first_layer_size(layer_sizes) or 0)
    model_type = payload.get("model_type")

    x = np.asarray(image_vector, dtype=np.float32).reshape(-1)

    if expected_input_dim and x.shape[0] != expected_input_dim:
        raise ValueError(
            f"Format image incompatible : {x.shape[0]} valeurs reçues, "
            f"{expected_input_dim} attendues par le modèle."
        )

    scores = _predict_scores_from_payload(payload, x)
    class_names = _class_names_from_payload(payload)
    winner_index = int(np.nanargmax(scores))
    winner_label = class_names[winner_index]

    return {
        "model": {
            "filename": path.name,
            "model_type": model_type,
            "accuracy": payload.get("accuracy") or parameters.get("accuracy"),
        },
        "prediction": {
            "index": winner_index,
            "label": winner_label,
        },
        "outputs": [
            {
                "label": class_names[index],
                "value": float(value),
                "winner": index == winner_index,
            }
            for index, value in enumerate(scores)
        ],
        "is_probability": False,
        "inference_engine": "json_saved_state",
        "note": _model_output_note(model_type),
    }


def _resolve_model_path(model_filename: str) -> Path:
    save_model_dir = SAVE_MODEL_DIR.resolve()
    candidate = (save_model_dir / Path(model_filename)).resolve()

    if save_model_dir != candidate and save_model_dir not in candidate.parents:
        raise FileNotFoundError(f"Modèle introuvable : {model_filename}")

    if not candidate.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {model_filename}")

    return candidate


def _read_model_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_models_for_interface(models: list[ModelInfo]) -> list[ModelInfo]:
    selected_models: list[ModelInfo] = []

    for family in MODEL_FAMILY_ORDER:
        family_models = [model for model in models if model.family == family]
        selected_model = _select_preferred_model(family, family_models)

        if selected_model is not None:
            selected_models.append(selected_model)

    return selected_models


def _select_preferred_model(family: str, models: list[ModelInfo]) -> ModelInfo | None:
    if not models:
        return None

    models_by_filename = {model.filename: model for model in models}

    for preferred_filename in PREFERRED_MODEL_FILENAMES.get(family, ()):
        if preferred_filename in models_by_filename:
            return models_by_filename[preferred_filename]

    return sorted(models, key=lambda model: model.filename)[0]


def _model_family(model_type: str | None, path: Path) -> str | None:
    normalized_type = (model_type or "").lower()
    path_parts = {part.lower() for part in path.parts}
    filename = path.name.lower()

    if normalized_type == "lineaire" or "linear" in path_parts or filename.startswith("model_linear"):
        return "linear"

    if normalized_type == "ovr_rbf" or "rbf" in path_parts or filename.startswith("model_rbf"):
        return "rbf"

    if _is_mlp_model_type(model_type) or "mlp" in path_parts or filename.startswith("model_mlp"):
        return "mlp"

    return None


def _model_payload_looks_usable(payload: dict[str, Any]) -> bool:
    model_type = payload.get("model_type")

    if model_type == "Lineaire":
        return bool(payload.get("submodels"))

    if model_type == "ovr_rbf":
        return bool(payload.get("submodels"))

    if _is_mlp_model_type(model_type):
        parameters = _model_parameters(payload)
        layer_sizes = parameters.get("layer_sizes") or []
        return bool(layer_sizes and (payload.get("weights") or payload.get("poids")))

    return False


def _class_names_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_class_names = payload.get("class_names")
    parameters = _model_parameters(payload)
    layer_sizes = parameters.get("layer_sizes") or []
    output_dim = int(parameters.get("output_dim") or _last_layer_size(layer_sizes) or len(CLASS_NAMES))

    if (
        isinstance(raw_class_names, list)
        and len(raw_class_names) == output_dim
        and all(isinstance(name, str) for name in raw_class_names)
    ):
        return tuple(raw_class_names)

    return CLASS_NAMES[:output_dim]


def _model_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("parametres") or payload.get("parameters") or {}


def _first_layer_size(layer_sizes: list[Any]) -> int | None:
    return int(layer_sizes[0]) if layer_sizes else None


def _last_layer_size(layer_sizes: list[Any]) -> int | None:
    return int(layer_sizes[-1]) if layer_sizes else None


def _predict_scores_from_payload(payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    model_type = payload.get("model_type")

    if model_type == "Lineaire":
        return _predict_linear_scores(payload, x)

    if model_type == "ovr_rbf":
        return _predict_rbf_decisions(payload, x)

    if _is_mlp_model_type(model_type):
        return _predict_mlp_scores(payload, x)

    raise ValueError(f"Type de modèle non supporté : {model_type!r}")


def _predict_linear_scores(payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    scores: list[float] = []

    for submodel in payload.get("submodels", []):
        weights = np.asarray(submodel["poids"], dtype=np.float32)
        bias = float(submodel["biais"])
        scores.append(float(np.dot(weights, x) + bias))

    return np.asarray(scores, dtype=np.float32)


def _predict_rbf_decisions(payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    decisions: list[float] = []

    for submodel in payload.get("submodels", []):
        clusters = np.asarray(submodel["clusters"], dtype=np.float32)
        weights = np.asarray(submodel["poids"], dtype=np.float32)
        gamma = float(submodel["gamma"])

        squared_distances = np.sum((clusters - x) ** 2, axis=1)
        activations = np.exp(-gamma * squared_distances)
        raw_score = float(np.dot(weights, activations))

        decisions.append(1.0 if raw_score >= 0.0 else -1.0)

    return np.asarray(decisions, dtype=np.float32)


def _predict_mlp_scores(payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    weights = payload.get("weights") or payload.get("poids")

    if weights is None:
        raise ValueError(
            "Modèle MLP détecté, mais son JSON ne contient pas encore de poids exploitables "
            "par l'interface. Il faudra brancher le format exact de sauvegarde MLP."
        )

    parameters = _model_parameters(payload)
    layer_sizes = [int(size) for size in parameters.get("layer_sizes", [])]
    is_classification = int(parameters.get("task_mode", 1)) == 1

    if not layer_sizes:
        raise ValueError("Modèle MLP détecté, mais son JSON ne contient pas 'parameters.layer_sizes'.")

    return _predict_flat_rust_mlp_weights(weights, layer_sizes, x, is_classification=is_classification)


def _predict_flat_rust_mlp_weights(
    weights: list[Any],
    layer_sizes: list[int],
    x: np.ndarray,
    is_classification: bool,
) -> np.ndarray:
    flat_weights = np.asarray(weights, dtype=np.float64)
    activation = np.asarray(x, dtype=np.float64)
    cursor = 0

    if activation.shape[0] != layer_sizes[0]:
        raise ValueError(
            f"Format image incompatible : {activation.shape[0]} valeurs reçues, "
            f"{layer_sizes[0]} attendues par le MLP."
        )

    for layer_index in range(1, len(layer_sizes)):
        previous_size = layer_sizes[layer_index - 1]
        current_size = layer_sizes[layer_index]
        matrix_size = (previous_size + 1) * (current_size + 1)
        matrix_values = flat_weights[cursor : cursor + matrix_size]

        if matrix_values.size != matrix_size:
            raise ValueError("Format de poids MLP incomplet ou invalide.")

        matrix = matrix_values.reshape(previous_size + 1, current_size + 1)
        cursor += matrix_size

        activation_with_bias = np.concatenate(([1.0], activation))
        next_values = activation_with_bias @ matrix[:, 1:]
        is_last_layer = layer_index == len(layer_sizes) - 1
        activation = np.tanh(next_values) if is_classification or not is_last_layer else next_values

    if cursor != flat_weights.size:
        raise ValueError("Format de poids MLP invalide : taille des poids incohérente.")

    return activation.astype(np.float32)


def _model_output_note(model_type: str | None) -> str:
    if model_type == "ovr_rbf":
        return "Les sorties RBF actuelles sont des décisions -1/+1, pas des probabilités."

    if _is_mlp_model_type(model_type):
        return "Les sorties MLP affichées sont les activations du réseau, pas des probabilités."

    return "Les sorties affichées sont des scores modèle bruts, pas des probabilités."


def _is_mlp_model_type(model_type: str | None) -> bool:
    return model_type in {"MLP", "mlp", "PMC", "pmc", "MultilayerPerceptron"}
