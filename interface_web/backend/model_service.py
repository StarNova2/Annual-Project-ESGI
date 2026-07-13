from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import CLASS_NAMES, SAVE_MODEL_DIR


@dataclass(frozen=True)
class ModelInfo:
    filename: str
    model_type: str | None
    accuracy: float | None
    input_dim: int | None
    output_dim: int | None
    class_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "model_type": self.model_type,
            "accuracy": self.accuracy,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "class_names": self.class_names,
        }


def list_saved_models(save_model_dir: Path = SAVE_MODEL_DIR) -> list[ModelInfo]:
    models: list[ModelInfo] = []

    if not save_model_dir.exists():
        return models

    for path in sorted(save_model_dir.glob("*.json")):
        try:
            payload = _read_model_payload(path)
        except (OSError, json.JSONDecodeError):
            continue

        parameters = payload.get("parametres", {})

        models.append(
            ModelInfo(
                filename=path.name,
                model_type=payload.get("model_type"),
                accuracy=payload.get("accuracy"),
                input_dim=parameters.get("input_dim") or payload.get("input_dim"),
                output_dim=parameters.get("output_dim") or payload.get("output_dim"),
                class_names=_class_names_from_payload(payload),
            )
        )

    return models


def predict_with_saved_model(model_filename: str, image_vector: list[float]) -> dict[str, Any]:
    path = _resolve_model_path(model_filename)
    payload = _read_model_payload(path)
    parameters = payload.get("parametres", {})
    expected_input_dim = int(parameters.get("input_dim", 0))
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
            "accuracy": payload.get("accuracy"),
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
    candidate = (save_model_dir / Path(model_filename).name).resolve()

    if candidate.parent != save_model_dir or not candidate.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {model_filename}")

    return candidate


def _read_model_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _class_names_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_class_names = payload.get("class_names")
    output_dim = int(payload.get("parametres", {}).get("output_dim", len(CLASS_NAMES)))

    if (
        isinstance(raw_class_names, list)
        and len(raw_class_names) == output_dim
        and all(isinstance(name, str) for name in raw_class_names)
    ):
        return tuple(raw_class_names)

    return CLASS_NAMES[:output_dim]


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

    return _predict_rust_mlp_weights(weights, x)


def _predict_rust_mlp_weights(weights: list[Any], x: np.ndarray) -> np.ndarray:
    activation = np.asarray(x, dtype=np.float64)

    for layer_index, layer_weights in enumerate(weights[1:], start=1):
        matrix = np.asarray(layer_weights, dtype=np.float64)

        if matrix.ndim != 2 or matrix.shape[0] != activation.shape[0] + 1:
            raise ValueError(
                "Format de poids MLP non reconnu. L'interface attend le format Rust "
                "weights[layer][previous_neuron_with_bias][current_neuron_with_unused_zero]."
            )

        activation_with_bias = np.concatenate(([1.0], activation))
        next_values = activation_with_bias @ matrix[:, 1:]
        activation = np.tanh(next_values)

        if layer_index == len(weights) - 1:
            return activation.astype(np.float32)

    raise ValueError("Format de poids MLP invalide : aucune couche exploitable.")


def _model_output_note(model_type: str | None) -> str:
    if model_type == "ovr_rbf":
        return "Les sorties RBF actuelles sont des décisions -1/+1, pas des probabilités."

    if _is_mlp_model_type(model_type):
        return "Les sorties MLP affichées sont les activations du réseau, pas des probabilités."

    return "Les sorties affichées sont des scores modèle bruts, pas des probabilités."


def _is_mlp_model_type(model_type: str | None) -> bool:
    return model_type in {"MLP", "mlp", "PMC", "pmc", "MultilayerPerceptron"}
