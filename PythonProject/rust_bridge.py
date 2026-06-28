from __future__ import annotations

import ctypes
from enum import IntEnum
from pathlib import Path

import numpy as np


class TaskMode(IntEnum):
    REGRESSION = 0
    CLASSIFICATION = 1


def _default_library_path() -> Path:
    # search of the compiled rust library
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "lib_classification" / "target" / "debug" / "lib_classification.dll",
        root / "lib_classification" / "target" / "release" / "lib_classification.dll",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No compiled lib_classification.dll found. Run `cargo build` in lib_classification first."
    )


def _as_float32(values: np.ndarray | list[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32))


class RustLib:
    def __init__(self, library_path: str | Path | None = None) -> None:
        self.path = Path(library_path) if library_path else _default_library_path()
        self.lib = ctypes.cdll.LoadLibrary(str(self.path))
        self._configure_signatures()

    def _configure_signatures(self) -> None:

        #----Declaration du modèle linéaire via ffi signatures----
        self.lib.linear_model_create.argtypes = [ctypes.c_size_t, ctypes.c_float]
        self.lib.linear_model_create.restype = ctypes.c_void_p

        self.lib.linear_model_train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self.lib.linear_model_train.restype = ctypes.c_int32

        self.lib.linear_model_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.linear_model_predict.restype = ctypes.c_float

        self.lib.linear_model_free.argtypes = [ctypes.c_void_p]
        self.lib.linear_model_free.restype = None


        #----Declaration du MLP via ffi signatures----
        self.lib.mlp_create.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
        self.lib.mlp_create.restype = ctypes.c_void_p

        self.lib.mlp_train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_ubyte,
        ]
        self.lib.mlp_train.restype = ctypes.c_int32

        self.lib.mlp_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_ubyte,
        ]
        self.lib.mlp_predict.restype = ctypes.c_int32

        self.lib.mlp_output_dim.argtypes = [ctypes.c_void_p]
        self.lib.mlp_output_dim.restype = ctypes.c_size_t

        self.lib.mlp_free.argtypes = [ctypes.c_void_p]
        self.lib.mlp_free.restype = None

        # ----Declaration du modèle RBF + lloyd via ffi signatures----
        self.lib.creation_RBF_model.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self.lib.creation_RBF_model.restype = ctypes.c_void_p

        self.lib.entrainement_RBF_model.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,    # sample_count
            ctypes.c_float,     # mouvement_max
            ctypes.c_float,     # max_loop
            ctypes.c_float      # gamma
        ]

        self.lib.entrainement_RBF_model.restype = ctypes.c_int32

        self.lib.RBF_model_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t
        ]
        self.lib.RBF_model_predict.restype = ctypes.c_float

        self.lib.RBF_model_free.argtypes = [ctypes.c_void_p]
        self.lib.RBF_model_free.restype = None







        # ----Declaration du modèle linéaire ----
        class MyDroite(ctypes.Structure): _fields_ = [
            ("a", ctypes.c_float),
            ("b", ctypes.c_float),
            ("c", ctypes.c_float),
        ]
        self.lib.initialisation_droite.restype = ctypes.POINTER(MyDroite)
        self.lib.training.argtypes = [ctypes.c_float, ctypes.c_int32, ctypes.c_int32, MyDroite,
                                 ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib.training.restype = ctypes.POINTER(MyDroite)
        self.lib.linear_classification_prediction.restype = ctypes.c_float
        self.lib.linear_classification_prediction.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float
        ]

def _check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with status {status}")


class LinearModelRust:
    def __init__(self, input_dim: int, learning_rate: float = 0.01, library: RustLib | None = None) -> None:
        # creation of the rust linear model
        self.library = library or RustLib()
        self.input_dim = int(input_dim)
        self._handle = self.library.lib.linear_model_create(self.input_dim, learning_rate)
        if not self._handle:
            raise RuntimeError("Failed to create Rust linear model")

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1_000) -> None:
        # training of the rust linear model
        x = _as_float32(x)
        y = _as_float32(y).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")
        if y.shape[0] != x.shape[0]:
            raise ValueError("y must contain one target per sample")

        status = self.library.lib.linear_model_train(
            self._handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.shape[0],
            epochs,
        )
        _check_status(status, "linear_model_train")

    def predict(self, x: np.ndarray) -> np.ndarray:
        # prediction with the rust linear model
        x = _as_float32(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")

        predictions = np.zeros((x.shape[0],), dtype=np.float32)
        for index, row in enumerate(x):
            predictions[index] = self.library.lib.linear_model_predict(
                self._handle,
                row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                row.shape[0],
            )
        return predictions

    def predict_labels(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.predict(x) >= 0.0, 1.0, -1.0).astype(np.float32)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.lib.linear_model_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()



class LinearModel:
    def __init__(self, entree_dim: int, pas_apprentissage: float = 0.01, library = None) -> None:
        self.library = library or RustLib()
        self.entree_dim = int(entree_dim)
        self.MyDroite = self.library.lib.initialisation_droite()
        if not self.MyDroite:
            raise RuntimeError("Erreur création modèle linéaire")

    #def entrainement(self, ):

class OVRLinearClassifier:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        library: RustLib | None = None,
    ) -> None:
        # creation of the one-vs-rest linear models
        self.library = library or RustLib()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.models = [
            LinearModelRust(self.input_dim, learning_rate=learning_rate, library=self.library)
            #LinearModel(self.input_dim, learning_rate = learning_rate, library = self.library)
            for _ in range(self.output_dim)
        ]

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1_000) -> None:
        # training of the one-vs-rest linear models
        x = _as_float32(x)
        y = _as_float32(y)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")
        if y.ndim != 2 or y.shape != (x.shape[0], self.output_dim):
            raise ValueError("y must be shaped (n_samples, output_dim)")

        for class_index, model in enumerate(self.models):
            model.fit(x, y[:, class_index], epochs=epochs)

    def predict_labels(self, x: np.ndarray) -> np.ndarray:
        # prediction with the one-vs-rest linear models
        scores = np.stack([model.predict(x) for model in self.models], axis=1)
        winners = np.argmax(scores, axis=1)
        labels = -np.ones((scores.shape[0], self.output_dim), dtype=np.float32)
        labels[np.arange(scores.shape[0]), winners] = 1.0
        return labels

    def close(self) -> None:
        for model in getattr(self, "models", []):
            model.close()

    def __del__(self) -> None:
        self.close()


class MLPRust:
    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.01,
        task_mode: TaskMode = TaskMode.CLASSIFICATION,
        library: RustLib | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")

        # creation of the rust mlp model
        self.library = library or RustLib()
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.input_dim = self.layer_sizes[0]
        self.output_dim = self.layer_sizes[-1]
        self.learning_rate = float(learning_rate)
        self.task_mode = TaskMode(task_mode)

        layer_array = (ctypes.c_size_t * len(self.layer_sizes))(*self.layer_sizes)
        self._handle = self.library.lib.mlp_create(layer_array, len(self.layer_sizes))
        if not self._handle:
            raise RuntimeError("Failed to create Rust MLP")

    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 50_000) -> None:
        # training of the rust mlp model
        x = _as_float32(x)
        y = _as_float32(y)

        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")
        if self.output_dim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2 or y.shape != (x.shape[0], self.output_dim):
            raise ValueError("y must match the MLP output shape")

        status = self.library.lib.mlp_train(
            self._handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.shape[0],
            steps,
            self.learning_rate,
            int(self.task_mode),
        )
        _check_status(status, "mlp_train")

    def predict(self, x: np.ndarray) -> np.ndarray:
        # prediction with the rust mlp model
        x = _as_float32(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")

        outputs = np.zeros((x.shape[0], self.output_dim), dtype=np.float32)
        for index, row in enumerate(x):
            row_output = outputs[index]
            status = self.library.lib.mlp_predict(
                self._handle,
                row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                row_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(self.task_mode),
            )
            _check_status(status, "mlp_predict")

        if self.output_dim == 1:
            return outputs.reshape(-1)
        return outputs

    def predict_labels(self, x: np.ndarray) -> np.ndarray:
        predictions = self.predict(x)
        if self.output_dim == 1:
            return np.where(predictions >= 0.0, 1.0, -1.0).astype(np.float32)

        winners = np.argmax(predictions, axis=1)
        labels = -np.ones_like(predictions)
        labels[np.arange(predictions.shape[0]), winners] = 1.0
        return labels.astype(np.float32)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.lib.mlp_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

class OVRRBF:

    def __init__(self, input_dim, nb_cluster, output_dim, library=None):
        self.input_dim = int(input_dim)   # ← manquait
        self.output_dim = int(output_dim)
        self.library = library
        self.models = [
            RBFModelRust(
                input_dim=input_dim,
                nb_cluster=nb_cluster,
                library=library
            )
            for _ in range(output_dim)
        ]

    def entrainement(self, x, y, mouvement_max, max_loop, gamma):
        x = _as_float32(x)
        y = _as_float32(y)  # ← PAS de .reshape(-1) ici

        for class_index, model in enumerate(self.models):
            y_binary = y[:, class_index]  # ← sélection par colonne
            y_binary = np.where(y_binary > 0, 1.0, -1.0).astype(np.float32)
            model.entrainement(x, y_binary, mouvement_max, max_loop, gamma)

    def prediction(self, x):
        scores = []
        for model in self.models:
            score = model.prediction(x)
            scores.append(score)
        scores = np.stack(scores, axis=1)  # (n_samples, output_dim)

        winners = np.argmax(scores, axis=1)

        # Conversion en one-hot -1/+1
        labels = -np.ones((scores.shape[0], self.output_dim), dtype=np.float32)
        labels[np.arange(scores.shape[0]), winners] = 1.0
        return labels

    def close(self):
        for model in getattr(self, "models", []):
            model.close()

    def __del__(self):
        self.close()


class RBFModelRust:

    def __init__(self, input_dim: int, nb_cluster: int, library: RustLib | None = None) -> None:
        self.library = library or RustLib()
        self.input_dim = int(input_dim)
        self._handle = self.library.lib.creation_RBF_model(
            ctypes.c_size_t(self.input_dim),
            ctypes.c_size_t(int(nb_cluster)),
        )
        if not self._handle:
            raise RuntimeError("Echec lors de la création du modèle RBF")

    def entrainement(self, x: np.ndarray, y: np.ndarray, mouvement_max: float, max_loop: float, gamma: float) -> None:
        x = _as_float32(x)
        y = _as_float32(y).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")
        if y.shape[0] != x.shape[0]:
            raise ValueError("y must contain one target per sample")

        status = self.library.lib.entrainement_RBF_model(
            self._handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(x.shape[0]),
            ctypes.c_float(float(mouvement_max)),
            ctypes.c_float(float(max_loop)),
            ctypes.c_float(float(gamma))
        )
        _check_status(status, "RBF_model_train")

    def prediction(self, x: np.ndarray) -> np.ndarray:
        x = _as_float32(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")

        predictions = np.zeros((x.shape[0],), dtype=np.float32)
        for index, row in enumerate(x):
            predictions[index] = self.library.lib.RBF_model_predict(
                self._handle,
                row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                row.shape[0],
            )
        return predictions

    def prediction_labels(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.prediction(x) >= 0.0, 1.0, -1.0).astype(np.float32)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.lib.RBF_model_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()