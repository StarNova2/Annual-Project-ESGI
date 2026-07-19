from __future__ import annotations

import ctypes
from enum import IntEnum
from pathlib import Path
import json

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
        self.lib.linear_model_create.argtypes = [ctypes.c_size_t, ctypes.c_float, ctypes.c_uint64]
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

        self.lib.linear_model_regression.restype = None
        self.lib.linear_model_regression.argtypes = [ctypes.c_void_p,
                                                     ctypes.POINTER(ctypes.c_float),
                                                     ctypes.POINTER(ctypes.c_float),
                                                     ctypes.c_size_t,
                                                     ctypes.c_size_t
                                                     ]
        self.lib.linear_model_get_poids.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.linear_model_get_poids.restype = ctypes.c_int32

        self.lib.linear_model_get_bias.argtypes = [ctypes.c_void_p]
        self.lib.linear_model_get_bias.restype = ctypes.c_float

        self.lib.linear_model_set_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_float]
        self.lib.linear_model_set_state.restype = ctypes.c_int32


        #----Declaration du MLP via ffi signatures----
        self.lib.mlp_create.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_size_t]
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

        self.lib.mlp_get_weights.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.mlp_get_weights.restype = ctypes.POINTER(ctypes.c_double)

        self.lib.mlp_free_weights.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]

        self.lib.mlp_set_weights.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self.lib.mlp_set_weights.restype = ctypes.c_int32
        
        self.lib.mlp_get_loss_history.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.mlp_get_loss_history.restype = ctypes.POINTER(ctypes.c_double)

        self.lib.mlp_free_loss_history.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        self.lib.mlp_free_loss_history.restype = None

        self.lib.mlp_free.argtypes = [ctypes.c_void_p]
        self.lib.mlp_free.restype = None

        # ----Declaration du modèle RBF + lloyd via ffi signatures----
        self.lib.creation_RBF_model.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64]
        self.lib.creation_RBF_model.restype = ctypes.c_void_p

        self.lib.entrainement_RBF_model.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,    # sample_count
            ctypes.c_float,     # mouvement_max
            ctypes.c_float,     # max_loop
            ctypes.c_float     # gamma
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

        self.lib.RBF_model_get_gamma.argtypes = [ctypes.c_void_p]
        self.lib.RBF_model_get_gamma.restype = ctypes.c_float

        self.lib.RBF_model_set.restype = ctypes.c_int32
        self.lib.RBF_model_set.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.c_float]

        self.lib.RBF_model_get_nb_cluster.argtye = [ctypes.c_void_p,
                                                    ctypes.POINTER(ctypes.c_float)]
        self.lib.RBF_model_get_nb_cluster.restype = ctypes.c_int32

        self.lib.RBF_model_get_clusters.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.RBF_model_get_clusters.restype = ctypes.c_int

        self.lib.RBF_model_get_poids.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.RBF_model_get_poids.restype = ctypes.c_int

        self.lib.RBF_model_predict_score.argtypes = [ctypes.c_void_p,
                                                     ctypes.POINTER(ctypes.c_float),
                                                     ctypes.c_size_t]
        self.lib.RBF_model_predict_score.restype = ctypes.c_float







def _check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with status {status}")


class LinearModelRust:
    def __init__(self, input_dim: int, learning_rate: float = 0.01 , seed = 42) -> None:
        # creation of the rust linear model
        self.library = RustLib()
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

        self.input_dim = int(input_dim)
        self._handle = self.library.lib.linear_model_create(self.input_dim, learning_rate, seed)
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
            epochs
        )
        _check_status(status, "linear_model_train")


    def regression(self,x : np.ndarray, y: np.ndarray):
        x = _as_float32(x)
        y = _as_float32(y)


        status = self.library.lib.linear_model_regression(
            self._handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.shape[0],
            x.shape[1]
        )
        #_check_status(status, "linear_model_regression")


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


    def get_poids(self) -> np.ndarray:
        out = np.zeros(self.input_dim, dtype=np.float32)
        status = self.library.lib.linear_model_get_poids(
            self._handle, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        _check_status(status, "linear_model_get_poids")
        return out


    def get_bias(self) -> float:
        return float(self.library.lib.linear_model_get_bias(self._handle))


    def set_state(self, poids: np.ndarray, bias: float) -> None:
        poids = _as_float32(poids).reshape(-1)
        status = self.library.lib.linear_model_set_state(
            self._handle,
            poids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(float(bias))
        )
        _check_status(status, "linear_model_set_state")


    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.lib.linear_model_free(self._handle)
            self._handle = None



    def __del__(self) -> None:
        self.close()


class OVRLinearClassifier:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        library: RustLib | None = None,
        seed = 42,
        accuracy : float = 0.0
    ) -> None:
        # creation of the one-vs-rest linear models
        self.library = RustLib()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.accuracy = float(accuracy)
        self.models = [
            LinearModelRust(self.input_dim, learning_rate=learning_rate, seed=seed+i)
            #LinearModel(self.input_dim, learning_rate = learning_rate, library = self.library)
            for i in range(self.output_dim)
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

    def regression(self, x: np.array, y: np.array):
        x = _as_float32(x)
        y = _as_float32(y)

        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")
        if y.ndim != 2 or y.shape != (x.shape[0], self.output_dim):
            raise ValueError("y must be shaped (n_samples, output_dim)")

        for class_index, model in enumerate(self.models):
            model.regression(x, y[:, class_index])


    def prediction(self, x: np.array) -> np.ndarray:
        return np.stack([model.predict(x) for model in self.models], axis=1)


    def predict_labels(self, x: np.ndarray) -> np.ndarray:
        # prediction with the one-vs-rest linear models
        scores = self.prediction(x)
        winners = np.argmax(scores, axis=1)
        labels = -np.ones((scores.shape[0], self.output_dim), dtype=np.float32)
        labels[np.arange(scores.shape[0]), winners] = 1.0
        return labels

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    @classmethod
    def charge(cls, path: str | Path) -> "OVRLinearClassifier":
        data = json.loads(Path(path).read_text())
        if data["model_type"] != "Lineaire":
            raise ValueError(f"Fichier incompatible : model_type={data['model_type']!r}")
        champs = data["parametres"]

        ovr = cls(
            input_dim=champs["input_dim"],
            output_dim=champs["output_dim"],
            learning_rate=champs["learning_rate"],
            seed=champs.get("seed", 42)
        )

        for model, submodel_data in zip(ovr.models, data["submodels"]):

            weights = np.array(submodel_data["poids"], dtype=np.float32)
            model.set_state(weights, submodel_data["biais"])

        ovr.class_names = tuple(data["class_names"])
        return ovr

    def sauvegarde(self, path: str | Path, log: dict, epochs) -> None:
        data = {
            "model_type": "Lineaire",
            "accuracy": self.accuracy,
            "parametres": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
                "log_loss": log
            },
            "class_names": list(getattr(self, "class_names", range(self.output_dim))),
            "epochs": epochs,
            "submodels": [
                {"poids": model.get_poids().tolist(), "biais": model.get_bias()}
                for model in self.models
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))



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
        seed: int = 42,
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
        self.seed = seed

        layer_array = (ctypes.c_size_t * len(self.layer_sizes))(*self.layer_sizes)
        self._handle = self.library.lib.mlp_create(layer_array, len(self.layer_sizes),seed)
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
    
    def get_loss_history(self) -> np.ndarray:
        if self._handle is None:
            raise ValueError("Model is closed.")
        
        length = ctypes.c_size_t(0)
        ptr = self.library.lib.mlp_get_loss_history(self._handle, ctypes.byref(length))
        
        if not ptr:
            return np.array([], dtype=np.float64)
        
        # Conversion du pointeur brut en tableau numpy
        size = length.value
        buffer = ctypes.string_at(ptr, size * ctypes.sizeof(ctypes.c_double))
        loss_array = np.frombuffer(buffer, dtype=np.float64).copy()
        
        # Libération de la mémoire allouée par Rust
        self.library.lib.mlp_free_loss_history(ptr, size)
        
        return loss_array

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.lib.mlp_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()
        
    def save(self, path: str | Path, extra_hparams: dict | None = None) -> None:
        length = ctypes.c_size_t()

        ptr = self.library.lib.mlp_get_weights(self._handle, ctypes.byref(length))
        weights = np.ctypeslib.as_array(ptr, shape=(length.value,)).copy()
        self.library.lib.mlp_free_weights(ptr, length.value)
        
        data = {
            "model_type": "mlp",
            "parameters": {
                "layer_sizes": self.layer_sizes,
                "learning_rate": self.learning_rate,
                "task_mode": int(self.task_mode),
                "seed": self.seed,
                **(extra_hparams or {}),
            },
            "class_names": list(self.class_names),
            "weights": weights.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        
    @classmethod
    def load(cls, path: str | Path) -> "MLPRust":
        data = json.loads(Path(path).read_text())

        if data["model_type"] != "mlp":
            raise ValueError(
                f"Fichier incompatible : model_type={data['model_type']!r}"
            )

        hp = data["parameters"]

        model = cls(
            layer_sizes=hp["layer_sizes"],
            learning_rate=hp["learning_rate"],
            task_mode=TaskMode(hp["task_mode"]),
            seed=hp.get("seed", 42),
        )

        weights = np.asarray(data["weights"], dtype=np.float64)

        status = model.library.lib.mlp_set_weights(
            model._handle,
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            weights.size,
        )
        _check_status(status, "mlp_set_weights")

        model.class_names = tuple(data.get("class_names", range(model.output_dim)))

        return model

class OVRRBF:

    def __init__(self, input_dim, nb_cluster, output_dim, library=None, seed = 42, accuracy : float = 0.0):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.library = library
        self.seed = int(seed)
        self.accuracy = float(accuracy)
        self.models = [
            RBFModelRust(
                input_dim=input_dim,
                nb_cluster=nb_cluster,
                library=library,
                seed=seed+i
            )
            for i in range(output_dim)
        ]

    def entrainement(self, x, y, mouvement_max, max_loop, gamma):
        x = _as_float32(x)
        y = _as_float32(y)

        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x must be shaped (n_samples, input_dim)")
        if y.ndim != 2 or y.shape != (x.shape[0], self.output_dim):
            raise ValueError("y must be shaped (n_samples, output_dim)")

        for class_index, model in enumerate(self.models):
            y_binary = y[:, class_index]
            y_binary = np.where(y_binary > 0, 1.0, -1.0).astype(np.float32)
            model.entrainement(x, y_binary, mouvement_max, max_loop, gamma)


    def prediction(self, x):
        scores = []
        for model in self.models:
            scores.append(model.prediction(x))

        return np.stack(scores, axis=1)

    def prediction_labels(self, x):
        scores = self.prediction(x)

        winners = np.argmax(scores, axis=1)

        labels = -np.ones((scores.shape[0], self.output_dim), dtype=np.float32)
        labels[np.arange(scores.shape[0]), winners] = 1.0

        return labels


    def close(self):
        for model in getattr(self, "models", []):
            model.close()

    def sauvegarde(self, path: str | Path, log_loss: dict, nb_cluster, mouvement_max, gamma, max_loop) -> None:
        data = {
            "model_type": "ovr_rbf",
            "accuracy": self.accuracy,
            "parametres": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "seed": self.seed,
                "nb_cluster": nb_cluster,
                "mouvement_max": mouvement_max,
                "max_loop": max_loop,
                "gamma": gamma,

            },
            "class_names": list(getattr(self, "class_names", range(self.output_dim))),
            "log_loss": log_loss,
            "submodels": [
                {
                    "gamma": model.get_gamma(),
                    "clusters": model.get_clusters().tolist(),
                    "poids": model.get_poids().tolist()
                }
                for model in self.models
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def charge(cls, path: str | Path, library: RustLib | None = None) -> "OVRRBF":
        data = json.loads(Path(path).read_text())
        if data["model_type"] != "ovr_rbf":
            raise ValueError(f"Fichier incompatible : model_type={data['model_type']!r}")

        hp = data["parametres"]
        nb_clusters = len(data["submodels"][0]["clusters"])

        ovr = cls(
            input_dim=hp["input_dim"],
            output_dim=hp["output_dim"],
            nb_cluster=nb_clusters,
            seed=hp.get("seed", 42),
            library=library
        )

        for model, submodel_data in zip(ovr.models, data["submodels"]):
            clusters = np.array(submodel_data["clusters"], dtype=np.float32)
            weights = np.array(submodel_data["poids"], dtype=np.float32)  # <- "poids", pas "weights"
            model.set_state(clusters, weights, submodel_data["gamma"])

        ovr.class_names = tuple(data["class_names"])
        return ovr


    def set_accuracy(self, accuracy):
        self.accuracy = accuracy


    def __del__(self):
        self.close()


class RBFModelRust:

    def __init__(self, input_dim: int, nb_cluster: int, library: RustLib | None = None, seed = 42) -> None:
        self.library = library or RustLib()
        self.nb_cluster = nb_cluster
        self.input_dim = int(input_dim)
        self.seed = seed
        self._handle = self.library.lib.creation_RBF_model(
            ctypes.c_size_t(self.input_dim),
            ctypes.c_size_t(int(nb_cluster)),
            ctypes.c_uint64(self.seed)
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
        #self.library.lib.entrainement_RBF_model(
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
            predictions[index] = self.library.lib.RBF_model_predict_score(
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


    def get_gamma(self) -> float:
        return float(self.library.lib.RBF_model_get_gamma(self._handle))

    def get_clusters(self) -> np.ndarray:
        out = np.zeros(self.nb_cluster * self.input_dim, dtype=np.float32)

        status = self.library.lib.RBF_model_get_clusters(self._handle, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        _check_status(status, "RBF_model_get_clusters")
        return out.reshape(self.nb_cluster, self.input_dim)

    def get_poids(self) -> np.ndarray:
        out = np.zeros(self.nb_cluster, dtype=np.float32)
        status = self.library.lib.RBF_model_get_poids(self._handle, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        _check_status(status, "RBF_model_get_poids")
        return out

    def set_state(self, clusters: np.ndarray, weights: np.ndarray, gamma: float) -> None:
        clusters = _as_float32(clusters).reshape(-1)
        weights = _as_float32(weights).reshape(-1)
        status = self.library.lib.RBF_model_set(
            self._handle,
            clusters.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(float(gamma)),
        )
        _check_status(status, "RBF_model_set_state")


    def __del__(self) -> None:
        self.close()