"""Estimadores propios y persistencia de modelos.

Viven en un módulo importable (no en `train.py`) para que `joblib` pueda deserializarlos
desde `predict.py`: una clase definida en `__main__` no se encuentra al cargar.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin, RegressorMixin

from mibici_mlops_module.config import MODELS_DIR


class BaselineColumna(BaseEstimator, RegressorMixin):
    """Baseline de P1: predice la salida con el valor de una columna ya existente.

    `salidas_t` es persistencia (t+1 = t) y `salidas_lag_168h` es el estacional semanal
    (misma hora de la semana pasada, que para t+1 es el rezago de t-167).
    """

    def __init__(self, columna: str = "salidas_t"):
        self.columna = columna

    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.columna].to_numpy(dtype=float)


class BaselineCuantiles(BaseEstimator, ClusterMixin):
    """Baseline de P3: agrupa las estaciones por cuantiles de su volumen de movimientos."""

    def __init__(self, columna: str = "movimientos_totales", n_grupos: int = 3):
        self.columna = columna
        self.n_grupos = n_grupos

    def fit(self, X: pd.DataFrame, y=None):
        qs = np.linspace(0, 1, self.n_grupos + 1)[1:-1]
        self.cortes_ = X[self.columna].quantile(qs).to_numpy()
        self.labels_ = self.predict(X)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.searchsorted(self.cortes_, X[self.columna].to_numpy(), side="right")


@dataclass
class ModeloGuardado:
    """Modelo entrenado más lo necesario para usarlo sin reentrenar ni adivinar columnas."""

    pregunta: str
    nombre: str
    modelo: object
    features: list[str]
    data_version: str
    git_commit: str
    run_id: str


def ruta_modelo(pregunta: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"{pregunta}.joblib"


def guardar_modelo(m: ModeloGuardado, models_dir: Path = MODELS_DIR) -> Path:
    ruta = ruta_modelo(m.pregunta, models_dir)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, ruta)
    return ruta


def cargar_modelo(pregunta: str, models_dir: Path = MODELS_DIR) -> ModeloGuardado:
    ruta = ruta_modelo(pregunta, models_dir)
    if not ruta.exists():
        raise FileNotFoundError(f"No hay modelo en {ruta}; entrena primero con modeling.train")
    return joblib.load(ruta)
