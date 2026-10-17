"""Baselines propios y persistencia de modelos.

Los baselines (`BaselineColumna` para P1 y `BaselineCuantiles` para P3) son la referencia
mínima que cualquier modelo debe superar. También aquí se guardan y cargan los modelos
entrenados (`ModeloGuardado`).

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

    `salidas_t` es persistencia (t+1 = t) y `salidas_lag_168h` es el estacional semanal (misma hora
    de la semana pasada; para t+1 es el rezago de t-167).

    Por qué es pertinente: son la referencia más simple posible. Si un modelo no las supera, no
    justifica su complejidad (Huyen, cap. 6).
    """

    def __init__(self, columna: str = "salidas_t"):
        """`columna` es la variable de la tabla que se usa como predicción."""
        self.columna = columna

    def fit(self, X, y=None):
        """No aprende nada; existe para cumplir la interfaz de scikit-learn."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve la columna elegida como predicción."""
        return X[self.columna].to_numpy(dtype=float)


class BaselineCuantiles(BaseEstimator, ClusterMixin):
    """Baseline de P3: agrupa las estaciones por cuantiles de su volumen de movimientos.

    Por qué es pertinente: es la tipología más simple (estaciones de poco, medio y mucho tráfico).
    K-Means solo se justifica si encuentra estructura que el volumen por sí solo no explica.
    """

    def __init__(self, columna: str = "movimientos_totales", n_grupos: int = 3):
        """`columna` es la variable de volumen y `n_grupos` el número de cuantiles."""
        self.columna = columna
        self.n_grupos = n_grupos

    def fit(self, X: pd.DataFrame, y=None):
        """Calcula los cortes de cada grupo con los cuantiles de la columna de volumen."""
        qs = np.linspace(0, 1, self.n_grupos + 1)[1:-1]
        self.cortes_ = X[self.columna].quantile(qs).to_numpy()
        self.labels_ = self.predict(X)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Asigna a cada estación el grupo según los cortes calculados."""
        return np.searchsorted(self.cortes_, X[self.columna].to_numpy(), side="right")


@dataclass
class ModeloGuardado:
    """Modelo entrenado más lo necesario para usarlo sin reentrenar ni adivinar columnas.

    Por qué es pertinente: guarda las variables exactas, la versión de los datos, el commit y el
    `run_id` de MLflow, de modo que cualquier predicción se puede rastrear hasta el run que la produjo.
    """

    pregunta: str
    nombre: str
    modelo: object
    features: list[str]
    data_version: str
    git_commit: str
    run_id: str


def ruta_modelo(pregunta: str, models_dir: Path = MODELS_DIR) -> Path:
    """Ruta del archivo del modelo de una pregunta: `models/<pregunta>.joblib`."""
    return models_dir / f"{pregunta}.joblib"


def guardar_modelo(m: ModeloGuardado, models_dir: Path = MODELS_DIR) -> Path:
    """Guarda el modelo y sus metadatos en `models/` (se crea la carpeta si no existe)."""
    ruta = ruta_modelo(m.pregunta, models_dir)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, ruta)
    return ruta


def cargar_modelo(pregunta: str, models_dir: Path = MODELS_DIR) -> ModeloGuardado:
    """Carga el modelo de una pregunta; si no existe, indica que primero hay que entrenar."""
    ruta = ruta_modelo(pregunta, models_dir)
    if not ruta.exists():
        raise FileNotFoundError(f"No hay modelo en {ruta}; entrena primero con modeling.train")
    return joblib.load(ruta)
