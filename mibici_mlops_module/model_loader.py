
"""Carga modelos de MiBici desde MLflow Model Registry.

Permite cargar los modelos P1, P2 y P3 mediante
su alias (champion/challenger) o versión registrada.
"""

import os

import mlflow
import mlflow.pyfunc

from mibici_mlops_module.modeling.registry import nombre_registro


def cargar_modelo_registry(
    pregunta: str = "p1",
    alias: str = "champion",
    version: str | int | None = None,
):
    """Carga un modelo desde MLflow para realizar inferencias.

    Args:
        pregunta: p1 (demanda), p2 (riesgo) o p3 (grupos).
        alias: Alias registrado en MLflow, por defecto champion.
        version: Versión específica; tiene prioridad sobre alias.

    Returns:
        Modelo MLflow PyFunc con método predict().
    """
    pregunta = pregunta.lower()

    if pregunta not in ("p1", "p2", "p3"):
        raise ValueError(f"Pregunta no válida: {pregunta}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI no está configurada. "
            "Indica la dirección del servidor MLflow."
        )

    mlflow.set_tracking_uri(tracking_uri)

    nombre = nombre_registro(pregunta)

    if version is not None:
        modelo_uri = f"models:/{nombre}/{version}"
    else:
        modelo_uri = f"models:/{nombre}@{alias}"

    try:
        modelo = mlflow.pyfunc.load_model(modelo_uri)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar {modelo_uri} "
            f"desde MLflow ({tracking_uri})"
        ) from exc

    return modelo
