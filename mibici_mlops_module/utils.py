"""Utilidades compartidas entre la ingesta y el entrenamiento."""

from contextlib import contextmanager
import os
import subprocess

import mlflow
import yaml

from mibici_mlops_module.config import PROJ_ROOT


def info_git() -> dict[str, str]:
    """Devuelve el commit de Git y si hay cambios sin confirmar.

    Por qué: un run solo se puede reproducir si se sabe con qué código corrió. `git_dirty=true`
    avisa que el commit por sí solo no basta para reproducirlo.
    """

    def git(*args: str) -> str:
        """Ejecuta `git <args>` en la raíz del repositorio y devuelve su salida (vacía si falla)."""
        return subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD") or "desconocido",
        "git_dirty": str(bool(git("status", "--porcelain"))).lower(),
    }


@contextmanager
def corrida_proyecto(nombre: str | None = None):
    """Abre la corrida que crea `mlflow run` para que los runs del código queden anidados en ella.

    Por qué: `mlflow run` crea una corrida y la expone en `MLFLOW_RUN_ID`; el primer
    `start_run` del proceso la reanudaría y el siguiente fallaría por haber una corrida activa.
    Con la corrida padre abierta, cada `start_run(nested=True)` cuelga de ella. Fuera de
    `mlflow run` (sin esa variable) no hace nada. Con `nombre` reemplaza el nombre aleatorio
    que MLflow le da a la corrida padre (p. ej. "puzzled-bat-941").
    """
    if os.getenv("MLFLOW_RUN_ID"):
        with mlflow.start_run():
            if nombre:
                mlflow.set_tag("mlflow.runName", nombre)
            yield
    else:
        yield


def anidada() -> bool:
    """True si ya hay una corrida activa (la del proyecto) y los runs nuevos deben anidarse."""
    return mlflow.active_run() is not None


def parsear_hp(pares: list[str]) -> dict:
    """Convierte ['n_estimators=300', 'max_depth='] en {'n_estimators': 300}.

    Los valores vacíos se omiten: así `mlflow run` puede pasar todos los parámetros del
    MLproject y solo cuentan los que se indicaron; el resto sale del YAML de hiperparámetros.
    """
    hp = {}
    for par in pares:
        clave, _, valor = par.partition("=")
        if valor.strip():
            hp[clave.strip()] = yaml.safe_load(valor)
    return hp
