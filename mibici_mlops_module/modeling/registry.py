"""Registra en el Model Registry de MLflow el modelo elegido de P1, P2 y P3.

El modelo a registrar es el de `models/` (el que `guardar_mejor` eligió por validación): de ahí
salen la corrida de origen, el commit y la versión de los datos, sin escribir identificadores a
mano. Cada versión queda con un alias (`champion` por defecto) y se verifica que cargue por
`models:/<nombre>@<alias>`.

    python -m mibici_mlops_module.modeling.registry                      # P1, P2 y P3 como champion
    python -m mibici_mlops_module.modeling.registry --pregunta p1
    python -m mibici_mlops_module.modeling.registry --pregunta p1 --alias challenger
"""

from typing import Annotated

from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import typer

from mibici_mlops_module.config import EQUIPO, MLFLOW_TRACKING_URI
from mibici_mlops_module.modeling.baseline import ModeloGuardado, cargar_modelo
from mibici_mlops_module.modeling.train import ENTRENADORES

app = typer.Typer()

# El nombre del registro usa "grupos" para P3; el entrenador la llama "tipologia"
TAREAS_REGISTRO = {"p1": "demanda", "p2": "riesgo", "p3": "grupos"}
DESCRIPCIONES = {
    "p1": "Predice las salidas de cada estación de MiBici en la hora siguiente (regresión), "
    "para que Operaciones decida el rebalanceo de bicicletas.",
    "p2": "Clasifica cada estación en la hora siguiente como vaciado, normal o saturación, "
    "para que Logística priorice las rutas de rebalanceo.",
    "p3": "Agrupa las estaciones de MiBici por su patrón de uso (K-Means) para la tipología "
    "que usa Planeación.",
}


def nombre_registro(pregunta: str) -> str:
    """Nombre consistente en el registro: mibici-equipo04-p1-demanda, -p2-riesgo, -p3-grupos."""
    return f"mibici-{EQUIPO}-{pregunta}-{TAREAS_REGISTRO[pregunta]}"


def validar_origen(client: MlflowClient, guardado: ModeloGuardado):
    """Devuelve la corrida de origen si terminó bien y entrenó con todos los datos.

    Por qué: `models/` se sobrescribe con cada entrenamiento, incluidas las pruebas rápidas
    (`--fraccion`). Registrar un modelo de una prueba rápida como producción sería un error
    silencioso, así que se rechaza.
    """
    run = client.get_run(guardado.run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"La corrida {guardado.run_id} no terminó correctamente")
    fraccion = float(run.data.params.get("fraccion_entrenamiento", 1.0))
    if fraccion != 1.0:
        raise ValueError(
            f"El modelo de {guardado.pregunta} se entrenó con fracción {fraccion}; "
            "entrena con todos los datos antes de registrarlo"
        )
    return run


def modelo_de_corrida(client: MlflowClient, run) -> str:
    """URI `models:/m-…` del modelo `modelo` que `log_model` guardó en la corrida.

    Por qué: desde MLflow 3 `log_model` crea un modelo registrado en el experimento y no deja
    artefactos en `runs:/<id>/modelo`; registrar desde esa ruta avisa y recurre a este mismo modelo.
    """
    encontrados = client.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string="name = 'modelo'",
        order_by=[{"field_name": "creation_time", "ascending": False}],
    )
    for modelo in encontrados:
        if modelo.source_run_id == run.info.run_id:
            return f"models:/{modelo.model_id}"
    raise ValueError(f"La corrida {run.info.run_id} no tiene un modelo 'modelo' registrado")


def version_con_alias(client: MlflowClient, nombre: str, alias: str):
    """Versión que hoy tiene el alias, o None si el modelo o el alias no existen."""
    try:
        return client.get_model_version_by_alias(nombre, alias)
    except MlflowException:
        return None


def registrar(pregunta: str, alias: str = "champion") -> str:
    """Registra el modelo guardado de la pregunta, le pone el alias y verifica que cargue.

    Es idempotente: si el alias ya apunta a una versión de la misma corrida, no crea otra versión.
    Devuelve la URI `models:/<nombre>@<alias>` ya verificada.
    """
    client = MlflowClient()
    guardado = cargar_modelo(pregunta)
    run = validar_origen(client, guardado)
    nombre = nombre_registro(pregunta)
    criterio = ENTRENADORES[pregunta].criterio

    actual = version_con_alias(client, nombre, alias)
    if actual is not None and actual.run_id == guardado.run_id:
        logger.info(f"[{pregunta}] {nombre} v{actual.version} ya tiene el alias '{alias}'")
        version = actual.version
    else:
        # Modelo que `ejecutar()` registró con `log_model`: conserva el vínculo con la corrida
        # (experimento, commit y versión de datos salen de ella)
        creada = mlflow.register_model(modelo_de_corrida(client, run), nombre)
        version = creada.version
        experimento = client.get_experiment(run.info.experiment_id).name
        client.update_registered_model(nombre, DESCRIPCIONES[pregunta])
        client.update_model_version(
            nombre,
            version,
            f"{guardado.nombre}: {criterio} = {run.data.metrics[criterio]:.4f} "
            f"(datos {guardado.data_version}, commit {guardado.git_commit[:7]}).",
        )
        etiquetas = {
            "question": pregunta.upper(),
            "experiment": experimento,
            "source_run_id": guardado.run_id,
            "git_commit": guardado.git_commit,
            "git_dirty": run.data.tags.get("git_dirty", "desconocido"),
            "data_version": guardado.data_version,
            "modelo": guardado.nombre,
            "seed": run.data.params.get("seed", "desconocida"),
            "criterio": criterio,
            "valor_criterio": f"{run.data.metrics[criterio]:.4f}",
        }
        for clave, valor in etiquetas.items():
            client.set_model_version_tag(nombre, version, clave, valor)
        client.set_registered_model_alias(nombre, alias, version)
        logger.success(f"[{pregunta}] {nombre} v{version} -> {alias}")

    uri = f"models:/{nombre}@{alias}"
    mlflow.pyfunc.load_model(uri)  # falla si el alias o los artefactos no se pueden cargar
    logger.success(f"[{pregunta}] carga verificada: {uri}")
    return uri


@app.command()
def main(
    pregunta: Annotated[str, typer.Option(help="p1, p2, p3 o todas")] = "todas",
    alias: Annotated[str, typer.Option(help="champion (producción) o challenger (siguiente)")] = "champion",
):
    """Registra el modelo guardado de una pregunta (o de todas) y le asigna el alias."""
    elegidas = list(ENTRENADORES) if pregunta in ("", "todas") else [pregunta.lower()]
    for p in elegidas:
        if p not in ENTRENADORES:
            raise typer.BadParameter(f"pregunta desconocida: {p}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for p in elegidas:
        registrar(p, alias)


if __name__ == "__main__":
    app()
