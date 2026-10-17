"""Entrena y evalúa de punta a punta, y registra ambos pasos en MLflow.

Para cada pregunta: (1) entrena todos los candidatos (runs de `train`), (2) carga el mejor
desde `models/` con el predictor de `predict` y lo evalúa sobre validación y prueba, y
(3) registra la evaluación como un run aparte (`P1-demanda-evaluacion-<modelo>`). Así lo que
se mide es el modelo guardado, el mismo que usará `predict` en producción.

    python -m mibici_mlops_module.modeling.pipeline
    python -m mibici_mlops_module.modeling.pipeline --pregunta p1 --fraccion 0.1
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import typer

from mibici_mlops_module.config import (
    EQUIPO,
    HIPERPARAMETROS_PATH,
    MLFLOW_EXPERIMENT_EQUIPO,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
)
from mibici_mlops_module.modeling.predict import PREDICTORES
from mibici_mlops_module.modeling.train import ENTRENADORES, BaseTrainer

app = typer.Typer()


def evaluar_y_registrar(trainer: BaseTrainer) -> dict[str, float]:
    """Evalúa con el predictor el modelo guardado y registra la evaluación como un run de MLflow.

    Por qué es pertinente: mide el archivo de `models/` que se usará en producción, no el objeto
    en memoria del entrenamiento. `max_dif_vs_entrenamiento` compara ambas mediciones y debe ser 0;
    si no lo es, el guardado o la carga del modelo está alterando las predicciones.
    """
    predictor = PREDICTORES[trainer.pregunta]()
    metricas, figuras = trainer.evaluar_predictor(predictor)
    entrenamiento = trainer.mejor["metricas"]
    # el modelo cargado debe reproducir las métricas del entrenamiento
    dif = max(abs(v - entrenamiento[k]) for k, v in metricas.items() if k in entrenamiento)

    guardado = predictor.guardado
    nombre_run = f"{trainer.pregunta.upper()}-{trainer.tarea}-evaluacion-{guardado.nombre}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_EQUIPO)
    with mlflow.start_run(run_name=nombre_run):
        mlflow.set_tags(
            {
                "equipo": EQUIPO,
                "pregunta": trainer.pregunta.upper(),
                "etapa": "evaluacion",
                "modelo": guardado.nombre,
                "baseline": "false",
                **trainer.git,
            }
        )
        mlflow.log_params(
            {
                "data_version": guardado.data_version,
                "run_entrenamiento": guardado.run_id,
                "modelo_guardado": guardado.nombre,
                "fraccion_entrenamiento": trainer.fraccion,
                **trainer.parametros_datos(),
            }
        )
        mlflow.log_metrics({**metricas, "max_dif_vs_entrenamiento": float(dif)})
        for archivo, fig in figuras.items():
            mlflow.log_figure(fig, f"figuras/{archivo}")
            plt.close(fig)
        mlflow.log_artifact(str(PROJ_ROOT / "data" / "manifest.json"), "datos")
    logger.success(f"[{nombre_run}] {trainer.criterio} = {metricas[trainer.criterio]:.4f} "
                   f"(dif. vs entrenamiento: {dif:.2e})")
    return metricas


@app.command()
def main(
    pregunta: Annotated[str, typer.Option(help="p1, p2, p3 o todas")] = "todas",
    fraccion: Annotated[float, typer.Option(help="Fracción del entrenamiento (pruebas rápidas)")] = 1.0,
    seed: int = 42,
    config: Annotated[Path, typer.Option(help="YAML con los hiperparámetros")] = HIPERPARAMETROS_PATH,
):
    """Entrena y evalúa una pregunta (p1, p2 o p3) o todas, registrando ambos pasos en MLflow."""
    elegidas = list(ENTRENADORES) if pregunta == "todas" else [pregunta.lower()]
    for p in elegidas:
        if p not in ENTRENADORES:
            raise typer.BadParameter(f"pregunta desconocida: {p}")
        trainer = ENTRENADORES[p](fraccion=fraccion, seed=seed, config_path=config)
        trainer.ejecutar()
        evaluar_y_registrar(trainer)


if __name__ == "__main__":
    app()
