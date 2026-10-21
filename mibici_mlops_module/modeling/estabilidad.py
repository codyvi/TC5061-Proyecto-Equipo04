"""Resume la estabilidad entre semillas y el costo de cada candidato a partir de los runs de MLflow.

Agrupa los runs hijos (`stage_of_work` baseline o candidate) por pregunta y modelo, y reporta la
media y la desviación estándar de la métrica de decisión entre semillas, junto con los tiempos
de entrenamiento e inferencia. Sirve para sustentar la selección del modelo en la PR.

    python -m mibici_mlops_module.modeling.estabilidad
    python -m mibici_mlops_module.modeling.estabilidad --pregunta p1 --salida reports/estabilidad.csv
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici_mlops_module.config import MLFLOW_EXPERIMENT_EQUIPO, MLFLOW_TRACKING_URI
from mibici_mlops_module.modeling.train import ENTRENADORES

app = typer.Typer()

COSTO = ["tiempo_entrenamiento_s", "tiempo_inferencia_ms_por_1000_filas"]


def resumen(pregunta: str) -> pd.DataFrame:
    """Media, desviación y número de semillas de la métrica de decisión y del costo, por modelo."""
    criterio = ENTRENADORES[pregunta].criterio
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_EQUIPO],
        filter_string=f"tags.question = '{pregunta.upper()}' and tags.stage_of_work != 'search'"
        " and tags.stage_of_work != 'evaluation'",
    )
    columnas = [f"metrics.{m}" for m in [criterio, *COSTO]]
    columnas = [c for c in columnas if c in runs]
    if runs.empty or f"metrics.{criterio}" not in runs:
        return pd.DataFrame()
    runs = runs.rename(columns={"tags.modelo": "modelo", "params.seed": "seed"})
    # una fila por (modelo, semilla): si se repitió la misma semilla, vale la corrida más reciente
    runs = runs.sort_values("start_time").drop_duplicates(["modelo", "seed"], keep="last")
    g = runs.groupby("modelo")
    out = g[columnas].agg(["mean", "std"])
    out.columns = [f"{c.removeprefix('metrics.')}_{est}" for c, est in out.columns]
    out.insert(0, "n_semillas", g["seed"].nunique())
    return out.sort_values(f"{criterio}_mean", ascending=ENTRENADORES[pregunta].mayor_es_mejor is False)


@app.command()
def main(
    pregunta: Annotated[str, typer.Option(help="p1, p2, p3 o todas")] = "todas",
    salida: Annotated[Path | None, typer.Option(help="CSV donde guardar el resumen")] = None,
):
    """Imprime (y opcionalmente guarda) la tabla de estabilidad por pregunta."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    elegidas = list(ENTRENADORES) if pregunta in ("", "todas") else [pregunta.lower()]
    tablas = []
    for p in elegidas:
        t = resumen(p)
        if t.empty:
            logger.warning(f"[{p}] sin runs en '{MLFLOW_EXPERIMENT_EQUIPO}'")
            continue
        print(f"\n== {p.upper()} (criterio: {ENTRENADORES[p].criterio}) ==")
        print(t.round(4).to_string())
        tablas.append(t.assign(pregunta=p.upper()))
    if salida and tablas:
        salida.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tablas).to_csv(salida)
        logger.success(f"Resumen guardado en {salida}")


if __name__ == "__main__":
    app()
