"""Monitoreo de data drift con Evidently: conjunto de referencia vs. conjunto actual.

Definición de los conjuntos (ver también reports/drift/README.md):

    referencia  conjunto == "entrenamiento"  (ene-abr 2026): lo que vio el modelo al entrenar.
    actual      conjunto == "prueba"         (jun 2026): el periodo posterior más reciente
                que se tiene de los datos abiertos, que simula el tráfico de producción.

Se omite mayo (validación) a propósito: así queda un mes completo entre ambos periodos y el
cambio de temporada (secas -> lluvias, fin de ciclo escolar) es visible.

    python -m mibici_mlops_module.monitoreo
"""

import json
from pathlib import Path
import re

from loguru import logger
import pandas as pd
import typer

from mibici_mlops_module.config import PROCESSED_DATA_DIR, REPORTS_DIR, SEED
from mibici_mlops_module.features import FEATURES_P1, FEATURES_P2

app = typer.Typer()

DRIFT_DIR = REPORTS_DIR / "drift"
CONJUNTO_REFERENCIA = "entrenamiento"
CONJUNTO_ACTUAL = "prueba"
MUESTRA = 50_000  # filas por conjunto: con ~1M vs ~280k todo diferiría "significativamente"

# Variables binarias/de calendario: se tratan como categóricas aunque estén codificadas en enteros
CATEGORICAS = ["hora", "dia_semana", "es_fin_de_semana", "es_festivo", "es_semana_santa"]
PREGUNTAS = {
    "p1": ("p1_demanda.parquet", FEATURES_P1),
    "p2": ("p2_riesgo.parquet", FEATURES_P2),
}


def cargar_conjuntos(
    archivo: str, features: list[str], muestra: int = MUESTRA, processed_dir: Path = PROCESSED_DATA_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (referencia, actual): muestras aleatorias reproducibles de las variables del modelo."""
    df = pd.read_parquet(processed_dir / archivo, columns=["conjunto", *features])
    ref = df[df["conjunto"] == CONJUNTO_REFERENCIA].drop(columns="conjunto")
    act = df[df["conjunto"] == CONJUNTO_ACTUAL].drop(columns="conjunto")
    return (
        ref.sample(min(muestra, len(ref)), random_state=SEED).reset_index(drop=True),
        act.sample(min(muestra, len(act)), random_state=SEED).reset_index(drop=True),
    )


def generar_reporte(pregunta: str, salida: Path = DRIFT_DIR, muestra: int = MUESTRA) -> dict:
    """Genera el reporte HTML/JSON de drift y calidad de datos de una pregunta y devuelve un resumen."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    archivo, features = PREGUNTAS[pregunta]
    ref, act = cargar_conjuntos(archivo, features, muestra)
    categoricas = [c for c in CATEGORICAS if c in features]
    # Evidently no acepta etiquetas numpy.int8 en categóricas: se pasan a texto
    ref[categoricas] = ref[categoricas].astype(str)
    act[categoricas] = act[categoricas].astype(str)
    numericas = [c for c in features if c not in categoricas]
    definicion = DataDefinition(numerical_columns=numericas, categorical_columns=categoricas)
    ds_ref = Dataset.from_pandas(ref, data_definition=definicion)
    ds_act = Dataset.from_pandas(act, data_definition=definicion)

    # DataDriftPreset: drift por variable (+ proporción de columnas con drift).
    # DataSummaryPreset: calidad de datos (nulos, constantes, rangos, estadísticos) por variable.
    reporte = Report([DataDriftPreset(), DataSummaryPreset()])
    evaluacion = reporte.run(current_data=ds_act, reference_data=ds_ref)

    salida.mkdir(parents=True, exist_ok=True)
    evaluacion.save_html(str(salida / f"drift_{pregunta}.html"))
    evaluacion.save_json(str(salida / f"drift_{pregunta}.json"))
    logger.info(f"[{pregunta}] reporte en {salida / f'drift_{pregunta}.html'}")
    return {"pregunta": pregunta, "filas_referencia": len(ref), "filas_actual": len(act)}


def graficar_drift(pregunta: str, salida: Path = DRIFT_DIR, umbral: float = 0.1) -> Path:
    """Exporta a PNG la distancia de drift por variable (lee el JSON del reporte de Evidently)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metricas = json.loads((salida / f"drift_{pregunta}.json").read_text())["metrics"]
    datos = {}
    for m in metricas:
        c = re.match(r"ValueDrift\(column=([^,]+),", m["metric_name"])
        if c:
            datos[c.group(1)] = m["value"]
    s = pd.Series(datos).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(s) + 1.5))
    ax.barh(s.index, s.values, color=["#c0392b" if v > umbral else "#4a7fb5" for v in s.values])
    ax.axvline(umbral, color="k", ls="--", lw=1, label=f"umbral = {umbral}")
    ax.set_xlabel("Distancia de drift (Wasserstein normada / Jensen-Shannon)")
    ax.set_title(f"Data drift por variable, {pregunta.upper()}: ene-abr vs. jun 2026")
    ax.legend(loc="lower right")
    fig.tight_layout()
    ruta = salida / f"drift_{pregunta}.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


@app.command()
def main(
    pregunta: str = typer.Option("todas", help="p1, p2 o todas"),
    muestra: int = typer.Option(MUESTRA, help="Filas por conjunto"),
):
    preguntas = list(PREGUNTAS) if pregunta == "todas" else [pregunta]
    resumen = [generar_reporte(p, muestra=muestra) for p in preguntas]
    for p in preguntas:
        graficar_drift(p)
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    (DRIFT_DIR / "resumen.json").write_text(json.dumps(resumen, indent=2))


if __name__ == "__main__":
    app()
