"""Liga el drift de las entradas con el desempeño del champion: ¿el corrimiento explica el error?

El reporte de Evidently dice qué variables cambiaron; aquí se mide si ese cambio se refleja en el error.
Para P1 (con el objetivo observado de mayo y junio) se parte el MAE por tramos de `salidas_media_24h`, la
variable con drift, y se separa el aumento de error de mayo a junio en dos efectos:

- efecto de mezcla (drift): junio tiene más filas en tramos donde el modelo suele errar más;
- efecto dentro de cada tramo: el modelo erra más aun con entradas del mismo tipo (otra causa: el
  comportamiento de los usuarios cambió, no solo la entrada).

Para P2 se compara el desempeño que el entrenamiento guardó en MLflow para mayo y junio.

    python -m mibici_mlops_module.modeling.drift_desempeno
"""

import json
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici_mlops_module.config import MLFLOW_TRACKING_URI, PROCESSED_DATA_DIR, REPORTS_DIR
from mibici_mlops_module.features import FEATURES_P1, OBJETIVO_P1
from mibici_mlops_module.modeling.registry import nombre_registro

app = typer.Typer()

DRIFT_DIR = REPORTS_DIR / "drift"
VARIABLE = "salidas_media_24h"
TRAMOS = 5
METRICAS_P2 = ["balanced_accuracy", "f1_macro", "recall_vaciado", "recall_normal", "recall_saturacion"]


def descomponer(df: pd.DataFrame) -> dict:
    """Parte el MAE de P1 por tramos de la variable con drift y descompone el cambio mayo -> junio.

    `df` trae `conjunto`, `VARIABLE` y `error` (valor absoluto). Los cortes salen de los cuantiles del
    entrenamiento, que es lo que el modelo conoce.
    """
    cortes = np.unique(df.loc[df["conjunto"] == "entrenamiento", VARIABLE].quantile(np.linspace(0, 1, TRAMOS + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    df = df.assign(tramo=pd.cut(df[VARIABLE], cortes, labels=False, include_lowest=True))
    etiquetas = []
    for i in range(len(cortes) - 1):
        lo, hi = cortes[i], cortes[i + 1]
        etiquetas.append(f"≤{hi:.2f}" if i == 0 else (f">{lo:.2f}" if np.isinf(hi) else f"{lo:.2f}–{hi:.2f}"))

    def por_conjunto(nombre):
        d = df[df["conjunto"] == nombre]
        g = d.groupby("tramo")["error"]
        return (g.size() / len(d)).reindex(range(len(etiquetas)), fill_value=0), g.mean().reindex(range(len(etiquetas)))

    w_ent, _ = por_conjunto("entrenamiento")
    w_may, mae_may = por_conjunto("validacion")
    w_jun, mae_jun = por_conjunto("prueba")
    mezcla = float(((w_jun - w_may) * mae_may).sum())
    dentro = float((w_jun * (mae_jun - mae_may)).sum())
    total = float(df.loc[df["conjunto"] == "prueba", "error"].mean() - df.loc[df["conjunto"] == "validacion", "error"].mean())
    return {
        "variable": VARIABLE,
        "tramos": [{"tramo": etiquetas[i], "peso_entrenamiento": float(w_ent[i]), "peso_mayo": float(w_may[i]),
                    "peso_junio": float(w_jun[i]), "mae_mayo": float(mae_may[i]), "mae_junio": float(mae_jun[i])}
                   for i in range(len(etiquetas))],
        "mae_mayo": float(df.loc[df["conjunto"] == "validacion", "error"].mean()),
        "mae_junio": float(df.loc[df["conjunto"] == "prueba", "error"].mean()),
        "cambio_total": total,
        "efecto_mezcla_drift": mezcla,
        "efecto_dentro_de_tramo": dentro,
    }


def semanal(df: pd.DataFrame) -> list[dict]:
    """MAE por semana en mayo y junio: muestra si el aumento de error es sostenido o un solo salto."""
    d = df[df["conjunto"].isin(["validacion", "prueba"])]
    s = d.groupby(d["hora_objetivo"].dt.to_period("W-SUN").dt.start_time)["error"].mean()
    return [{"semana": str(k.date()), "mae": float(v)} for k, v in s.items()]


def escala(df: pd.DataFrame) -> dict:
    """Demanda media y MAE relativo (MAE / demanda media) por mes.

    Con un conteo, el error absoluto crece con el nivel de la demanda; el MAE relativo separa ese efecto de
    escala de un deterioro real del modelo.
    """
    out = {}
    for nombre, clave in (("validacion", "mayo"), ("prueba", "junio")):
        d = df[df["conjunto"] == nombre]
        media = float(d[OBJETIVO_P1].mean())
        out[clave] = {"demanda_media": media, "mae": float(d["error"].mean()), "mae_relativo": float(d["error"].mean() / media)}
    return out


def desempeno_p2() -> dict:
    """Métricas de P2 en mayo (validación) y junio (prueba) del champion, leídas de su corrida en MLflow."""
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(nombre_registro("p2"), "champion")
    m = client.get_run(version.run_id).data.metrics
    return {k: {"mayo": float(m[f"validacion_{k}"]), "junio": float(m[f"prueba_{k}"])} for k in METRICAS_P2}


def graficar(res: dict, salida: Path = DRIFT_DIR) -> Path:
    """Izquierda: peso de cada tramo (entrenamiento vs. junio). Derecha: MAE por tramo en mayo y junio."""
    t = pd.DataFrame(res["p1"]["tramos"])
    x = np.arange(len(t))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.bar(x - 0.2, t["peso_entrenamiento"], 0.4, color="#4a7fb5", label="entrenamiento (ene-abr)")
    a.bar(x + 0.2, t["peso_junio"], 0.4, color="#c0392b", label="junio")
    a.set_xticks(x, t["tramo"], rotation=20)
    a.set_ylabel("proporción de filas")
    a.set_xlabel(f"tramo de {VARIABLE} (cortes del entrenamiento)")
    a.set_title("Dónde cayó la entrada: el drift")
    a.legend()
    b.bar(x - 0.2, t["mae_mayo"], 0.4, color="#4a7fb5", label="mayo (validación)")
    b.bar(x + 0.2, t["mae_junio"], 0.4, color="#c0392b", label="junio (prueba)")
    b.set_xticks(x, t["tramo"], rotation=20)
    b.set_ylabel("MAE (bicicletas por estación y hora)")
    b.set_xlabel(f"tramo de {VARIABLE}")
    b.set_title("Error del champion por tramo")
    b.legend()
    fig.tight_layout()
    ruta = salida / "desempeno_p1.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


@app.command()
def main():
    """Genera reports/drift/desempeno.json y desempeno_p1.png."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    df = pd.read_parquet(PROCESSED_DATA_DIR / "p1_demanda.parquet",
                         columns=["conjunto", "hora_objetivo", OBJETIVO_P1, *FEATURES_P1])
    df = df[df["conjunto"].isin(["entrenamiento", "validacion", "prueba"])].reset_index(drop=True)
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre_registro('p1')}@champion")
    pred = np.asarray(modelo.predict(df[FEATURES_P1])).ravel()
    df["error"] = np.abs(df[OBJETIVO_P1].to_numpy() - pred)
    res = {"p1": {**descomponer(df), "semanal": semanal(df), "escala": escala(df)}, "p2": desempeno_p2()}
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    (DRIFT_DIR / "desempeno.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    graficar(res)
    logger.info(json.dumps({k: v for k, v in res["p1"].items() if k != "semanal"}, indent=2, ensure_ascii=False))
    logger.info(json.dumps(res["p2"], indent=2))


if __name__ == "__main__":
    app()
