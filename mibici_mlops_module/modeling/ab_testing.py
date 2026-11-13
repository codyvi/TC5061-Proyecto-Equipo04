"""Versión alternativa de P1 y A/B testing contra la versión vigente, sobre junio.

Flujo (ver también reports/ab_testing/README.md):

1. Dos alternativas, cada una cambia un solo factor respecto al champion (v1), y se registran como nuevas
   versiones del mismo modelo del Model Registry:
   - `challenger` (v2): mismos hiperparámetros, datos de enero a mayo (efecto de actualizar los datos);
   - `challenger-hp` (v3): datos de enero a abril, otros hiperparámetros del YAML (efecto de la configuración).
2. `comparar`: evalúa las tres versiones sobre el mismo conjunto, junio (prueba), que ninguna vio al
   entrenar, y decide cuál queda vigente. Con `--promover` mueve el alias `champion`.

    python -m mibici_mlops_module.modeling.ab_testing
    python -m mibici_mlops_module.modeling.ab_testing --promover
"""

import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skops.io import dumps as skops_dumps
from skops.io import get_untrusted_types
import typer

from mibici_mlops_module.config import (
    EQUIPO,
    MLFLOW_EXPERIMENT_EQUIPO,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SEED,
)
from mibici_mlops_module.features import FEATURES_P1, OBJETIVO_P1
from mibici_mlops_module.modeling.registry import nombre_registro, version_con_alias
from mibici_mlops_module.modeling.train import cargar_hiperparametros
from mibici_mlops_module.utils import info_git

app = typer.Typer()

AB_DIR = REPORTS_DIR / "ab_testing"
PREGUNTA = "p1"
CHALLENGERS = {
    "challenger": {
        "run": "P1-demanda-challenger-reentrenado-ene-may",
        "conjuntos": ("entrenamiento", "validacion"),  # ene-may: nada de junio
        "hp": None,  # los mismos del champion
        "descripcion": "Alternativa de P1: mismos hiperparametros que v1, reentrenada con enero a mayo 2026.",
        "entrenamiento": "2026-01..2026-05",
    },
    "challenger-hp": {
        "run": "P1-demanda-challenger-hiperparametros",
        "conjuntos": ("entrenamiento",),  # ene-abr, igual que el champion
        "hp": "lento-lr0.05",  # combinación del YAML de hiperparámetros
        "descripcion": "Alternativa de P1: datos de enero a abril como v1, con los hiperparametros 'lento-lr0.05' del YAML.",
        "entrenamiento": "2026-01..2026-04",
    },
}
CONJUNTO_EVALUACION = "prueba"  # junio
BOOTSTRAP = 10_000


def cargar_tabla(processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Tabla de P1 con las variables del modelo, el objetivo, la fecha y el conjunto."""
    columnas = ["conjunto", "hora_objetivo", OBJETIVO_P1, *FEATURES_P1]
    return pd.read_parquet(processed_dir / "p1_demanda.parquet", columns=columnas)


def hiperparametros_champion(client: mlflow.MlflowClient, nombre: str) -> dict:
    """Hiperparámetros con los que se entrenó el champion, leídos de su corrida en MLflow."""
    champion = version_con_alias(client, nombre, "champion")
    if champion is None:
        raise RuntimeError(f"{nombre} no tiene alias champion; registra primero con modeling.registry")
    params = client.get_run(champion.run_id).data.params
    tipos = {"learning_rate": float, "max_iter": int, "max_leaf_nodes": int, "l2_regularization": float}
    return {k.removeprefix("modelo."): tipos[k.removeprefix("modelo.")](v)
            for k, v in params.items() if k.removeprefix("modelo.") in tipos}


def hiperparametros_yaml(nombre_combinacion: str) -> dict:
    """Combinación de HistGradientBoosting de P1 en `configs/hiperparametros.yaml`, por su `nombre`."""
    for hp in cargar_hiperparametros()["p1"]["hist-gradient-boosting-poisson"]:
        if hp.get("nombre") == nombre_combinacion:
            return {k: v for k, v in hp.items() if k != "nombre"}
    raise KeyError(f"no hay una combinación '{nombre_combinacion}' en el YAML de hiperparámetros")


def registrar_challenger(df: pd.DataFrame, alias: str = "challenger", forzar: bool = False) -> int:
    """Entrena una versión alternativa y la registra con su alias (`challenger` o `challenger-hp`).

    Es idempotente: si el alias ya existe, no entrena otra vez (salvo `forzar`).
    Devuelve el número de versión.
    """
    cfg = CHALLENGERS[alias]
    client = mlflow.MlflowClient()
    nombre = nombre_registro(PREGUNTA)
    actual = version_con_alias(client, nombre, alias)
    if actual is not None and not forzar:
        logger.info(f"{nombre} v{actual.version} ya tiene el alias '{alias}'")
        return int(actual.version)

    hp = hiperparametros_yaml(cfg["hp"]) if cfg["hp"] else hiperparametros_champion(client, nombre)
    train = df[df["conjunto"].isin(cfg["conjuntos"])]
    modelo = HistGradientBoostingRegressor(loss="poisson", early_stopping=True, random_state=SEED, **hp)
    git = info_git()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_EQUIPO)
    with mlflow.start_run(run_name=cfg["run"]) as run:
        mlflow.set_tags({
            "equipo": EQUIPO, "question": "P1", "stage_of_work": "challenger",
            "modelo": f"hist-gradient-boosting-poisson-{alias}", **git,
        })
        mlflow.log_params({
            "seed": SEED,
            "features": ",".join(FEATURES_P1),
            "n_entrenamiento": len(train),
            "entrenamiento_desde": str(train["hora_objetivo"].min()),
            "entrenamiento_hasta": str(train["hora_objetivo"].max()),
            "motivo": cfg["descripcion"],
            **{f"modelo.{k}": v for k, v in {"loss": "poisson", **hp}.items()},
        })
        modelo.fit(train[FEATURES_P1], train[OBJETIVO_P1])
        X = train[FEATURES_P1].head(500).astype({c: "float64" for c in FEATURES_P1
                                                  if str(train[c].dtype).startswith(("int", "uint"))})
        info = mlflow.sklearn.log_model(
            modelo, name="modelo", signature=infer_signature(X, modelo.predict(X)),
            input_example=X.head(3), serialization_format="skops",
            skops_trusted_types=get_untrusted_types(data=skops_dumps(modelo)),
        )
        version = mlflow.register_model(info.model_uri, nombre)
    n = version.version
    client.update_model_version(nombre, n, cfg["descripcion"])
    for k, v in {"source_run_id": run.info.run_id, "git_commit": git["git_commit"],
                 "entrenamiento": cfg["entrenamiento"], "modelo": "hist-gradient-boosting-poisson"}.items():
        client.set_model_version_tag(nombre, n, k, v)
    client.set_registered_model_alias(nombre, alias, n)
    logger.success(f"{nombre} v{n} -> {alias}")
    return int(n)


def predecir(alias: str, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Predicciones del modelo del Registry con ese alias; devuelve también su versión."""
    nombre = nombre_registro(PREGUNTA)
    version = version_con_alias(mlflow.MlflowClient(), nombre, alias).version
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version}")
    return np.asarray(modelo.predict(X)).ravel(), version


def comparar(df: pd.DataFrame, semilla: int = SEED) -> tuple[dict, pd.DataFrame]:
    """Evalúa el champion y las alternativas sobre junio y mide si cada diferencia es consistente.

    Además de MAE y RMSE globales, calcula la diferencia de error absoluto por día contra el champion y su
    intervalo de confianza (bootstrap sobre los 30 días): así se ve si una alternativa gana por igual en
    todo el mes o solo por unos pocos días.
    """
    junio = df[df["conjunto"] == CONJUNTO_EVALUACION].reset_index(drop=True)
    X, y = junio[FEATURES_P1], junio[OBJETIVO_P1].to_numpy()
    aliases = ["champion", *CHALLENGERS]
    preds, versiones = {}, {}
    for alias in aliases:
        preds[alias], versiones[alias] = predecir(alias, X)

    def metricas(pred):
        return {"mae": float(mean_absolute_error(y, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, pred)))}

    junio["dia"] = junio["hora_objetivo"].dt.date
    for alias in aliases:
        junio[f"err_{alias}"] = np.abs(y - preds[alias])
    por_dia = junio.groupby("dia")[[f"err_{a}" for a in aliases]].mean()

    res = {
        "evaluacion": "junio 2026 (prueba), mismas filas para todas las versiones",
        "filas": len(junio),
        "versiones": {a: {"version": versiones[a], **metricas(preds[a])} for a in aliases},
        "comparaciones": {},
    }
    rng = np.random.default_rng(semilla)
    champion = res["versiones"]["champion"]
    for alias in CHALLENGERS:
        v = res["versiones"][alias]
        dif = (por_dia["err_champion"] - por_dia[f"err_{alias}"]).to_numpy()  # > 0: gana la alternativa
        medias = rng.choice(dif, size=(BOOTSTRAP, len(dif)), replace=True).mean(axis=1)
        ic = np.percentile(medias, [2.5, 97.5])
        mejora_mae = bool(v["mae"] < champion["mae"])
        rmse_ok = bool(v["rmse"] <= champion["rmse"] * 1.01)
        consistente = bool(ic[0] > 0)
        res["comparaciones"][alias] = {
            "mejora_mae_pct": float((1 - v["mae"] / champion["mae"]) * 100),
            "cambio_rmse_pct": float((v["rmse"] / champion["rmse"] - 1) * 100),
            "dif_mae_diaria_media": float(dif.mean()),
            "dif_mae_diaria_ic95": [float(ic[0]), float(ic[1])],
            "dias_gana": int((dif > 0).sum()),
            "dias_totales": len(dif),
            "cumple": {"baja_mae": mejora_mae, "rmse_no_empeora_1pct": rmse_ok, "ic95_excluye_0": consistente},
            "gana": mejora_mae and rmse_ok and consistente,
        }
    ganadores = [a for a, c in res["comparaciones"].items() if c["gana"]]
    res["decision"] = min(ganadores, key=lambda a: res["versiones"][a]["mae"]) if ganadores else "champion"
    res["criterio"] = ("Una alternativa reemplaza al champion solo si baja el MAE, su RMSE no empeora más de "
                       "1 % y el intervalo de 95 % de la diferencia diaria de MAE excluye el 0. El MAE es la "
                       "métrica con la que el proyecto elige modelos en validación.")
    return res, por_dia


def graficar(res: dict, por_dia: pd.DataFrame, salida: Path = AB_DIR) -> Path:
    """Dos paneles: MAE por día de cada versión y barras de MAE/RMSE globales."""
    colores = {"champion": "#4a7fb5", "challenger": "#c0392b", "challenger-hp": "#2e8b57"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    dias = pd.to_datetime(por_dia.index)
    for alias, v in res["versiones"].items():
        a.plot(dias, por_dia[f"err_{alias}"], color=colores[alias], label=f"{alias} v{v['version']}")
    a.set_ylabel("MAE diario (bicicletas por estación y hora)")
    a.set_title("A/B testing P1: error por día, junio 2026")
    a.legend()
    a.tick_params(axis="x", rotation=30)
    n = len(res["versiones"])
    ancho = 0.8 / n
    x = np.arange(2)
    for i, (alias, v) in enumerate(res["versiones"].items()):
        pos = x + (i - (n - 1) / 2) * ancho
        b.bar(pos, [v["mae"], v["rmse"]], ancho, color=colores[alias], label=alias)
        for xi, val in zip(pos, [v["mae"], v["rmse"]]):
            b.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    b.set_xticks(x, ["MAE", "RMSE"])
    b.set_title("Métricas globales en junio")
    b.legend(fontsize=8)
    fig.tight_layout()
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / "ab_p1.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


@app.command()
def main(
    promover: Annotated[bool, typer.Option(help="Mueve el alias champion si gana una alternativa")] = False,
    forzar: Annotated[bool, typer.Option(help="Reentrena aunque ya exista el alias")] = False,
):
    """Registra las alternativas de P1, las compara con el champion en junio y guarda el reporte."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    df = cargar_tabla()
    for alias in CHALLENGERS:
        registrar_challenger(df, alias, forzar)
    res, por_dia = comparar(df)
    AB_DIR.mkdir(parents=True, exist_ok=True)
    (AB_DIR / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    por_dia.to_csv(AB_DIR / "mae_por_dia.csv")
    graficar(res, por_dia)
    logger.info(json.dumps(res, indent=2, ensure_ascii=False))
    if promover and res["decision"] != "champion":
        client = mlflow.MlflowClient()
        nombre = nombre_registro(PREGUNTA)
        nuevo = res["versiones"][res["decision"]]["version"]
        client.set_registered_model_alias(nombre, "anterior", res["versiones"]["champion"]["version"])
        client.set_registered_model_alias(nombre, "champion", nuevo)
        logger.success(f"{nombre}: champion -> v{nuevo}")


if __name__ == "__main__":
    app()
