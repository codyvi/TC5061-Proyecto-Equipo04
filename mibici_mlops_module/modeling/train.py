"""Entrenamiento de P1 (demanda), P2 (riesgo) y P3 (tipología) con seguimiento en MLflow.

Cada pregunta tiene un entrenador (`P1Trainer`, `P2Trainer`, `P3Trainer`) que hereda de
`BaseTrainer`. La base define el flujo común (un run de MLflow por candidato, con
parámetros, métricas, figuras, modelo y etiquetas); cada subclase define sus candidatos,
sus métricas y sus figuras. Al final guarda en `models/` el mejor candidato de la
pregunta según la validación (nunca la prueba).

    python -m mibici_mlops_module.modeling.train                  # P1, P2 y P3
    python -m mibici_mlops_module.modeling.train --pregunta p2
    python -m mibici_mlops_module.modeling.train --fraccion 0.1   # prueba rápida
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Annotated

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from skops.io import dumps as skops_dumps
from skops.io import get_untrusted_types
import typer
import yaml

from mibici_mlops_module.config import (
    EQUIPO,
    FIN_ENTRENAMIENTO,
    HIPERPARAMETROS_PATH,
    INICIO,
    INTERIM_DATA_DIR,
    MLFLOW_EXPERIMENT_EQUIPO,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    U_RIESGO,
)
from mibici_mlops_module.dataset import cargar_manifest
from mibici_mlops_module.features import (
    FEATURES_P1,
    FEATURES_P2,
    FEATURES_P3,
    OBJETIVO_P1,
    OBJETIVO_P2,
)
from mibici_mlops_module.modeling.estimadores import (
    BaselineColumna,
    BaselineCuantiles,
    ModeloGuardado,
    guardar_modelo,
)

app = typer.Typer()

CLASES_P2 = ["vaciado", "normal", "saturacion"]


def info_git() -> dict[str, str]:
    """Commit con el que se ejecuta el run y si hay cambios sin confirmar."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"git_commit": git("rev-parse", "HEAD") or "desconocido", "git_dirty": str(bool(git("status", "--porcelain"))).lower()}


def cargar_hiperparametros(ruta: Path = HIPERPARAMETROS_PATH) -> dict:
    """Lee `configs/hiperparametros.yaml`: {pregunta: {modelo: [combinaciones]}}."""
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


ABREVIATURAS = {
    "n_estimators": "n",
    "max_depth": "prof",
    "min_samples_leaf": "hoja",
    "max_samples": "muestra",
    "learning_rate": "lr",
    "max_iter": "iter",
    "max_leaf_nodes": "hojas",
    "l2_regularization": "l2",
}


def etiqueta(hp: dict) -> str:
    """Resumen legible de los hiperparámetros para el nombre del run: n100-prof14-hoja20."""
    return "-".join(f"{ABREVIATURAS.get(k, k)}{v}" for k, v in hp.items())


def nombre_y_params(hp: dict) -> tuple[str, dict]:
    """Separa el `nombre` opcional del YAML de los hiperparámetros; si falta, lo deriva."""
    params = {k: v for k, v in hp.items() if k != "nombre"}
    return hp.get("nombre") or etiqueta(params), params


@dataclass
class Candidato:
    nombre: str
    modelo: object
    params: dict = field(default_factory=dict)
    baseline: bool = False


class BaseTrainer(ABC):
    """Flujo común: un run de MLflow por candidato y guardado del mejor en `models/`."""

    pregunta: str  # "p1", "p2" o "p3"
    tarea: str  # parte del nombre del run: demanda, riesgo o tipologia
    features: list[str]
    criterio: str  # métrica de validación con la que se elige el mejor candidato
    mayor_es_mejor: bool = True

    def __init__(
        self,
        fraccion: float = 1.0,
        seed: int = 42,
        tabla_path: Path | None = None,
        config_path: Path = HIPERPARAMETROS_PATH,
    ):
        self.config_path = config_path
        self.hp = cargar_hiperparametros(config_path)[self.pregunta]
        self.fraccion = fraccion
        self.seed = seed
        self.tabla_path = tabla_path
        self.manifest = cargar_manifest()
        self.git = info_git()

    # ---- a definir por cada pregunta -------------------------------------------------
    @abstractmethod
    def cargar(self) -> None:
        """Lee la tabla de `data/processed/` y prepara las particiones."""

    @abstractmethod
    def candidatos(self) -> list[Candidato]:
        """Baselines primero y luego los modelos candidatos."""

    @abstractmethod
    def ajustar(self, modelo) -> None:
        """Ajusta el modelo con los datos de entrenamiento."""

    @abstractmethod
    def evaluar(self, c: Candidato) -> dict[str, float]:
        """Devuelve las métricas del candidato (ya ajustado)."""

    @abstractmethod
    def evaluar_predictor(self, predictor) -> tuple[dict[str, float], dict[str, plt.Figure]]:
        """Evalúa un predictor (modelo cargado desde `models/`); devuelve métricas y figuras."""

    @abstractmethod
    def graficar(self, c: Candidato) -> dict[str, plt.Figure]:
        """Figuras del candidato, por nombre de archivo."""

    @abstractmethod
    def parametros_datos(self) -> dict:
        """Rangos de fechas, filas y otros parámetros de los datos usados."""

    @abstractmethod
    def muestra_firma(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Entradas y salidas de ejemplo para la firma del modelo."""

    def finalizar(self, resultados: list[dict]) -> None:
        """Gancho para artefactos que comparan candidatos (opcional)."""

    # ---- flujo común ----------------------------------------------------------------
    def ejecutar(self) -> list[dict]:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_EQUIPO)
        self.cargar()
        resultados = []
        for c in self.candidatos():
            nombre_run = f"{self.pregunta.upper()}-{self.tarea}-{c.nombre}"
            with mlflow.start_run(run_name=nombre_run) as run:
                logger.info(f"[{nombre_run}] entrenando")
                mlflow.set_tags(
                    {
                        "equipo": EQUIPO,
                        "pregunta": self.pregunta.upper(),
                        "modelo": c.nombre,
                        "baseline": str(c.baseline).lower(),
                        **self.git,
                    }
                )
                mlflow.log_params(
                    {
                        "data_version": self.manifest["data_version"],
                        "seed": self.seed,
                        "fraccion_entrenamiento": self.fraccion,
                        "features": ",".join(self.features),
                        **self.parametros_datos(),
                        **{f"modelo.{k}": v for k, v in c.params.items()},
                    }
                )
                self.ajustar(c.modelo)
                metricas = self.evaluar(c)
                mlflow.log_metrics(metricas)
                for archivo, fig in self.graficar(c).items():
                    mlflow.log_figure(fig, f"figuras/{archivo}")
                    plt.close(fig)
                X, y = self.muestra_firma()
                # enteros como float64: la firma tolera nulos y MLflow no avisa
                X = X.astype({col: "float64" for col in X.select_dtypes("integer")})
                mlflow.sklearn.log_model(
                    c.modelo,
                    name="modelo",
                    signature=infer_signature(X, y),
                    serialization_format="skops",
                    skops_trusted_types=get_untrusted_types(data=skops_dumps(c.modelo)),
                )
                mlflow.log_artifact(str(PROJ_ROOT / "data" / "manifest.json"), "datos")
                mlflow.log_artifact(str(self.config_path), "config")
                resultados.append(
                    {"candidato": c, "metricas": metricas, "run_id": run.info.run_id}
                )
                logger.info(f"[{nombre_run}] {self.criterio} = {metricas.get(self.criterio):.4f}")
        self.guardar_mejor(resultados)
        self.finalizar(resultados)
        return resultados

    def guardar_mejor(self, resultados: list[dict]) -> Path:
        """Elige por validación entre los no baseline y guarda el modelo en `models/`."""
        elegibles = [r for r in resultados if not r["candidato"].baseline]
        signo = 1 if self.mayor_es_mejor else -1
        mejor = max(elegibles, key=lambda r: signo * r["metricas"][self.criterio])
        ruta = guardar_modelo(
            ModeloGuardado(
                pregunta=self.pregunta,
                nombre=mejor["candidato"].nombre,
                modelo=mejor["candidato"].modelo,
                features=self.features,
                data_version=self.manifest["data_version"],
                git_commit=self.git["git_commit"],
                run_id=mejor["run_id"],
            )
        )
        logger.success(f"[{self.pregunta}] mejor: {mejor['candidato'].nombre} -> {ruta}")
        self.mejor = mejor
        return ruta


class SupervisadoTrainer(BaseTrainer):
    """P1 y P2: tablas estación × hora con partición temporal entrenamiento/validación/prueba."""

    archivo: str
    objetivo: str

    def cargar(self) -> None:
        df = pd.read_parquet(self.tabla_path or PROCESSED_DATA_DIR / self.archivo)
        self.df = df
        self.train = df[df["conjunto"] == "entrenamiento"]
        if self.fraccion < 1:
            self.train = self.train.sample(frac=self.fraccion, random_state=self.seed)
        estaciones_train = set(df.loc[df["conjunto"] == "entrenamiento", "estacion"])
        self.eval = {}
        for conjunto in ("validacion", "prueba"):
            d = df[df["conjunto"] == conjunto]
            self.eval[conjunto] = d
            # estaciones sin historia en entrenamiento: arranque en frío
            self.eval[f"{conjunto}_nuevas"] = d[~d["estacion"].isin(estaciones_train)]

    def parametros_datos(self) -> dict:
        p = {"n_entrenamiento": len(self.train)}
        for conjunto in ("entrenamiento", "validacion", "prueba"):
            h = self.df.loc[self.df["conjunto"] == conjunto, "hora_objetivo"]
            p[f"{conjunto}_desde"] = str(h.min())
            p[f"{conjunto}_hasta"] = str(h.max())
        return p

    def ajustar(self, modelo) -> None:
        modelo.fit(self.train[self.features], self.train[self.objetivo])

    def muestra_firma(self):
        X = self.train[self.features].head(500)
        return X, self.predecir(self._modelo_actual, X)

    @staticmethod
    def predecir(modelo, X: pd.DataFrame) -> np.ndarray:
        return modelo.predict(X)

    @abstractmethod
    def metricas_conjunto(self, y, pred) -> dict[str, float]:
        """Métricas de un conjunto (sin prefijo)."""

    def _evaluar_con(self, predecir_df, baseline: bool) -> dict[str, float]:
        """Métricas por conjunto con `predecir_df(d) -> predicciones`."""
        self.pred = {}
        metricas = {}
        for conjunto, d in self.eval.items():
            if d.empty:
                continue
            pred = predecir_df(d)
            self.pred[conjunto] = pred
            for k, v in self.metricas_conjunto(d[self.objetivo], pred).items():
                metricas[f"{conjunto}_{k}"] = v
        return self.agregar_comparacion(metricas, baseline)

    def agregar_comparacion(self, metricas: dict, baseline: bool) -> dict:
        """Gancho para métricas derivadas (por ejemplo, la mejora contra el baseline)."""
        return metricas

    def evaluar(self, c: Candidato) -> dict[str, float]:
        self._modelo_actual = c.modelo
        return self._evaluar_con(lambda d: self.predecir(c.modelo, d[self.features]), c.baseline)

    def evaluar_predictor(self, predictor):
        metricas = self._evaluar_con(
            lambda d: predictor.predecir(d)[predictor.columna_pred].to_numpy(), baseline=False
        )
        figuras = self.graficar(Candidato(predictor.guardado.nombre, predictor.modelo))
        return metricas, figuras


class P1Trainer(SupervisadoTrainer):
    """Regresión: salidas por estación en la siguiente hora."""

    pregunta = "p1"
    tarea = "demanda"
    archivo = "p1_demanda.parquet"
    objetivo = OBJETIVO_P1
    features = FEATURES_P1
    criterio = "validacion_mae"
    mayor_es_mejor = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.referencia: dict[str, float] = {}  # mejor MAE de los baselines, por conjunto

    def candidatos(self) -> list[Candidato]:
        cands = [
            Candidato("baseline-persistencia", BaselineColumna("salidas_t"), {"columna": "salidas_t"}, True),
            Candidato("baseline-estacional-semanal", BaselineColumna("salidas_lag_168h"), {"columna": "salidas_lag_168h"}, True),
        ]
        for hp in self.hp["random-forest"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(
                f"random-forest-{nombre}",
                RandomForestRegressor(n_jobs=-1, random_state=self.seed, **hp),
                hp,
            ))
        for hp in self.hp["hist-gradient-boosting-poisson"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(
                f"hist-gradient-boosting-poisson-{nombre}",
                HistGradientBoostingRegressor(
                    loss="poisson", early_stopping=True, random_state=self.seed, **hp
                ),
                {"loss": "poisson", **hp},
            ))
        return cands

    def metricas_conjunto(self, y, pred) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        }

    def agregar_comparacion(self, m: dict, baseline: bool) -> dict:
        for k, v in list(m.items()):
            if not k.endswith("_mae"):
                continue
            if baseline:
                self.referencia[k] = min(v, self.referencia.get(k, np.inf))
            elif k in self.referencia:
                m[f"{k}_mejora_vs_baseline"] = 1 - v / self.referencia[k]
        return m

    def graficar(self, c: Candidato) -> dict[str, plt.Figure]:
        d = self.eval["prueba"].assign(pred=self.pred["prueba"])
        d = d[d["hora_objetivo"] < d["hora_objetivo"].min() + pd.Timedelta(days=14)]
        total = d.groupby("hora_objetivo")[[self.objetivo, "pred"]].sum()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(total.index, total[self.objetivo], label="real", lw=1.2)
        ax.plot(total.index, total["pred"], label="predicho", lw=1.2)
        ax.set(title=f"P1 {c.nombre}: salidas totales por hora, 2 primeras semanas de prueba",
               ylabel="salidas (todas las estaciones)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        return {"real_vs_predicho_prueba.png": fig}


class P2Trainer(SupervisadoTrainer):
    """Clasificación: vaciado, normal o saturación en la siguiente hora."""

    pregunta = "p2"
    tarea = "riesgo"
    archivo = "p2_riesgo.parquet"
    objetivo = OBJETIVO_P2
    features = FEATURES_P2
    criterio = "validacion_balanced_accuracy"

    def candidatos(self) -> list[Candidato]:
        cands = [
            Candidato("baseline-clase-mayoritaria", DummyClassifier(strategy="most_frequent"), {"strategy": "most_frequent"}, True),
        ]
        for hp in self.hp["arbol-balanceado"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(
                f"arbol-balanceado-{nombre}",
                DecisionTreeClassifier(class_weight="balanced", random_state=self.seed, **hp),
                {"class_weight": "balanced", **hp},
            ))
        for hp in self.hp["random-forest-class-weight"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(
                f"random-forest-class-weight-{nombre}",
                RandomForestClassifier(
                    class_weight="balanced_subsample", n_jobs=-1, random_state=self.seed, **hp
                ),
                {"class_weight": "balanced_subsample", **hp},
            ))
        return cands

    def parametros_datos(self) -> dict:
        return {**super().parametros_datos(), "umbral_riesgo": U_RIESGO}

    def metricas_conjunto(self, y, pred) -> dict[str, float]:
        recalls = recall_score(y, pred, labels=CLASES_P2, average=None, zero_division=0)
        return {
            **{f"recall_{c}": float(r) for c, r in zip(CLASES_P2, recalls)},
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, labels=CLASES_P2, average="macro", zero_division=0)),
        }

    def graficar(self, c: Candidato) -> dict[str, plt.Figure]:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(
            self.eval["prueba"][self.objetivo], self.pred["prueba"], labels=CLASES_P2,
            normalize="true", values_format=".2f", cmap="Blues", ax=ax,
        )
        ax.set_title(f"P2 {c.nombre}: matriz de confusión (prueba, por fila)")
        fig.tight_layout()
        return {"matriz_confusion_prueba.png": fig}


class P3Trainer(BaseTrainer):
    """Clustering de estaciones por patrón de uso (perfil de entrenamiento)."""

    pregunta = "p3"
    tarea = "tipologia"
    features = FEATURES_P3
    criterio = "silhouette"

    def cargar(self) -> None:
        self.perfiles = pd.read_parquet(self.tabla_path or PROCESSED_DATA_DIR / "p3_perfiles.parquet")
        cat = pd.read_parquet(INTERIM_DATA_DIR / "nomenclatura.parquet")
        self.coords = self.perfiles[["estacion"]].merge(
            cat[["estacion_id", "latitud", "longitud"]], left_on="estacion", right_on="estacion_id"
        )
        self.X = self.perfiles[self.features]
        self.X_escalado = StandardScaler().fit_transform(self.X)  # para comparar métricas

    def candidatos(self) -> list[Candidato]:
        cands = [Candidato("baseline-cuantiles-volumen", BaselineCuantiles(), {"n_grupos": 3}, True)]
        for hp in self.hp["kmeans"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(
                Candidato(
                    f"kmeans-{nombre}",
                    make_pipeline(
                        StandardScaler(),
                        KMeans(n_clusters=hp["k"], n_init=hp["n_init"], random_state=self.seed),
                    ),
                    {**hp, "escalado": "StandardScaler"},
                )
            )
        return cands

    def parametros_datos(self) -> dict:
        return {
            "perfil_desde": str(INICIO),
            "perfil_hasta": str(FIN_ENTRENAMIENTO),
            "n_estaciones": len(self.perfiles),
        }

    def ajustar(self, modelo) -> None:
        modelo.fit(self.X)

    def muestra_firma(self):
        return self.X.head(100), self._modelo_actual.predict(self.X.head(100))

    def _metricas_etiquetas(self, modelo) -> dict[str, float]:
        n_grupos = len(np.unique(self.etiquetas))
        metricas = {"n_grupos": float(n_grupos)}
        if n_grupos >= 2:
            metricas |= {
                "silhouette": float(silhouette_score(self.X_escalado, self.etiquetas)),
                "davies_bouldin": float(davies_bouldin_score(self.X_escalado, self.etiquetas)),
                "calinski_harabasz": float(calinski_harabasz_score(self.X_escalado, self.etiquetas)),
            }
        if hasattr(modelo, "steps"):
            metricas["inertia"] = float(modelo[-1].inertia_)
        return metricas

    def evaluar(self, c: Candidato) -> dict[str, float]:
        self._modelo_actual = c.modelo
        self.etiquetas = np.asarray(c.modelo.predict(self.X))
        return self._metricas_etiquetas(c.modelo)

    def evaluar_predictor(self, predictor):
        out = predictor.predecir(self.perfiles)
        self.etiquetas = out[predictor.columna_pred].to_numpy()
        metricas = self._metricas_etiquetas(predictor.modelo)
        return metricas, self.graficar(Candidato(predictor.guardado.nombre, predictor.modelo))

    def graficar(self, c: Candidato) -> dict[str, plt.Figure]:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        sc = ax.scatter(self.coords["longitud"], self.coords["latitud"], c=self.etiquetas,
                        cmap="tab10", s=22, alpha=0.85)
        ax.legend(*sc.legend_elements(), title="grupo", loc="best", fontsize=8)
        ax.set(title=f"P3 {c.nombre}: mapa de estaciones por grupo", xlabel="longitud", ylabel="latitud")
        ax.set_aspect(1 / np.cos(np.deg2rad(20.67)))
        fig.tight_layout()
        return {"mapa_grupos.png": fig}

    def finalizar(self, resultados: list[dict]) -> None:
        """Sube al run del mejor k la comparación de k (codo y silhouette) y los perfiles."""
        km = [r for r in resultados if not r["candidato"].baseline]
        ks = [r["candidato"].params["k"] for r in km]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(ks, [r["metricas"]["inertia"] for r in km], "o-")
        axs[0].set(title="Codo", xlabel="k", ylabel="inercia")
        axs[1].plot(ks, [r["metricas"]["silhouette"] for r in km], "o-")
        axs[1].set(title="Silhouette", xlabel="k")
        fig.tight_layout()
        modelo = self.mejor["candidato"].modelo
        perfil = self.perfiles.assign(grupo=modelo.predict(self.X)).groupby("grupo")[self.features].mean()
        with mlflow.start_run(run_id=self.mejor["run_id"]):
            mlflow.log_figure(fig, "figuras/barrido_k.png")
            mlflow.log_table(perfil.reset_index(), "perfil_por_grupo.json")
        plt.close(fig)


ENTRENADORES = {"p1": P1Trainer, "p2": P2Trainer, "p3": P3Trainer}


@app.command()
def main(
    pregunta: Annotated[str, typer.Option(help="p1, p2, p3 o todas")] = "todas",
    fraccion: Annotated[float, typer.Option(help="Fracción del entrenamiento (pruebas rápidas)")] = 1.0,
    seed: int = 42,
    config: Annotated[Path, typer.Option(help="YAML con los hiperparámetros")] = HIPERPARAMETROS_PATH,
):
    elegidas = list(ENTRENADORES) if pregunta == "todas" else [pregunta.lower()]
    for p in elegidas:
        if p not in ENTRENADORES:
            raise typer.BadParameter(f"pregunta desconocida: {p}")
        ENTRENADORES[p](fraccion=fraccion, seed=seed, config_path=config).ejecutar()
    logger.success(f"Entrenamiento completo; experimento '{MLFLOW_EXPERIMENT_EQUIPO}' en {MLFLOW_TRACKING_URI}")

    # Separar conjuntos usando la columna que tu equipo ya construyó
    df_train = df_p1[df_p1["conjunto"] == "entrenamiento"]
    df_test = df_p1[df_p1["conjunto"] == "validacion"]

    X_train = df_train[FEATURES_P1]
    y_train = df_train[OBJETIVO_P1]
    X_test = df_test[FEATURES_P1]
    y_test = df_test[OBJETIVO_P1]

    logger.info(f"Entrenando baseline P1 con {len(X_train)} registros...")
    
    # Configurar MLflow (puedes cambiar "equipo04" por el de tu equipo si lo requieren)
    mlflow.set_experiment("mibici-equipo04")
    
    with mlflow.start_run(run_name="P1_baseline_regresion"):
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        predicciones = modelo.predict(X_test)
        test_mae = mean_absolute_error(y_test, predicciones)
        
        # 1. Registrar Parámetros
        mlflow.log_param("modelo", "LinearRegression")
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("features", ", ".join(FEATURES_P1))
        
        # 2. Registrar Métricas
        mlflow.log_metric("validacion_mae", test_mae)
        
        # 3. Registrar Etiquetas
        mlflow.set_tag("pregunta", "P1")
        mlflow.set_tag("equipo", "Equipo04")
        mlflow.set_tag("commit", obtener_commit())
        
        # 4. Registrar Artefacto (el modelo)
        mlflow.sklearn.log_model(modelo, "modelo_baseline_p1")
        
    logger.success(f"Entrenamiento completado y registrado en MLflow. MAE de validación: {test_mae:.4f}")

if __name__ == "__main__":
    app()