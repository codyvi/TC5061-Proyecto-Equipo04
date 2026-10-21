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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Annotated

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    PrecisionRecallDisplay,
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
    SEED,
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
from mibici_mlops_module.modeling.baseline import (
    BaselineColumna,
    BaselineCuantiles,
    ModeloGuardado,
    guardar_modelo,
)
from mibici_mlops_module.utils import anidada, corrida_proyecto, info_git, parsear_hp

app = typer.Typer()

# Alias cortos de modelo por pregunta (también se aceptan los nombres completos del YAML)
ALIAS_MODELOS = {
    "p1": {
        "rf": "random-forest",
        "hgb": "hist-gradient-boosting-poisson",
        "lr": "regresion-lineal",
    },
    "p2": {"dt": "arbol-balanceado", "rf": "random-forest-class-weight"},
    "p3": {"kmeans": "kmeans"},
}

CLASES_P2 = ["vaciado", "normal", "saturacion"]


def cargar_hiperparametros(ruta: Path = HIPERPARAMETROS_PATH) -> dict:
    """Lee `configs/hiperparametros.yaml`: {pregunta: {modelo: [combinaciones]}}.

    Por qué: los hiperparámetros viven en un archivo y no en el código, así se prueban otras
    combinaciones sin tocar `train.py` y el archivo usado queda registrado en cada run.
    """
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
    """Un modelo por entrenar (o baseline) con el nombre de su run y sus hiperparámetros.

    `baseline=True` marca los modelos de referencia: se evalúan igual que los demás, pero nunca se
    guardan como el mejor. Sirven para saber cuánto aporta realmente un modelo más complejo.
    """
    nombre: str
    modelo: object
    params: dict = field(default_factory=dict)
    baseline: bool = False
    familia: str = ""  # model_family en MLflow; vacío = "baseline" o el nombre sin sufijo

    def __post_init__(self):
        self.familia = self.familia or ("baseline" if self.baseline else self.nombre)


class BaseTrainer(ABC):
    """Flujo común de entrenamiento: un run de MLflow por candidato y guardado del mejor.

    Qué hace: carga los datos, recorre los candidatos, registra en cada run parámetros, métricas,
    figuras, modelo y etiquetas, y guarda en `models/` el mejor según la validación.

    Por qué es pertinente: P1, P2 y P3 comparten el procedimiento y solo cambian los datos, las
    métricas y las figuras. Con una base común, los runs de las tres preguntas son comparables y
    quedan registrados con los mismos campos (equipo, pregunta, commit, versión de datos).
    """

    pregunta: str  # "p1", "p2" o "p3"
    tarea: str  # parte del nombre del run: demanda, riesgo o tipologia
    features: list[str]
    criterio: str  # métrica de validación con la que se elige el mejor candidato
    mayor_es_mejor: bool = True

    def __init__(
        self,
        fraccion: float = 1.0,
        seed: int = SEED,
        tabla_path: Path | None = None,
        config_path: Path = HIPERPARAMETROS_PATH,
        modelo: str | None = None,
        hp: dict | None = None,
    ):
        """Lee los hiperparámetros de la pregunta, la versión de los datos y el commit de Git.

        `fraccion` usa solo parte del entrenamiento (pruebas rápidas) y `seed` fija los modelos.
        Con `modelo` se entrena solo ese modelo (más los baselines): sin `hp` usa todas sus
        combinaciones del YAML; con `hp` parte de la primera y sobrescribe lo indicado.
        """
        self.config_path = config_path
        self.hp = cargar_hiperparametros(config_path)[self.pregunta]
        if modelo:
            self.hp = self._solo_modelo(modelo, hp or {})
        self.fraccion = fraccion
        self.seed = seed
        self.tabla_path = tabla_path
        self.manifest = cargar_manifest()
        self.git = info_git()

    def _solo_modelo(self, modelo: str, hp: dict) -> defaultdict:
        """Deja una única combinación del modelo elegido: la del YAML con `hp` encima."""
        modelo = ALIAS_MODELOS[self.pregunta].get(modelo, modelo)
        if modelo not in ALIAS_MODELOS[self.pregunta].values():
            raise typer.BadParameter(f"modelo desconocido para {self.pregunta}: {modelo}")
        combinaciones = self.hp.get(modelo) or [{}]
        if not hp:  # sin hiperparámetros indicados se conserva la búsqueda completa del YAML
            return defaultdict(list, {modelo: combinaciones})
        # con hiperparámetros distintos, el nombre del run se deriva de ellos
        base = {k: v for k, v in combinaciones[0].items() if k != "nombre"}
        return defaultdict(list, {modelo: [{**base, **hp}]})

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
        """Entrena todos los candidatos como una búsqueda: un run padre y un run hijo por candidato.

        Cada hijo lleva las etiquetas `equipo`, `question`, `model_family`, `stage_of_work`
        (baseline o candidate) y el commit; los parámetros que lo hacen reproducible; las métricas;
        el conjunto de entrenamiento (`log_input`); y los artefactos organizados en `plots/`,
        `data/` y `config/`, más el modelo con su firma y un ejemplo de entrada.
        Devuelve, por candidato, sus métricas y el `run_id`.
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_EQUIPO)
        self.cargar()
        resultados = []
        etiquetas_base = {
            "equipo": EQUIPO,
            "question": self.pregunta.upper(),
            "pregunta": self.pregunta.upper(),
            "data_version": self.manifest["data_version"],
            **self.git,
        }
        entrenamiento = self.train if hasattr(self, "train") else self.X
        # enteros como float64, igual que en la firma: así MLflow no avisa de posibles nulos
        entrenamiento = entrenamiento.astype(
            {col: "float64" for col in entrenamiento.select_dtypes("integer")}
        )
        dataset = mlflow.data.from_pandas(
            entrenamiento,
            targets=getattr(self, "objetivo", None),
            name=f"{self.pregunta}_entrenamiento",
        )

        # nested=anidada(): bajo `mlflow run` ya hay una corrida activa (la del proyecto)
        with mlflow.start_run(run_name=f"{self.pregunta.upper()}-busqueda-{self.tarea}", nested=anidada()):
            mlflow.set_tags({**etiquetas_base, "stage_of_work": "search"})
            for c in self.candidatos():
                nombre_run = f"{self.pregunta.upper()}-{self.tarea}-{c.nombre}"
                with mlflow.start_run(run_name=nombre_run, nested=True) as run:
                    logger.info(f"[{nombre_run}] entrenando")
                    mlflow.set_tags(
                        {
                            **etiquetas_base,
                            "model_family": c.familia,
                            "modelo": c.nombre,
                            "baseline": str(c.baseline).lower(),
                            "stage_of_work": "baseline" if c.baseline else "candidate",
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
                    mlflow.log_input(dataset, context="training")
                    t0 = time.perf_counter()
                    self.ajustar(c.modelo)
                    segundos_entrenamiento = time.perf_counter() - t0
                    metricas = self.evaluar(c)
                    X, y = self.muestra_firma()
                    metricas["tiempo_entrenamiento_s"] = segundos_entrenamiento
                    metricas["tiempo_inferencia_ms_por_1000_filas"] = self.ms_por_1000_filas(X)
                    mlflow.log_metrics(metricas)
                    for archivo, fig in self.graficar(c).items():
                        mlflow.log_figure(fig, f"plots/{archivo}")
                        plt.close(fig)
                    mlflow.log_artifact(str(PROJ_ROOT / "data" / "manifest.json"), "data")
                    mlflow.log_artifact(str(self.config_path), "config")
                    # enteros como float64: la firma tolera nulos y MLflow no avisa
                    X = X.astype({col: "float64" for col in X.select_dtypes("integer")})
                    mlflow.sklearn.log_model(
                        c.modelo,
                        name="modelo",
                        signature=infer_signature(X, y),
                        input_example=X.head(3),
                        serialization_format="skops",
                        skops_trusted_types=get_untrusted_types(data=skops_dumps(c.modelo)),
                    )
                    resultados.append({"candidato": c, "metricas": metricas, "run_id": run.info.run_id})
                    logger.info(f"[{nombre_run}] {self.criterio} = {metricas.get(self.criterio):.4f}")
        self.guardar_mejor(resultados)
        self.finalizar(resultados)
        return resultados

    def ms_por_1000_filas(self, X: pd.DataFrame, repeticiones: int = 5) -> float:
        """Tiempo de inferencia del modelo actual, en milisegundos por cada 1000 filas.

        Por qué: junto con `tiempo_entrenamiento_s` documenta el costo del candidato. Se toma la
        mediana de varias repeticiones sobre la misma muestra de la firma para reducir el ruido.
        """
        tiempos = []
        for _ in range(repeticiones):
            t0 = time.perf_counter()
            self._modelo_actual.predict(X)
            tiempos.append(time.perf_counter() - t0)
        return float(np.median(tiempos) / len(X) * 1000 * 1000)

    def guardar_mejor(self, resultados: list[dict]) -> Path:
        """Elige por validación entre los no baseline y guarda el modelo en `models/`.

        Por qué: elegir con la validación (y no con la prueba) evita que el conjunto de prueba influya
        en la decisión; así las métricas de prueba siguen siendo una estimación honesta.
        """
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
    """Base de P1 y P2: tablas estación × hora con partición temporal.

    Por qué es pertinente: ambas preguntas predicen lo que ocurre en la hora siguiente (t+1), así que
    entrenan con enero a abril, validan con mayo y prueban con junio. Una partición aleatoria
    mezclaría futuro y pasado y daría métricas demasiado optimistas para un uso operativo.
    """

    archivo: str
    objetivo: str

    def cargar(self) -> None:
        """Lee la tabla de la pregunta y separa entrenamiento, validación y prueba.

        Además de los conjuntos completos, separa las estaciones sin historia en entrenamiento
        (`*_nuevas`): miden el arranque en frío, es decir, qué tan bien funciona el modelo en
        estaciones que abren después de entrenar, un caso real de la red.
        """
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
        """Rango de fechas (por hora objetivo) y tamaño de cada partición.

        Se registran como parámetros del run para que cualquier resultado indique con qué periodo
        se entrenó, validó y probó.
        """
        p = {"n_entrenamiento": len(self.train)}
        for conjunto in ("entrenamiento", "validacion", "prueba"):
            h = self.df.loc[self.df["conjunto"] == conjunto, "hora_objetivo"]
            p[f"{conjunto}_desde"] = str(h.min())
            p[f"{conjunto}_hasta"] = str(h.max())
        return p

    def ajustar(self, modelo) -> None:
        """Entrena el modelo con las variables de la pregunta (`features`), nunca con el resto."""
        modelo.fit(self.train[self.features], self.train[self.objetivo])

    def muestra_firma(self):
        """Entradas y salidas de ejemplo con las que MLflow documenta el formato del modelo."""
        X = self.train[self.features].head(500)
        return X, self.predecir(self._modelo_actual, X)

    @staticmethod
    def predecir(modelo, X: pd.DataFrame) -> np.ndarray:
        """Predicción del modelo para una tabla de variables."""
        return modelo.predict(X)

    @abstractmethod
    def metricas_conjunto(self, y, pred) -> dict[str, float]:
        """Métricas de un conjunto (sin prefijo)."""

    def _evaluar_con(self, predecir_df, baseline: bool) -> dict[str, float]:
        """Calcula las métricas de cada conjunto con `predecir_df(d) -> predicciones`.

        Compartir esta rutina entre `evaluar` (durante el entrenamiento) y `evaluar_predictor`
        (con el modelo guardado) asegura que ambas mediciones sean idénticas y comparables.
        """
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
        """Gancho para métricas derivadas, como la mejora contra el baseline (por defecto no agrega nada)."""
        return metricas

    def evaluar(self, c: Candidato) -> dict[str, float]:
        """Evalúa un candidato ya entrenado en validación, prueba y estaciones nuevas."""
        self._modelo_actual = c.modelo
        return self._evaluar_con(lambda d: self.predecir(c.modelo, d[self.features]), c.baseline)

    def evaluar_predictor(self, predictor):
        """Evalúa el modelo cargado desde `models/` con el predictor, en los mismos conjuntos.

        Por qué: mide el archivo que se usará en producción, no el objeto en memoria. Si ambas
        mediciones difieren, el guardado o la carga del modelo está fallando.
        """
        metricas = self._evaluar_con(
            lambda d: predictor.predecir(d)[predictor.columna_pred].to_numpy(), baseline=False
        )
        figuras = self.graficar(Candidato(predictor.guardado.nombre, predictor.modelo))
        return metricas, figuras


class P1Trainer(SupervisadoTrainer):
    """P1 (demanda): predice las salidas de cada estación en la siguiente hora (regresión).

    Por qué es pertinente: Operaciones decide el rebalanceo de bicicletas según cuántas saldrán de
    cada estación en la próxima hora. La variable es un conteo no negativo, y se compara contra
    baselines simples (persistencia y estacional semanal) porque un modelo solo vale la pena si los
    supera. El mejor se elige por el MAE de validación.
    """

    pregunta = "p1"
    tarea = "demanda"
    archivo = "p1_demanda.parquet"
    objetivo = OBJETIVO_P1
    features = FEATURES_P1
    criterio = "validacion_mae"
    mayor_es_mejor = False

    def __init__(self, *args, **kwargs):
        """Prepara el diccionario donde se guarda el mejor MAE de los baselines, para medir la mejora."""
        super().__init__(*args, **kwargs)
        self.referencia: dict[str, float] = {}  # mejor MAE de los baselines, por conjunto

    def candidatos(self) -> list[Candidato]:
        """Dos baselines y tres familias de modelos con las combinaciones del YAML.

        - Persistencia (la hora siguiente = la hora actual) y estacional semanal (la misma hora de
          la semana pasada): referencias que cualquier modelo debe superar.
        - Regresión lineal: modelo simple e interpretable; muestra cuánto se gana al pasar de una
          relación lineal a una no lineal (se empieza por lo simple, Huyen cap. 6). Puede predecir
          valores negativos, que no tienen sentido para un conteo.
        - Random Forest: captura relaciones no lineales entre hora, día y estación sin
          transformar las variables.
        - HistGradientBoosting con pérdida Poisson: las salidas son un conteo no negativo, y la
          pérdida Poisson modela justamente ese tipo de variable, sin predecir valores negativos.
        """
        cands = [
            Candidato("baseline-persistencia", BaselineColumna("salidas_t"), {"columna": "salidas_t"}, True),
            Candidato("baseline-estacional-semanal", BaselineColumna("salidas_lag_168h"), {"columna": "salidas_lag_168h"}, True),
        ]
        for hp in self.hp.get("regresion-lineal", []):
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(f"regresion-lineal-{nombre}", LinearRegression(**hp), hp, familia="regresion-lineal"))
        for hp in self.hp["random-forest"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(
                f"random-forest-{nombre}",
                RandomForestRegressor(n_jobs=-1, random_state=self.seed, **hp),
                hp,
                familia="random-forest",
            ))
        for hp in self.hp["hist-gradient-boosting-poisson"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(
                f"hist-gradient-boosting-poisson-{nombre}",
                HistGradientBoostingRegressor(
                    loss="poisson", early_stopping=True, random_state=self.seed, **hp
                ),
                {"loss": "poisson", **hp},
                familia="hist-gradient-boosting-poisson",
            ))
        return cands

    def metricas_conjunto(self, y, pred) -> dict[str, float]:
        """MAE y RMSE, ambos en bicicletas por estación y hora.

        Por qué: el MAE es el error típico, fácil de explicar a Operaciones. El RMSE penaliza más
        los errores grandes, que son los picos de demanda donde fallar produce estaciones vacías.
        """
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        }

    def agregar_comparacion(self, m: dict, baseline: bool) -> dict:
        """Calcula la mejora relativa del MAE contra el mejor baseline: 1 - MAE / MAE_baseline.

        Por qué: un MAE aislado no dice si el modelo sirve; la mejora indica cuánto aporta sobre
        la alternativa trivial. Es positiva si el modelo supera al baseline.
        """
        for k, v in list(m.items()):
            if not k.endswith("_mae"):
                continue
            if baseline:
                self.referencia[k] = min(v, self.referencia.get(k, np.inf))
            elif k in self.referencia:
                m[f"{k}_mejora_vs_baseline"] = 1 - v / self.referencia[k]
        return m

    def graficar(self, c: Candidato) -> dict[str, plt.Figure]:
        """Compara las salidas reales y predichas, sumadas por hora, en las dos primeras semanas de prueba.

        Por qué: muestra si el modelo sigue el ritmo diario y semanal de la demanda y dónde se
        equivoca (por ejemplo, en los picos).
        """
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
    """P2 (riesgo): clasifica cada estación en la siguiente hora como vaciado, normal o saturación.

    Por qué es pertinente: Logística prioriza rutas según qué estaciones se vaciarán o llenarán. Las
    clases están muy desbalanceadas (alrededor de 3.5 % vaciado, 93 % normal y 3.5 % saturación),
    por lo que la exactitud sería engañosa: se usan recall por clase y balanced accuracy, y los
    modelos ponderan las clases. El mejor se elige por la balanced accuracy de validación.
    """

    pregunta = "p2"
    tarea = "riesgo"
    archivo = "p2_riesgo.parquet"
    objetivo = OBJETIVO_P2
    features = FEATURES_P2
    criterio = "validacion_balanced_accuracy"

    def candidatos(self) -> list[Candidato]:
        """Un baseline y dos modelos que compensan el desbalance con `class_weight`.

        - Clase mayoritaria: siempre predice "normal"; da una balanced accuracy de 1/3 y es el piso.
        - Árbol balanceado: modelo simple e interpretable; `class_weight='balanced'` evita que
          ignore las clases raras, que son justo las que importan.
        - Random Forest con `class_weight`: combina muchos árboles para reducir el sobreajuste.
        """
        cands = [
            Candidato("baseline-clase-mayoritaria", DummyClassifier(strategy="most_frequent"), {"strategy": "most_frequent"}, True),
        ]
        for hp in self.hp["arbol-balanceado"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(
                f"arbol-balanceado-{nombre}",
                DecisionTreeClassifier(class_weight="balanced", random_state=self.seed, **hp),
                {"class_weight": "balanced", **hp},
                familia="arbol-balanceado",
            ))
        for hp in self.hp["random-forest-class-weight"]:
            nombre, hp = nombre_y_params(hp)
            cands.append(Candidato(
                f"random-forest-class-weight-{nombre}",
                RandomForestClassifier(
                    class_weight="balanced_subsample", n_jobs=-1, random_state=self.seed, **hp
                ),
                {"class_weight": "balanced_subsample", **hp},
                familia="random-forest-class-weight",
            ))
        return cands

    def parametros_datos(self) -> dict:
        """Agrega el umbral de riesgo, que es parte de la definición de la etiqueta."""
        return {**super().parametros_datos(), "umbral_riesgo": U_RIESGO}

    def metricas_conjunto(self, y, pred) -> dict[str, float]:
        """Recall por clase, balanced accuracy y F1 macro.

        Por qué: el recall de vaciado y de saturación dice qué proporción de los eventos reales se
        detecta, que es lo que le importa a Logística. La balanced accuracy y el F1 macro
        promedian las clases por igual, sin dejar que la clase "normal" domine.
        """
        recalls = recall_score(y, pred, labels=CLASES_P2, average=None, zero_division=0)
        return {
            **{f"recall_{c}": float(r) for c, r in zip(CLASES_P2, recalls)},
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, labels=CLASES_P2, average="macro", zero_division=0)),
        }

    def graficar(self, c: Candidato) -> dict[str, plt.Figure]:
        """Matriz de confusión de prueba, normalizada por fila.

        Por qué: cada fila muestra qué porcentaje de los vaciados (o saturaciones) reales se detecta
        y con qué se confunde, información que una sola métrica no da.
        """
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(
            self.eval["prueba"][self.objetivo], self.pred["prueba"], labels=CLASES_P2,
            normalize="true", values_format=".2f", cmap="Blues", ax=ax,
        )
        ax.set_title(f"P2 {c.nombre}: matriz de confusión (prueba, por fila)")
        fig.tight_layout()
        return {"matriz_confusion_prueba.png": fig, "curva_precision_recall_prueba.png": self._curva_pr(c)}

    def _curva_pr(self, c: Candidato) -> plt.Figure:
        """Curva precisión-recall de vaciado y saturación (cada una contra el resto), en prueba.

        Por qué: con clases tan raras, la curva muestra cuánta precisión se pierde al detectar más
        eventos reales, que es el equilibrio que enfrenta Logística al decidir cuántas rutas armar.
        """
        d = self.eval["prueba"]
        proba = c.modelo.predict_proba(d[self.features])
        fig, ax = plt.subplots(figsize=(6, 5))
        for clase in ("vaciado", "saturacion"):
            PrecisionRecallDisplay.from_predictions(
                d[self.objetivo] == clase, proba[:, list(c.modelo.classes_).index(clase)],
                name=clase, ax=ax,
            )
        ax.set_title(f"P2 {c.nombre}: curva precisión-recall (prueba)")
        fig.tight_layout()
        return fig


class P3Trainer(BaseTrainer):
    """P3 (tipología): agrupa las estaciones por su patrón de uso (clustering).

    Por qué es pertinente: Planeación necesita una tipología (residencial, de destino, de ocio o de
    alto tráfico) y no existe una etiqueta que predecir, por eso es aprendizaje no supervisado. El
    perfil usa solo el periodo de entrenamiento, así que el agrupamiento no depende de lo que
    ocurre en validación ni prueba. El mejor k se elige por el coeficiente de silhouette.
    """

    pregunta = "p3"
    tarea = "tipologia"
    features = FEATURES_P3
    criterio = "entrenamiento_silhouette"

    def cargar(self) -> None:
        """Lee los perfiles de estación, sus coordenadas y las variables escaladas.

        Las coordenadas vienen de la nomenclatura y solo se usan para el mapa. Las variables
        escaladas se usan para calcular las métricas de agrupamiento con la misma escala en todos
        los candidatos.
        """
        self.perfiles = pd.read_parquet(self.tabla_path or PROCESSED_DATA_DIR / "p3_perfiles.parquet")
        cat = pd.read_parquet(INTERIM_DATA_DIR / "nomenclatura.parquet")
        self.coords = self.perfiles[["estacion"]].merge(
            cat[["estacion_id", "latitud", "longitud"]], left_on="estacion", right_on="estacion_id"
        )
        self.X = self.perfiles[self.features]
        self.X_escalado = StandardScaler().fit_transform(self.X)  # para comparar métricas

    def candidatos(self) -> list[Candidato]:
        """Un baseline por cuantiles de volumen y un K-Means por cada k del YAML.

        - Cuantiles de volumen: agrupa solo por cantidad de tráfico; K-Means debe aportar estructura
          más allá del volumen para justificarse.
        - K-Means con `StandardScaler`: el escalado es necesario porque las variables tienen escalas
          distintas (movimientos totales frente a proporciones) y K-Means usa distancia euclídea.
        """
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
                    familia="kmeans",
                )
            )
        return cands

    def parametros_datos(self) -> dict:
        """Periodo del que sale el perfil (solo entrenamiento) y número de estaciones agrupadas."""
        return {
            "perfil_desde": str(INICIO),
            "perfil_hasta": str(FIN_ENTRENAMIENTO),
            "n_estaciones": len(self.perfiles),
        }

    def ajustar(self, modelo) -> None:
        """Ajusta el agrupamiento con los perfiles de todas las estaciones (no hay conjuntos de validación ni prueba)."""
        modelo.fit(self.X)

    def muestra_firma(self):
        """Perfiles de ejemplo y el grupo asignado, para documentar el formato del modelo en MLflow."""
        return self.X.head(100), self._modelo_actual.predict(self.X.head(100))

    def _metricas_etiquetas(self, modelo) -> dict[str, float]:
        """Silhouette, Davies-Bouldin, Calinski-Harabasz e inercia de la agrupación actual.

        Por qué: sin etiquetas reales, la calidad se mide por qué tan compactos y separados quedan los
        grupos. Silhouette (mayor es mejor) elige el k; Davies-Bouldin (menor es mejor) y
        Calinski-Harabasz (mayor es mejor) lo confirman; la inercia alimenta el codo.
        """
        n_grupos = len(np.unique(self.etiquetas))
        metricas = {"entrenamiento_n_grupos": float(n_grupos)}
        if n_grupos >= 2:
            metricas |= {
                "entrenamiento_silhouette": float(silhouette_score(self.X_escalado, self.etiquetas)),
                "entrenamiento_davies_bouldin": float(davies_bouldin_score(self.X_escalado, self.etiquetas)),
                "entrenamiento_calinski_harabasz": float(calinski_harabasz_score(self.X_escalado, self.etiquetas)),
            }
        if hasattr(modelo, "steps"):
            metricas["entrenamiento_inertia"] = float(modelo[-1].inertia_)
        return metricas

    def evaluar(self, c: Candidato) -> dict[str, float]:
        """Asigna grupo a cada estación con el candidato ya ajustado y calcula las métricas de agrupamiento."""
        self._modelo_actual = c.modelo
        self.etiquetas = np.asarray(c.modelo.predict(self.X))
        return self._metricas_etiquetas(c.modelo)

    def evaluar_predictor(self, predictor):
        """Asigna los grupos con el modelo cargado desde `models/` y calcula las mismas métricas."""
        out = predictor.predecir(self.perfiles)
        self.etiquetas = out[predictor.columna_pred].to_numpy()
        metricas = self._metricas_etiquetas(predictor.modelo)
        return metricas, self.graficar(Candidato(predictor.guardado.nombre, predictor.modelo))

    def graficar(self, c: Candidato) -> dict[str, plt.Figure]:
        """Mapa de las estaciones coloreadas por grupo.

        Por qué: es lo que permite a Planeación interpretar los grupos, por ejemplo si los de ocio se
        concentran en parques o los de destino en zonas de oficinas.
        """
        fig, ax = plt.subplots(figsize=(6.5, 6))
        sc = ax.scatter(self.coords["longitud"], self.coords["latitud"], c=self.etiquetas,
                        cmap="tab10", s=22, alpha=0.85)
        ax.legend(*sc.legend_elements(), title="grupo", loc="best", fontsize=8)
        ax.set(title=f"P3 {c.nombre}: mapa de estaciones por grupo", xlabel="longitud", ylabel="latitud")
        ax.set_aspect(1 / np.cos(np.deg2rad(20.67)))
        fig.tight_layout()
        return {"mapa_grupos.png": fig, "perfiles_grupos.png": self._perfiles(c)}

    def _perfiles(self, c: Candidato) -> plt.Figure:
        """Perfil de cada grupo: media de cada variable (estandarizada) por grupo.

        Por qué: es la figura que permite nombrar los grupos (residencial, de destino, de ocio o de
        alto tráfico) al ver en qué variables se separan del promedio de la red.
        """
        z = pd.DataFrame(self.X_escalado, columns=self.features).groupby(self.etiquetas).mean()
        fig, ax = plt.subplots(figsize=(8, 3.5 + 0.4 * len(z)))
        im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(self.features)), self.features, rotation=45, ha="right")
        ax.set_yticks(range(len(z)), [f"grupo {g}" for g in z.index])
        fig.colorbar(im, ax=ax, label="media estandarizada")
        ax.set_title(f"P3 {c.nombre}: perfil de cada grupo")
        fig.tight_layout()
        return fig

    def finalizar(self, resultados: list[dict]) -> None:
        """Sube al run del mejor k la comparación de k (codo y silhouette) y los perfiles."""
        km = [r for r in resultados if not r["candidato"].baseline]
        ks = [r["candidato"].params["k"] for r in km]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(ks, [r["metricas"]["entrenamiento_inertia"] for r in km], "o-")
        axs[0].set(title="Codo", xlabel="k", ylabel="inercia")
        axs[1].plot(ks, [r["metricas"]["entrenamiento_silhouette"] for r in km], "o-")
        axs[1].set(title="Silhouette", xlabel="k")
        fig.tight_layout()
        modelo = self.mejor["candidato"].modelo
        perfil = self.perfiles.assign(grupo=modelo.predict(self.X)).groupby("grupo")[self.features].mean()
        with mlflow.start_run(run_id=self.mejor["run_id"], nested=anidada()):
            mlflow.log_figure(fig, "plots/barrido_k.png")
            mlflow.log_table(perfil.reset_index(), "data/perfil_por_grupo.json")
        plt.close(fig)


ENTRENADORES = {"p1": P1Trainer, "p2": P2Trainer, "p3": P3Trainer}


@app.command()
def main(
    pregunta: Annotated[str, typer.Option(help="p1, p2, p3 o todas")] = "todas",
    fraccion: Annotated[float, typer.Option(help="Fracción del entrenamiento (pruebas rápidas)")] = 1.0,
    seed: int = SEED,
    config: Annotated[Path, typer.Option(help="YAML con los hiperparámetros")] = HIPERPARAMETROS_PATH,
    modelo: Annotated[str | None, typer.Option(help="Solo este modelo (p. ej. rf, hgb, dt, kmeans)")] = None,
    hp: Annotated[list[str] | None, typer.Option(help="Hiperparámetro clave=valor, repetible")] = None,
):
    """Entrena una pregunta (p1, p2 o p3) o todas, y registra los runs en MLflow."""
    elegidas = list(ENTRENADORES) if pregunta in ("", "todas") else [pregunta.lower()]
    for p in elegidas:
        if p not in ENTRENADORES:
            raise typer.BadParameter(f"pregunta desconocida: {p}")
    with corrida_proyecto():
        for p in elegidas:
            ENTRENADORES[p](
                fraccion=fraccion, seed=seed, config_path=config, modelo=modelo, hp=parsear_hp(hp or [])
            ).ejecutar()
    logger.success(f"Entrenamiento completo; experimento '{MLFLOW_EXPERIMENT_EQUIPO}' en {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    app()
