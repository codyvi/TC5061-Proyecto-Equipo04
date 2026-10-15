"""Inferencia con los modelos entrenados de P1, P2 y P3 (guardados en `models/`).

Cada pregunta tiene un predictor (`P1Predictor`, `P2Predictor`, `P3Predictor`) que carga el
modelo y valida que la tabla de entrada traiga las columnas con las que se entrenó.

    python -m mibici_mlops_module.modeling.predict --pregunta p1 --conjunto prueba
    python -m mibici_mlops_module.modeling.predict --pregunta p3
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated

from loguru import logger
import pandas as pd
import typer

from mibici_mlops_module.config import MODELS_DIR, PROCESSED_DATA_DIR
from mibici_mlops_module.features import IDENTIFICADORES
from mibici_mlops_module.modeling.estimadores import cargar_modelo

app = typer.Typer()


class BasePredictor(ABC):
    """Carga el modelo guardado y agrega columnas de predicción a una tabla de entrada."""

    pregunta: str
    archivo: str  # tabla de entrada por defecto, en data/processed/
    columna_pred: str  # columna de la predicción principal en la salida

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.guardado = cargar_modelo(self.pregunta, models_dir)
        self.modelo = self.guardado.modelo
        logger.info(
            f"[{self.pregunta}] modelo '{self.guardado.nombre}' "
            f"(datos {self.guardado.data_version}, commit {self.guardado.git_commit[:7]})"
        )

    def _entrada(self, df: pd.DataFrame) -> pd.DataFrame:
        faltan = [c for c in self.guardado.features if c not in df.columns]
        if faltan:
            raise ValueError(f"Faltan columnas para {self.pregunta}: {faltan}")
        return df[self.guardado.features]

    @abstractmethod
    def predecir(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve los identificadores de `df` con las columnas de predicción."""

    @staticmethod
    def _ids(df: pd.DataFrame) -> pd.DataFrame:
        return df[[c for c in IDENTIFICADORES if c in df.columns]].copy()


class P1Predictor(BasePredictor):
    pregunta = "p1"
    archivo = "p1_demanda.parquet"
    columna_pred = "pred_salidas_t1"

    def predecir(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ids(df)
        out[self.columna_pred] = self.modelo.predict(self._entrada(df))
        return out


class P2Predictor(BasePredictor):
    pregunta = "p2"
    archivo = "p2_riesgo.parquet"
    columna_pred = "pred_riesgo_t1"

    def predecir(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._entrada(df)
        out = self._ids(df)
        out[self.columna_pred] = self.modelo.predict(X)
        if hasattr(self.modelo, "predict_proba"):
            proba = self.modelo.predict_proba(X)
            for clase, p in zip(self.modelo.classes_, proba.T):
                out[f"prob_{clase}"] = p
        return out


class P3Predictor(BasePredictor):
    pregunta = "p3"
    archivo = "p3_perfiles.parquet"
    columna_pred = "grupo"

    def predecir(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["estacion"]].copy()
        out[self.columna_pred] = self.modelo.predict(self._entrada(df))
        return out


PREDICTORES = {"p1": P1Predictor, "p2": P2Predictor, "p3": P3Predictor}


@app.command()
def main(
    pregunta: Annotated[str, typer.Option(help="p1, p2 o p3")],
    conjunto: Annotated[str, typer.Option(help="entrenamiento, validacion o prueba (P1 y P2)")] = "prueba",
    entrada: Annotated[Path | None, typer.Option(help="Parquet de entrada (por defecto, la tabla de la pregunta)")] = None,
    salida: Annotated[Path | None, typer.Option(help="Parquet de salida")] = None,
    models_dir: Path = MODELS_DIR,
):
    pregunta = pregunta.lower()
    if pregunta not in PREDICTORES:
        raise typer.BadParameter(f"pregunta desconocida: {pregunta}")
    predictor = PREDICTORES[pregunta](models_dir)
    df = pd.read_parquet(entrada or PROCESSED_DATA_DIR / predictor.archivo)
    if pregunta != "p3" and "conjunto" in df.columns:
        df = df[df["conjunto"] == conjunto]
    pred = predictor.predecir(df)
    salida = salida or PROCESSED_DATA_DIR / f"{pregunta}_predicciones.parquet"
    pred.to_parquet(salida, index=False)
    logger.success(f"{len(pred):,} predicciones -> {salida}")


if __name__ == "__main__":
    app()
