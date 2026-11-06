"""API REST (FastAPI) para servir los modelos de MiBici desde el MLflow Model Registry.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Variables de entorno:
    MLFLOW_TRACKING_URI  servidor de MLflow (obligatoria; ver `model_loader`)
    MODELO_ALIAS         alias del Registry a cargar (por defecto `champion`)
    MODELO_VERSION       versión específica; tiene prioridad sobre el alias
    MLFLOW_HTTP_REQUEST_TIMEOUT / _MAX_RETRIES
                         tiempo límite (10 s) y reintentos (1) de cada llamada a MLflow;
                         sin esto, un servidor inalcanzable bloquea el arranque varios minutos

Endpoints: `GET /health`, `POST /predict/p1` (demanda) y `POST /predict/p2` (riesgo).
"""

from contextlib import asynccontextmanager
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from loguru import logger
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mibici_mlops_module.features import FEATURES_P1, FEATURES_P2
from mibici_mlops_module.model_loader import cargar_modelo_registry

PREGUNTAS_SERVIDAS = ("p1", "p2")


# ---------------------------------------------------------------------------
# Esquemas de entrada y salida. Los campos y su orden salen de FEATURES_P1/P2
# (ver `test_api.py`); los rangos son los observados en los datos de entrenamiento
# con un margen, para rechazar valores que no tienen sentido.
# ---------------------------------------------------------------------------
class _Entrada(BaseModel):
    """Rechaza campos desconocidos: un nombre mal escrito no debe ignorarse en silencio."""

    model_config = ConfigDict(extra="forbid")


class CalendarioMixin(BaseModel):
    hora: int = Field(ge=0, le=23, description="Hora del día en t")
    dia_semana: int = Field(ge=0, le=6, description="0 = lunes … 6 = domingo")
    es_fin_de_semana: Literal[0, 1]
    es_festivo: Literal[0, 1]
    es_semana_santa: Literal[0, 1]


class EntradaP1(_Entrada, CalendarioMixin):
    """Variables de P1 (demanda): estación y hora t, con sus rezagos de salidas."""

    latitud: float = Field(ge=20.0, le=21.5)
    longitud: float = Field(ge=-104.0, le=-103.0)
    salidas_t: float = Field(ge=0)
    salidas_lag_1h: float = Field(ge=0)
    salidas_lag_24h: float = Field(ge=0)
    salidas_lag_168h: float = Field(ge=0)
    salidas_media_24h: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "hora": 8, "dia_semana": 1, "es_fin_de_semana": 0, "es_festivo": 0,
                "es_semana_santa": 0, "latitud": 20.67, "longitud": -103.35,
                "salidas_t": 5, "salidas_lag_1h": 3, "salidas_lag_24h": 6,
                "salidas_lag_168h": 4, "salidas_media_24h": 2.5,
            }
        }
    }


class EntradaP2(_Entrada, CalendarioMixin):
    """Variables de P2 (riesgo): estación y hora t, con salidas, llegadas e índice de flujo."""

    latitud: float = Field(ge=20.0, le=21.5)
    longitud: float = Field(ge=-104.0, le=-103.0)
    capacidad: int = Field(gt=0, le=100, description="Anclajes de la estación")
    salidas_t: float = Field(ge=0)
    llegadas_t: float = Field(ge=0)
    salidas_lag_1h: float = Field(ge=0)
    salidas_lag_24h: float = Field(ge=0)
    salidas_lag_168h: float = Field(ge=0)
    llegadas_lag_1h: float = Field(ge=0)
    llegadas_lag_24h: float = Field(ge=0)
    llegadas_lag_168h: float = Field(ge=0)
    indice_flujo_t: float = Field(ge=-20, le=20, description="(llegadas - salidas) / capacidad en t")
    indice_flujo_acum_3h: float = Field(ge=-60, le=60)

    model_config = {
        "json_schema_extra": {
            "example": {
                "hora": 8, "dia_semana": 1, "es_fin_de_semana": 0, "es_festivo": 0,
                "es_semana_santa": 0, "latitud": 20.67, "longitud": -103.35, "capacidad": 20,
                "salidas_t": 5, "llegadas_t": 2, "salidas_lag_1h": 3, "salidas_lag_24h": 6,
                "salidas_lag_168h": 4, "llegadas_lag_1h": 2, "llegadas_lag_24h": 3,
                "llegadas_lag_168h": 3, "indice_flujo_t": -0.15, "indice_flujo_acum_3h": -0.3,
            }
        }
    }


class SalidaP1(BaseModel):
    pred_salidas_t1: float = Field(description="Salidas esperadas en la hora siguiente")
    modelo: str


class SalidaP2(BaseModel):
    pred_riesgo_t1: str = Field(description="vaciado, normal o saturación (hora siguiente)")
    modelo: str


class Salud(BaseModel):
    estado: Literal["ok", "degradado"]
    modelos_cargados: list[str]


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos del Registry al arrancar. Si uno falla, la API arranca sin él (503)."""
    # Con los valores por defecto de MLflow (120 s y 7 reintentos) un servidor caído deja el
    # arranque colgado minutos; así la API falla rápido y queda `degradado`
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    alias = os.getenv("MODELO_ALIAS", "champion")
    version = os.getenv("MODELO_VERSION") or None
    app.state.modelos = {}
    for pregunta in PREGUNTAS_SERVIDAS:
        try:
            app.state.modelos[pregunta] = cargar_modelo_registry(pregunta, alias, version)
            logger.info(f"[{pregunta}] modelo cargado del Registry")
        except Exception as exc:  # noqa: BLE001 - se reporta en /health y como 503
            causa = exc.__cause__ or exc  # el loader envuelve el error real de MLflow
            logger.error(f"[{pregunta}] no se pudo cargar: {exc}. Causa: {causa}")
    yield


app = FastAPI(
    title="MiBici MLOps - Equipo 04",
    description="Predicción de demanda (P1) y riesgo (P2) por estación de MiBici.",
    version="0.1.0",
    lifespan=lifespan,
)


def _predecir(pregunta: str, datos: BaseModel, columnas: list[str]):
    modelo = app.state.modelos.get(pregunta)
    if modelo is None:
        raise HTTPException(503, f"El modelo {pregunta} no está disponible")
    X = pd.DataFrame([datos.model_dump()])[columnas]
    # El esquema del modelo en MLflow exige tipos exactos (double/float); un int64 se rechaza
    esquema = modelo.metadata.get_input_schema()
    if esquema is not None:
        X = X.astype({c: t for c, t in zip(esquema.input_names(), esquema.pandas_types()) if c in X})
    return modelo.predict(X)[0]


@app.get("/health", response_model=Salud, tags=["estado"])
def health():
    """Indica qué modelos se cargaron del Registry."""
    cargados = sorted(app.state.modelos)
    estado = "ok" if len(cargados) == len(PREGUNTAS_SERVIDAS) else "degradado"
    return Salud(estado=estado, modelos_cargados=cargados)


@app.post("/predict/p1", response_model=SalidaP1, tags=["predicción"])
def predecir_demanda(datos: EntradaP1):
    """P1: salidas esperadas de la estación en la hora siguiente (regresión)."""
    pred = _predecir("p1", datos, FEATURES_P1)
    return SalidaP1(pred_salidas_t1=max(float(pred), 0.0), modelo="p1")


@app.post("/predict/p2", response_model=SalidaP2, tags=["predicción"])
def predecir_riesgo(datos: EntradaP2):
    """P2: riesgo de la estación en la hora siguiente (clasificación)."""
    pred = _predecir("p2", datos, FEATURES_P2)
    return SalidaP2(pred_riesgo_t1=str(pred), modelo="p2")
