"""Fixtures compartidas por las pruebas."""

import numpy as np
import pandas as pd
import pytest

from mibici_mlops_module.features import FEATURES_P1, OBJETIVO_P1


@pytest.fixture
def tabla_p1(tmp_path):
    """Tabla P1 sintética (mismas columnas que `p1_demanda.parquet`) guardada en un parquet.

    La demanda depende de la hora y de `salidas_lag_24h`, así que un modelo sí tiene qué aprender.
    """
    rng = np.random.default_rng(0)
    n = 600
    hora_t = pd.Timestamp("2026-02-01") + pd.to_timedelta(rng.integers(0, 24 * 60, n), unit="h")
    df = pd.DataFrame({"estacion": rng.integers(1, 20, n), "hora_t": hora_t})
    df["hora_objetivo"] = df["hora_t"] + pd.Timedelta(hours=1)
    df["conjunto"] = np.where(df.index < 400, "entrenamiento", "validacion")
    df["hora"] = df["hora_objetivo"].dt.hour.astype("int8")
    df["dia_semana"] = df["hora_objetivo"].dt.dayofweek.astype("int8")
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["es_festivo"] = np.int8(0)
    df["es_semana_santa"] = np.int8(0)
    df["latitud"] = 20.65 + rng.normal(0, 0.01, n)
    df["longitud"] = -103.35 + rng.normal(0, 0.01, n)
    for col in ("salidas_t", "salidas_lag_1h", "salidas_lag_24h", "salidas_lag_168h"):
        df[col] = rng.poisson(5, n).astype("float32")
    df["salidas_media_24h"] = df["salidas_lag_24h"].astype("float64") * 0.9
    df[OBJETIVO_P1] = (df["hora"] // 6 + df["salidas_lag_24h"]).astype("int32")
    ruta = tmp_path / "p1_sintetica.parquet"
    df.to_parquet(ruta)
    assert set(FEATURES_P1) <= set(df.columns)
    return ruta
