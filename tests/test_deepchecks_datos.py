"""Validación de integridad y calidad de los datos con DeepChecks.

Cada chequeo corre sobre una muestra reproducible de `data/processed/` (se omite si el pipeline
aún no generó las tablas) y, además, sobre tablas sintéticas con el defecto a propósito: así se
comprueba que el chequeo realmente detecta el problema y no pasa por estar mal configurado.
"""

import numpy as np
import pandas as pd
import pytest

from mibici_mlops_module.config import PROCESSED_DATA_DIR, SEED
from mibici_mlops_module.features import (
    FEATURES_CALENDARIO,
    FEATURES_P1,
    FEATURES_P2,
    FEATURES_P3,
    OBJETIVO_P1,
    OBJETIVO_P2,
)

import _deepchecks_compat  # noqa: F401  (debe importarse antes que deepchecks)

pytest.importorskip("deepchecks")
from deepchecks.tabular import Dataset  # noqa: E402
from deepchecks.tabular.checks import (  # noqa: E402
    DataDuplicates,
    MixedNulls,
    PercentOfNulls,
    FeatureDrift,
    LabelDrift,
)

MUESTRA = 60_000
# Variables cuyo valor cambia por diseño entre entrenamiento y prueba (Semana Santa y los
# festivos caen solo en el periodo de entrenamiento), no son drift de datos.
VARIABLES_DE_CALENDARIO = set(FEATURES_CALENDARIO)
UMBRAL_DRIFT = 0.10  # puntaje de drift (0 = iguales, 1 = totalmente distintos)


def _dataset(df, columnas, objetivo=None):
    return Dataset(
        df[columnas + ([objetivo] if objetivo else [])],
        label=objetivo,
        cat_features=[c for c in columnas if c in FEATURES_CALENDARIO],
    )


def _leer(archivo):
    ruta = PROCESSED_DATA_DIR / archivo
    if not ruta.exists():
        pytest.skip(f"falta {ruta.name}: corre el pipeline de variables primero")
    df = pd.read_parquet(ruta)
    return df.sample(min(MUESTRA, len(df)), random_state=SEED)


@pytest.fixture(scope="module")
def p1():
    return _leer("p1_demanda.parquet")


@pytest.fixture(scope="module")
def p2():
    return _leer("p2_riesgo.parquet")


@pytest.fixture(scope="module")
def p3():
    ruta = PROCESSED_DATA_DIR / "p3_perfiles.parquet"
    if not ruta.exists():
        pytest.skip("falta p3_perfiles.parquet")
    return pd.read_parquet(ruta)


# ---- tipos de datos -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "fixture,variables,objetivo",
    [("p1", FEATURES_P1, OBJETIVO_P1), ("p2", FEATURES_P2, OBJETIVO_P2)],
)
def test_variables_declaradas_existen_y_las_numericas_son_numericas(
    request, fixture, variables, objetivo
):
    df = request.getfixturevalue(fixture)
    assert set(variables + [objetivo]) <= set(df.columns)
    no_numericas = [c for c in variables if not pd.api.types.is_numeric_dtype(df[c])]
    assert no_numericas == []


def test_p3_solo_tiene_variables_numericas(p3):
    assert set(FEATURES_P3) <= set(p3.columns)
    assert all(pd.api.types.is_numeric_dtype(p3[c]) for c in FEATURES_P3)


def test_mixed_nulls_no_hay_marcadores_de_nulo_como_texto(p2):
    """Detecta 'N/A', 'null', '' y similares, también los mezclados con NaN."""
    columnas = MixedNulls().run(_dataset(p2, FEATURES_P2, OBJETIVO_P2)).value["columns"]
    assert {c: v for c, v in columnas.items() if v} == {}


def test_mixed_nulls_detecta_marcadores_sinteticos(p2):
    sucio = p2.copy()
    sucio[OBJETIVO_P2] = sucio[OBJETIVO_P2].astype(object)
    sucio.iloc[:50, sucio.columns.get_loc(OBJETIVO_P2)] = "N/A"
    columnas = MixedNulls().run(_dataset(sucio, FEATURES_P2, OBJETIVO_P2)).value["columns"]
    assert columnas[OBJETIVO_P2]


# ---- valores faltantes --------------------------------------------------------------------
@pytest.mark.parametrize(
    "fixture,variables,objetivo",
    [("p1", FEATURES_P1, OBJETIVO_P1), ("p2", FEATURES_P2, OBJETIVO_P2)],
)
def test_sin_valores_faltantes_en_las_tablas_de_entrenamiento(
    request, fixture, variables, objetivo
):
    df = request.getfixturevalue(fixture)
    nulos = PercentOfNulls().run(_dataset(df, variables, objetivo)).value
    assert nulos["Percent of nulls in sample"].max() == 0


def test_chequeo_de_nulos_detecta_faltantes_sinteticos(p1):
    sucio = p1.copy()
    sucio.iloc[:100, sucio.columns.get_loc("salidas_t")] = np.nan
    nulos = PercentOfNulls().run(_dataset(sucio, FEATURES_P1, OBJETIVO_P1)).value
    assert nulos.loc["salidas_t", "Percent of nulls in sample"] > 0


# ---- duplicados ---------------------------------------------------------------------------
def test_sin_filas_duplicadas_con_su_identificador(p1):
    """Cada fila es una (estación, hora): no debe repetirse."""
    columnas = ["estacion", "hora_t"] + FEATURES_P1
    assert DataDuplicates().run(Dataset(p1[columnas], cat_features=[])).value == 0


def test_chequeo_de_duplicados_detecta_filas_repetidas(p1):
    columnas = ["estacion", "hora_t"] + FEATURES_P1
    con_repetidas = pd.concat([p1[columnas], p1[columnas].head(1000)], ignore_index=True)
    proporcion = DataDuplicates().run(Dataset(con_repetidas, cat_features=[])).value
    assert proporcion == pytest.approx(1000 / len(con_repetidas), rel=0.05)


# ---- drift entre entrenamiento y prueba ---------------------------------------------------
@pytest.mark.parametrize(
    "fixture,variables,objetivo",
    [("p1", FEATURES_P1, OBJETIVO_P1), ("p2", FEATURES_P2, OBJETIVO_P2)],
)
def test_sin_drift_de_variables_entre_entrenamiento_y_prueba(
    request, fixture, variables, objetivo
):
    df = request.getfixturevalue(fixture)
    tr, te = (df[df["conjunto"] == c] for c in ("entrenamiento", "prueba"))
    drift = FeatureDrift().run(_dataset(tr, variables), _dataset(te, variables)).value
    puntajes = {
        v: d["Drift score"] for v, d in drift.items() if v not in VARIABLES_DE_CALENDARIO
    }
    con_drift = {v: round(float(p), 3) for v, p in puntajes.items() if p > UMBRAL_DRIFT}
    assert con_drift == {}, f"variables con drift mayor a {UMBRAL_DRIFT}: {con_drift}"


def test_sin_drift_del_objetivo_de_p1(p1):
    tr, te = (p1[p1["conjunto"] == c] for c in ("entrenamiento", "prueba"))
    drift = LabelDrift().run(
        _dataset(tr, FEATURES_P1, OBJETIVO_P1), _dataset(te, FEATURES_P1, OBJETIVO_P1)
    ).value["Drift score"]
    assert drift <= UMBRAL_DRIFT


def test_chequeo_de_drift_detecta_un_corrimiento_sintetico(p1):
    """Control negativo: si se desplaza una variable en prueba, el chequeo debe marcarla."""
    tr = p1[p1["conjunto"] == "entrenamiento"]
    te = p1[p1["conjunto"] == "prueba"].copy()
    te["salidas_lag_24h"] = te["salidas_lag_24h"] + 10
    drift = FeatureDrift().run(
        _dataset(tr, FEATURES_P1), _dataset(te, FEATURES_P1)
    ).value
    assert drift["salidas_lag_24h"]["Drift score"] > 0.5
    assert drift["latitud"]["Drift score"] <= UMBRAL_DRIFT


# ---- partición temporal -------------------------------------------------------------------
def test_particiones_ordenadas_en_el_tiempo_y_sin_traslape(p1):
    rangos = p1.groupby("conjunto")["hora_objetivo"].agg(["min", "max"])
    assert rangos.loc["entrenamiento", "max"] < rangos.loc["validacion", "min"]
    assert rangos.loc["validacion", "max"] < rangos.loc["prueba", "min"]
