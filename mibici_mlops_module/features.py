"""Ingeniería de variables de MiBici para P1 (demanda), P2 (riesgo) y P3 (tipología).

Convención temporal: cada fila de P1 y P2 es una estación en la hora **t** y se predice
lo que ocurre en **t+1**. Toda variable de entrada usa solo información disponible al
final de la hora t (rezagos, ventanas hacia atrás o calendario de la hora objetivo, que
se conoce de antemano). Los objetivos son lo único que mira hacia t+1.

    python -m mibici_mlops_module.features
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_mlops_module.config import (
    FESTIVOS,
    FIN,
    FIN_ENTRENAMIENTO,
    FIN_VALIDACION,
    INICIO,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    SEMANA_SANTA,
    U_RIESGO,
)
from mibici_mlops_module.utils import corrida_proyecto

app = typer.Typer()

# ---------------------------------------------------------------------------
# Columnas de las tablas. El entrenamiento debe tomar las variables de aquí
# (FEATURES_*), nunca "todas las columnas menos el objetivo".
# ---------------------------------------------------------------------------
IDENTIFICADORES = ["estacion", "hora_t", "hora_objetivo", "conjunto"]
FEATURES_CALENDARIO = ["hora", "dia_semana", "es_fin_de_semana", "es_festivo", "es_semana_santa"]
FEATURES_P1 = [
    *FEATURES_CALENDARIO,
    "latitud",
    "longitud",
    "salidas_t",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "salidas_media_24h",
]
FEATURES_P2 = [
    *FEATURES_CALENDARIO,
    "latitud",
    "longitud",
    "capacidad",
    "salidas_t",
    "llegadas_t",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "llegadas_lag_1h",
    "llegadas_lag_24h",
    "llegadas_lag_168h",
    "indice_flujo_t",
    "indice_flujo_acum_3h",
]
FEATURES_P3 = [
    "horas_operando",
    "movimientos_totales",
    "prom_salidas_hora",
    "prom_llegadas_hora",
    "balance_neto",
    "prop_pico_am",
    "prop_pico_pm",
    "prop_fin_de_semana",
]
OBJETIVO_P1 = "y_salidas_t1"
OBJETIVO_P2 = "y_riesgo_t1"
# Auxiliar de P2: el índice de flujo en t+1 del que se deriva el objetivo. Se conserva para
# análisis, pero usarlo como variable de entrada sería una fuga total.
AUXILIAR_P2 = "indice_flujo_t1"


# ---------------------------------------------------------------------------
# Malla estación × hora
# ---------------------------------------------------------------------------
def malla_estacion_hora(v: pd.DataFrame, estaciones) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cuenta salidas y llegadas por estación y hora, y marca cuándo opera cada estación.

    Qué hace: construye una fila por estación y hora del semestre. Las salidas se cuentan
    por hora de inicio del viaje en el origen y las llegadas por hora de fin en el destino.
    Las horas sin viajes valen 0. Fuera de la ventana de operación de cada estación (del
    día de su primer movimiento al de su último) los valores quedan como NaN.

    Por qué sirve: es la base de P1, P2 y P3. La red creció durante el semestre y rellenar
    con 0 las horas anteriores a la apertura enseñaría "demanda cero" en estaciones que no
    existían. Limitación: la ventana usa el primer y el último movimiento de toda la serie,
    pero solo decide qué filas existen, no aporta valores a las variables.
    """
    horas = pd.date_range(INICIO, FIN, freq="h")
    idx = pd.MultiIndex.from_product([estaciones, horas], names=["estacion", "hora_t"])
    sal = v.groupby([v["origen_id"], v["inicio_viaje"].dt.floor("h")]).size()
    lle = v.groupby([v["destino_id"], v["fin_viaje"].dt.floor("h")]).size()
    sal.index.names = lle.index.names = ["estacion", "hora_t"]
    m = pd.DataFrame(
        {
            "salidas": sal.reindex(idx, fill_value=0).astype("float32"),
            "llegadas": lle.reindex(idx, fill_value=0).astype("float32"),
        }
    ).reset_index()
    mov = pd.concat(
        [
            v[["origen_id", "inicio_viaje"]].set_axis(["estacion", "t"], axis=1),
            v[["destino_id", "fin_viaje"]].set_axis(["estacion", "t"], axis=1),
        ]
    )
    ventana = mov.groupby("estacion")["t"].agg(apertura="min", cierre="max")
    ventana["apertura"] = ventana["apertura"].dt.normalize()
    ventana["cierre"] = ventana["cierre"].dt.normalize() + pd.Timedelta(hours=23)
    m = m.merge(ventana, left_on="estacion", right_index=True)
    fuera = (m["hora_t"] < m["apertura"]) | (m["hora_t"] > m["cierre"])
    m.loc[fuera, ["salidas", "llegadas"]] = np.nan
    m["operando"] = ~fuera
    m = m.drop(columns=["apertura", "cierre"])
    return m.sort_values(["estacion", "hora_t"]).reset_index(drop=True), ventana


# ---------------------------------------------------------------------------
# Variables de P1 y P2
# ---------------------------------------------------------------------------
def agregar_atributos_estacion(m: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Agrega ubicación y capacidad de cada estación (`latitud`, `longitud`, `capacidad`).

    Por qué sirve: la demanda depende de dónde está la estación (centro, universidades) y,
    en P2, la capacidad convierte el flujo neto en una fracción de los anclajes: la misma
    diferencia llega antes al vaciado en una estación pequeña. Son atributos fijos de la
    estación, no cambian con el tiempo, así que no contienen información del futuro.
    """
    return m.merge(
        cat[["estacion_id", "latitud", "longitud", "capacidad"]],
        left_on="estacion",
        right_on="estacion_id",
        how="left",
    ).drop(columns="estacion_id")


def agregar_calendario(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Agrega hora, día de la semana, fin de semana, festivo y Semana Santa de `col`.

    Por qué sirve (P1 y P2): el uso de la bicicleta sigue la rutina semanal, con picos de
    entrada y salida entre semana y un patrón distinto en fin de semana, festivos y
    vacaciones. Se calculan sobre la hora **objetivo** (t+1): el calendario se conoce de
    antemano, así que no usa información posterior al momento de la predicción.
    """
    h = df[col]
    df["hora"] = h.dt.hour.astype("int8")
    df["dia_semana"] = h.dt.dayofweek.astype("int8")
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["es_festivo"] = h.dt.normalize().isin(FESTIVOS).astype("int8")
    df["es_semana_santa"] = h.dt.normalize().isin(SEMANA_SANTA).astype("int8")
    return df


def agregar_rezagos(m: pd.DataFrame) -> pd.DataFrame:
    """Agrega las salidas y llegadas de la hora actual y de horas pasadas de la misma estación.

    Variables: `salidas_t`, `salidas_lag_1h` (t-1), `salidas_lag_24h` (t-23) y
    `salidas_lag_168h` (t-167), y las mismas para llegadas.

    Por qué sirve: la demanda es muy autocorrelacionada. Los rezagos están definidos
    respecto a la hora objetivo t+1: t-23 es la misma hora del día anterior y t-167 la misma
    hora de la semana anterior. Todos miran hacia atrás, nunca hacia t+1. Para P2 las
    llegadas importan igual, porque el riesgo depende del flujo neto.
    """
    g = m.groupby("estacion", sort=False)
    m["salidas_t"] = m["salidas"]
    m["llegadas_t"] = m["llegadas"]
    for col in ("salidas", "llegadas"):
        m[f"{col}_lag_1h"] = g[col].shift(1)
        m[f"{col}_lag_24h"] = g[col].shift(23)
        m[f"{col}_lag_168h"] = g[col].shift(167)
    return m


def agregar_ventanas_moviles(m: pd.DataFrame) -> pd.DataFrame:
    """Agrega `salidas_media_24h`: promedio de salidas de las últimas 24 horas (t-23 a t).

    Por qué sirve (P1): suaviza el ruido de un solo rezago y resume el nivel reciente de
    actividad de la estación. La ventana termina en t, así que no incluye t+1.
    """
    g = m.groupby("estacion", sort=False)
    m["salidas_media_24h"] = g["salidas"].transform(lambda s: s.rolling(24).mean())
    return m


def agregar_indice_flujo(m: pd.DataFrame) -> pd.DataFrame:
    """Agrega el índice de flujo de la hora t y su acumulado de las últimas 3 horas.

    `indice_flujo_t` = (llegadas - salidas) / capacidad; `indice_flujo_acum_3h` es la suma
    de t-2 a t.

    Por qué sirve (P2): mide cuánto se ha llenado o vaciado la estación como fracción de sus
    anclajes. Si ya se viene vaciando o saturando, es más probable que siga. Es la misma
    magnitud que define el objetivo, pero calculada en horas pasadas.
    """
    m["indice_flujo_t"] = (m["llegadas"] - m["salidas"]) / m["capacidad"]
    m["indice_flujo_acum_3h"] = m.groupby("estacion", sort=False)["indice_flujo_t"].transform(
        lambda s: s.rolling(3).sum()
    )
    return m


def agregar_objetivos(m: pd.DataFrame, u: float = U_RIESGO) -> pd.DataFrame:
    """Agrega los objetivos en la hora siguiente t+1 (la única columna que mira al futuro).

    - `y_salidas_t1` (P1): salidas de la estación en t+1.
    - `indice_flujo_t1`: índice de flujo en t+1 (auxiliar, no es variable de entrada).
    - `y_riesgo_t1` (P2): `vaciado` si el índice en t+1 es <= -u, `saturacion` si es >= +u
      y `normal` en otro caso. Con u = 0.20 equivale a un cambio neto del 20 % de los
      anclajes en una hora (percentiles 2.5 y 97.5 en entrenamiento).
    """
    g = m.groupby("estacion", sort=False)
    m[OBJETIVO_P1] = g["salidas"].shift(-1)
    flujo_t1 = (g["llegadas"].shift(-1) - g["salidas"].shift(-1)) / m["capacidad"]
    m[AUXILIAR_P2] = flujo_t1
    m[OBJETIVO_P2] = np.select(
        [flujo_t1 <= -u, flujo_t1 >= u], ["vaciado", "saturacion"], "normal"
    )
    m.loc[flujo_t1.isna(), OBJETIVO_P2] = None
    return m


def asignar_conjunto(hora_objetivo: pd.Series) -> np.ndarray:
    """Asigna entrenamiento (hasta abril), validación (mayo) o prueba (junio).

    Por qué sirve: la partición es temporal y se hace por la hora **objetivo** (t+1). Un
    muestreo aleatorio mezclaría futuro y pasado y daría métricas demasiado optimistas.
    """
    return np.select(
        [hora_objetivo < FIN_ENTRENAMIENTO, hora_objetivo < FIN_VALIDACION],
        ["entrenamiento", "validacion"],
        "prueba",
    )


def construir_p1_p2(m: pd.DataFrame, cat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arma las tablas de P1 (demanda) y P2 (riesgo) a partir de la malla estación × hora.

    Encadena atributos de estación, rezagos, ventanas, índice de flujo, objetivos,
    calendario y partición. Conserva solo las filas con toda su historia (7 días) y con
    objetivo, dentro de la ventana de operación: una estación nueva entra cuando acumula
    7 días (arranque en frío). Devuelve `(p1, p2)` con las columnas
    `IDENTIFICADORES + FEATURES_Px + [objetivo]`; P2 incluye además `indice_flujo_t1`.
    """
    m = agregar_atributos_estacion(m, cat)
    m["hora_objetivo"] = m["hora_t"] + pd.Timedelta(hours=1)
    m = agregar_rezagos(m)
    m = agregar_ventanas_moviles(m)
    m = agregar_indice_flujo(m)
    m = agregar_objetivos(m)
    m = agregar_calendario(m, "hora_objetivo")
    m["conjunto"] = asignar_conjunto(m["hora_objetivo"])

    req = [
        "salidas_lag_168h",
        "salidas_media_24h",
        "llegadas_lag_168h",
        "indice_flujo_acum_3h",
        OBJETIVO_P1,
    ]
    m = m.dropna(subset=req).reset_index(drop=True)

    p1 = m[IDENTIFICADORES + FEATURES_P1 + [OBJETIVO_P1]].copy()
    p1[OBJETIVO_P1] = p1[OBJETIVO_P1].astype("int32")
    p2 = m[IDENTIFICADORES + FEATURES_P2 + [AUXILIAR_P2, OBJETIVO_P2]].copy()
    return p1, p2


# ---------------------------------------------------------------------------
# P3
# ---------------------------------------------------------------------------
def construir_p3(m: pd.DataFrame) -> pd.DataFrame:
    """Resume cada estación en un perfil de uso para agruparlas (clustering).

    Variables: horas en operación, movimientos totales, promedio de salidas y llegadas por
    hora, balance neto (llegadas - salidas) / movimientos, y proporción de movimientos en
    el pico de la mañana (7-8 h), el de la tarde (17-19 h) y el fin de semana.

    Por qué sirve (P3): distingue estaciones residenciales (salen en la mañana), de destino
    (reciben en la mañana), de ocio (fin de semana) o de mucho tráfico, para planeación.
    Sin fuga: usa solo el periodo de entrenamiento (enero a abril) y solo horas en
    operación, de modo que el perfil no depende de lo que ocurre en validación ni prueba.
    Las estaciones que abrieron después no tienen perfil.
    """
    t = m[(m["hora_t"] < FIN_ENTRENAMIENTO) & m["operando"]].copy()
    t["mov"] = t["salidas"] + t["llegadas"]
    h, dow = t["hora_t"].dt.hour, t["hora_t"].dt.dayofweek
    t["mov_pico_am"] = t["mov"].where(h.between(7, 8), 0)  # 7:00-8:59
    t["mov_pico_pm"] = t["mov"].where(h.between(17, 19), 0)  # 17:00-19:59
    t["mov_finde"] = t["mov"].where(dow >= 5, 0)
    g = t.groupby("estacion")
    p = pd.DataFrame(
        {
            "movimientos_totales": g["mov"].sum(),
            "prom_salidas_hora": g["salidas"].mean(),
            "prom_llegadas_hora": g["llegadas"].mean(),
            "mov_pico_am": g["mov_pico_am"].sum(),
            "mov_pico_pm": g["mov_pico_pm"].sum(),
            "mov_finde": g["mov_finde"].sum(),
            "llegadas": g["llegadas"].sum(),
            "salidas": g["salidas"].sum(),
        }
    )
    tot = p["movimientos_totales"].replace(0, np.nan)
    p["balance_neto"] = (p["llegadas"] - p["salidas"]) / tot
    p["prop_pico_am"] = p["mov_pico_am"] / tot
    p["prop_pico_pm"] = p["mov_pico_pm"] / tot
    p["prop_fin_de_semana"] = p["mov_finde"] / tot
    p["horas_operando"] = g["mov"].size()
    return p[FEATURES_P3].reset_index()


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def _exigir(condicion: bool, mensaje: str) -> None:
    """Lanza `AssertionError` si la condición es falsa.

    Por qué no `assert`: con `python -O` los `assert` se eliminan y la validación no correría.
    """
    if not condicion:
        raise AssertionError(mensaje)


def validar_sin_fuga(viajes: pd.DataFrame, p1: pd.DataFrame, n: int = 2000) -> None:
    """Comprueba sobre una muestra de P1 que no haya fuga de información.

    Recalcula desde los viajes originales que `y_salidas_t1` sean exactamente las salidas de
    la hora siguiente y que `salidas_t` y `salidas_lag_168h` sean salidas de t y de t-167.
    Falla con `AssertionError` si algo no coincide, o si hay nulos en P1.
    """
    conteo = viajes.groupby([viajes["origen_id"], viajes["inicio_viaje"].dt.floor("h")]).size()
    salidas_en = lambda e, h: conteo.get((e, h), 0)
    muestra = p1.sample(min(n, len(p1)), random_state=SEED)
    H = pd.Timedelta(hours=1)

    _exigir(
        all(
            salidas_en(e, h + H) == y
            for e, h, y in zip(muestra.estacion, muestra.hora_t, muestra[OBJETIVO_P1])
        ),
        f"{OBJETIVO_P1} no coincide con las salidas de la hora siguiente",
    )
    _exigir(
        all(
            salidas_en(e, h) == x
            for e, h, x in zip(muestra.estacion, muestra.hora_t, muestra.salidas_t)
        ),
        "salidas_t no coincide con las salidas de la hora t",
    )
    _exigir(
        all(
            salidas_en(e, h - 167 * H) == x
            for e, h, x in zip(muestra.estacion, muestra.hora_t, muestra.salidas_lag_168h)
        ),
        "salidas_lag_168h no coincide con las salidas de t-167",
    )
    _exigir(p1.isna().sum().sum() == 0, "P1 tiene valores nulos")


class FeatureBuilder:
    """Ejecuta el pipeline de variables: malla, P1, P2 y P3, con validación y guardado."""

    def build(
        self,
        viajes: pd.DataFrame,
        cat: pd.DataFrame,
        output_dir: Path,
        resumen: dict | None = None,
    ) -> dict:
        """Construye P1, P2 y P3, valida que no haya fuga y guarda Parquet y resumen.json."""
        estaciones = sorted(set(viajes["origen_id"]) | set(viajes["destino_id"]))
        _exigir(
            (cat.set_index("estacion_id").loc[estaciones, "capacidad"] > 0).all(),
            "hay estaciones sin capacidad positiva en el catálogo",
        )
        m, _ = malla_estacion_hora(viajes, estaciones)
        p1, p2 = construir_p1_p2(m, cat)
        p3 = construir_p3(m)
        validar_sin_fuga(viajes, p1)
        _exigir(p2.isna().sum().sum() == 0, "P2 tiene valores nulos")

        output_dir.mkdir(parents=True, exist_ok=True)
        p1.to_parquet(output_dir / "p1_demanda.parquet", index=False)
        p2.to_parquet(output_dir / "p2_riesgo.parquet", index=False)
        p3.to_parquet(output_dir / "p3_perfiles.parquet", index=False)

        resumen = dict(resumen or {})
        resumen.update(
            {
                "estaciones": len(estaciones),
                "filas_malla": len(m),
                "filas_malla_operando": int(m["operando"].sum()),
                "filas_p1": len(p1),
                "filas_p2": len(p2),
                "estaciones_p1": int(p1["estacion"].nunique()),
                "estaciones_p3": len(p3),
            }
        )
        (output_dir / "resumen.json").write_text(
            json.dumps(resumen, ensure_ascii=False, indent=2, default=int), encoding="utf-8"
        )
        return resumen


@app.command()
def main(
    viajes_path: Path = INTERIM_DATA_DIR / "viajes_limpios.parquet",
    nomenclatura_path: Path = INTERIM_DATA_DIR / "nomenclatura.parquet",
    resumen_limpieza_path: Path = INTERIM_DATA_DIR / "resumen_limpieza.json",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    """Construye las tablas P1, P2 y P3 desde los viajes limpios y las guarda en `data/processed/`."""
    logger.info("Generating features from dataset...")
    viajes = pd.read_parquet(viajes_path)
    cat = pd.read_parquet(nomenclatura_path)
    resumen = json.loads(resumen_limpieza_path.read_text(encoding="utf-8"))
    with corrida_proyecto("Construcción de variables"):
        r = FeatureBuilder().build(viajes, cat, output_dir, resumen)
    logger.success(
        f"Features generation complete: P1/P2 {r['filas_p1']:,} filas, "
        f"P3 {r['estaciones_p3']} estaciones -> {output_dir}"
    )


if __name__ == "__main__":
    app()
