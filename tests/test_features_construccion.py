"""Pruebas de la construcción de la malla y de las tablas de ingeniería de variables."""

import numpy as np
import pandas as pd
import pytest

from mibici_mlops_module.config import FESTIVOS, FIN_ENTRENAMIENTO, INICIO, SEMANA_SANTA
from mibici_mlops_module.features import (
    FEATURES_P3,
    agregar_calendario,
    agregar_indice_flujo,
    construir_p3,
    malla_estacion_hora,
)


# ---- malla_estacion_hora ------------------------------------------------------------------
def _viajes():
    """Estación 1: viaja del 2 al 3 de enero. Estación 2: solo recibe el 5 de enero."""
    return pd.DataFrame(
        {
            "origen_id": [1, 1, 1],
            "destino_id": [2, 2, 1],
            "inicio_viaje": pd.to_datetime(
                ["2026-01-02 10:15", "2026-01-02 10:45", "2026-01-03 08:00"]
            ),
            "fin_viaje": pd.to_datetime(
                ["2026-01-02 10:30", "2026-01-02 11:05", "2026-01-05 09:00"]
            ),
        }
    )


@pytest.fixture(scope="module")
def malla():
    m, ventana = malla_estacion_hora(_viajes(), [1, 2])
    return m.set_index(["estacion", "hora_t"]), ventana


def test_malla_cuenta_salidas_y_llegadas_por_hora(malla):
    m, _ = malla
    h10 = pd.Timestamp("2026-01-02 10:00")
    assert m.loc[(1, h10), "salidas"] == 2
    assert m.loc[(2, h10), "llegadas"] == 1
    assert m.loc[(2, pd.Timestamp("2026-01-02 11:00")), "llegadas"] == 1


def test_malla_horas_sin_viajes_valen_cero_dentro_de_la_ventana(malla):
    m, _ = malla
    assert m.loc[(1, pd.Timestamp("2026-01-02 12:00")), "salidas"] == 0


def test_malla_antes_de_la_apertura_es_nan_y_no_opera(malla):
    """Límite: la estación 2 abre el 2 de enero; antes no debe aparecer como demanda cero."""
    m, ventana = malla
    antes = m.loc[(2, pd.Timestamp("2026-01-01 12:00"))]
    assert np.isnan(antes["salidas"]) and np.isnan(antes["llegadas"])
    assert not antes["operando"]
    assert ventana.loc[2, "apertura"] == pd.Timestamp("2026-01-02")


def test_malla_despues_del_cierre_es_nan(malla):
    m, ventana = malla
    assert ventana.loc[2, "cierre"] == pd.Timestamp("2026-01-02 23:00")
    assert np.isnan(m.loc[(2, pd.Timestamp("2026-01-03 00:00")), "salidas"])


def test_malla_tiene_una_fila_por_estacion_y_hora(malla):
    m, _ = malla
    horas = len(pd.date_range(INICIO, "2026-06-30 23:00", freq="h"))
    assert len(m) == 2 * horas
    assert not m.index.duplicated().any()


# ---- agregar_calendario -------------------------------------------------------------------
def test_calendario_marca_fin_de_semana_festivo_y_semana_santa():
    horas = pd.DataFrame(
        {
            "h": pd.to_datetime(
                [
                    "2026-01-03 17:00",  # sábado
                    FESTIVOS[0],  # 1 de enero (jueves, festivo)
                    SEMANA_SANTA[0],  # lunes 30 de marzo
                    "2026-02-10 08:00",  # martes sin nada especial
                ]
            )
        }
    )
    r = agregar_calendario(horas, "h")
    assert list(r["es_fin_de_semana"]) == [1, 0, 0, 0]
    assert list(r["es_festivo"]) == [0, 1, 0, 0]
    assert list(r["es_semana_santa"]) == [0, 0, 1, 0]
    assert list(r["hora"]) == [17, 0, 0, 8]
    assert list(r["dia_semana"]) == [5, 3, 0, 1]


def test_calendario_tipos_compactos():
    r = agregar_calendario(pd.DataFrame({"h": pd.to_datetime(["2026-01-01 05:00"])}), "h")
    assert all(r[c].dtype == "int8" for c in ("hora", "dia_semana", "es_festivo"))


# ---- agregar_indice_flujo -----------------------------------------------------------------
def test_indice_flujo_y_acumulado_de_3h():
    m = pd.DataFrame(
        {
            "estacion": [1] * 4,
            "salidas": [0.0, 2.0, 0.0, 4.0],
            "llegadas": [2.0, 0.0, 2.0, 0.0],
            "capacidad": [10] * 4,
        }
    )
    r = agregar_indice_flujo(m)
    assert list(r["indice_flujo_t"]) == pytest.approx([0.2, -0.2, 0.2, -0.4])
    assert r["indice_flujo_acum_3h"].isna().sum() == 2  # límite: faltan 2 horas de historia
    assert r["indice_flujo_acum_3h"].iloc[2] == pytest.approx(0.2)
    assert r["indice_flujo_acum_3h"].iloc[3] == pytest.approx(-0.4)


# ---- construir_p3 -------------------------------------------------------------------------
def _malla_p3():
    """Estación 1 opera 48 h; estación 2 nunca tiene movimientos; estación 3 solo en mayo."""
    horas = pd.date_range("2026-01-05", periods=48, freq="h")  # lunes y martes
    uno = pd.DataFrame(
        {"estacion": 1, "hora_t": horas, "salidas": 1.0, "llegadas": 3.0, "operando": True}
    )
    dos = uno.assign(estacion=2, salidas=0.0, llegadas=0.0)
    tres = uno.assign(estacion=3, hora_t=FIN_ENTRENAMIENTO + pd.to_timedelta(range(48), "h"))
    return pd.concat([uno, dos, tres], ignore_index=True)


def test_p3_resume_cada_estacion_con_las_columnas_esperadas():
    p3 = construir_p3(_malla_p3()).set_index("estacion")
    assert list(p3.columns) == FEATURES_P3
    fila = p3.loc[1]
    assert fila["horas_operando"] == 48
    assert fila["movimientos_totales"] == 48 * 4
    assert fila["prom_salidas_hora"] == pytest.approx(1)
    assert fila["balance_neto"] == pytest.approx(0.5)  # (3 - 1) / 4
    assert fila["prop_fin_de_semana"] == 0  # solo lunes y martes


def test_p3_excluye_periodo_posterior_al_entrenamiento():
    """Sin fuga: la estación que solo tiene datos desde mayo no recibe perfil."""
    assert 3 not in construir_p3(_malla_p3())["estacion"].values


def test_p3_estacion_sin_movimientos_tiene_proporciones_nan_y_no_division_por_cero():
    """Límite: movimientos_totales = 0 no debe dar infinitos."""
    p3 = construir_p3(_malla_p3()).set_index("estacion")
    assert p3.loc[2, "movimientos_totales"] == 0
    assert p3.loc[2, ["balance_neto", "prop_pico_am", "prop_fin_de_semana"]].isna().all()
    assert not np.isinf(p3.select_dtypes("number").to_numpy()).any()
