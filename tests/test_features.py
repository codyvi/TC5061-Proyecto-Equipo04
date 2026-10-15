"""Pruebas unitarias de ingeniería de variables MiBici (Equipo 04).

Ejecutar desde la raíz del repositorio:
    python -m pytest tests/test_features.py -v
"""

import pandas as pd
import pytest

from mibici_mlops_module.features import (
    agregar_objetivos,
    agregar_rezagos,
    agregar_ventanas_moviles,
    asignar_conjunto,
    validar_sin_fuga,
)
from mibici_mlops_module.config import FIN_ENTRENAMIENTO, FIN_VALIDACION


def _malla_ejemplo(n=200):
    """Serie horaria de dos estaciones con datos conocidos."""
    horas = pd.date_range('2026-01-01', periods=n, freq='h')
    registros = []
    for estacion, factor in [(101, 1), (202, 2)]:
        for i, hora in enumerate(horas):
            registros.append({
                'estacion': estacion,
                'hora_t': hora,
                'salidas': float(i * factor),
                'llegadas': float(i * factor + 1),
                'capacidad': 20,
            })
    return pd.DataFrame(registros)


def test_rezagos_no_mezclan_estaciones_ni_usan_futuro():
    m = agregar_rezagos(_malla_ejemplo())
    for estacion, factor in [(101, 1), (202, 2)]:
        fila = m.loc[(m['estacion'] == estacion) & (m['hora_t'] == pd.Timestamp('2026-01-09 00:00'))].iloc[0]
        assert fila['salidas_t'] == 192 * factor
        assert fila['salidas_lag_1h'] == 191 * factor
        assert fila['salidas_lag_24h'] == 169 * factor
        assert fila['salidas_lag_168h'] == 25 * factor


def test_media_movil_solo_usa_horas_pasadas_y_actual():
    m = agregar_ventanas_moviles(_malla_ejemplo())
    fila = m.loc[(m['estacion'] == 101) & (m['hora_t'] == pd.Timestamp('2026-01-02 00:00'))].iloc[0]
    assert fila['salidas_media_24h'] == pytest.approx(sum(range(1, 25)) / 24)


def test_objetivos_son_de_la_hora_siguiente():
    m = agregar_objetivos(_malla_ejemplo(n=4), u=0.20)
    primera = m.loc[m['estacion'] == 101].iloc[0]
    assert primera['y_salidas_t1'] == 1
    assert primera['indice_flujo_t1'] == pytest.approx(1 / 20)
    assert primera['y_riesgo_t1'] == 'normal'
    ultima = m.loc[m['estacion'] == 101].iloc[-1]
    assert pd.isna(ultima['y_salidas_t1'])
    assert pd.isna(ultima['y_riesgo_t1'])


def test_clases_de_riesgo_segun_umbral():
    m = pd.DataFrame({
        'estacion': [101, 101, 101, 101],
        'salidas': [0.0, 6.0, 0.0, 0.0],
        'llegadas': [0.0, 0.0, 6.0, 0.0],
        'capacidad': [20, 20, 20, 20],
    })
    resultado = agregar_objetivos(m, u=0.20)
    assert resultado['y_riesgo_t1'].iloc[0] == 'vaciado'
    assert resultado['y_riesgo_t1'].iloc[1] == 'saturacion'
    assert resultado['y_riesgo_t1'].iloc[2] == 'normal'


def test_particion_temporal_respetada():
    horas = pd.Series([
        pd.Timestamp(FIN_ENTRENAMIENTO) - pd.Timedelta(hours=1),
        pd.Timestamp(FIN_ENTRENAMIENTO),
        pd.Timestamp(FIN_VALIDACION) - pd.Timedelta(hours=1),
        pd.Timestamp(FIN_VALIDACION),
    ])
    assert list(asignar_conjunto(horas)) == [
        'entrenamiento', 'validacion', 'validacion', 'prueba'
    ]


def test_validacion_sin_fuga_acepta_muestras_pequenas():
    """Prueba de regresión: debe pasar tras ajustar sample(min(n, len(p1)))."""
    viajes = pd.DataFrame({
        'origen_id': [101],
        'inicio_viaje': [pd.Timestamp('2026-01-01 01:00')],
    })
    p1 = pd.DataFrame({
        'estacion': [101],
        'hora_t': [pd.Timestamp('2026-01-01 00:00')],
        'y_salidas_t1': [1],
        'salidas_t': [0],
        'salidas_lag_168h': [0],
    })
    validar_sin_fuga(viajes, p1)
