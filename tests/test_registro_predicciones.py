"""Pruebas del registro de predicciones de la API y del drift sobre el tráfico real.

No requieren servidor MLflow: los modelos se simulan como en `test_api.py`.
"""

import json
from unittest.mock import MagicMock, patch

from app import main as api
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici_mlops_module import monitoreo
from mibici_mlops_module.features import FEATURES_P1
from mibici_mlops_module.registro_predicciones import (
    ENV_ACTIVO,
    ENV_RUTA,
    RegistroPredicciones,
    registro_activo,
)

P1 = api.EntradaP1.model_config["json_schema_extra"]["example"]
P2 = api.EntradaP2.model_config["json_schema_extra"]["example"]


def _modelos():
    p1, p2 = MagicMock(), MagicMock()
    p1.predict.return_value = np.array([4.2])
    p2.predict.return_value = np.array(["saturación"])
    p1.metadata.run_id = "run-p1"
    return {"p1": p1, "p2": p2}


@pytest.fixture
def cliente(monkeypatch, tmp_path):
    """API con modelos simulados y el registro activo en un archivo temporal."""
    ruta = tmp_path / "predicciones.jsonl"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv(ENV_ACTIVO, "1")
    monkeypatch.setenv(ENV_RUTA, str(ruta))
    modelos = _modelos()
    with (
        patch.object(api, "cargar_modelo_registry", side_effect=lambda p, *a: modelos[p]),
        TestClient(api.app) as c,
    ):
        c.ruta = ruta
        yield c


# ---------------------------------------------------------------------------
# Clase RegistroPredicciones
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("valor, esperado", [("1", True), ("true", True), ("SI", True),
                                             ("0", False), ("", False), ("no", False)])
def test_registro_activo_por_variable(monkeypatch, valor, esperado):
    monkeypatch.setenv(ENV_ACTIVO, valor)
    assert registro_activo() is esperado


def test_apagado_no_escribe(tmp_path):
    ruta = tmp_path / "r.jsonl"
    registro = RegistroPredicciones(ruta, activo=False)
    assert registro.registrar("p1", {"hora": 8}, 1.0) is None
    assert not ruta.exists()


def test_registrar_y_leer(tmp_path):
    registro = RegistroPredicciones(tmp_path / "sub" / "r.jsonl")  # crea la carpeta si falta
    registro.registrar("p1", {"hora": np.int8(8), "salidas_t": np.float32(5)}, np.float64(4.2),
                       modelo="champion", run_id="abc", latencia_ms=1.23456)
    registro.registrar("p2", {"hora": 9}, "vaciado", latencia_ms=2)
    df = registro.leer()
    assert len(df) == 2
    assert {"fecha_utc", "pregunta", "prediccion", "modelo", "run_id", "latencia_ms", "hora"} <= set(df)
    p1 = registro.leer(pregunta="p1")
    assert p1.loc[0, "hora"] == 8 and p1.loc[0, "prediccion"] == pytest.approx(4.2)
    assert p1.loc[0, "latencia_ms"] == pytest.approx(1.235)


def test_lineas_danadas_se_omiten(tmp_path):
    ruta = tmp_path / "r.jsonl"
    registro = RegistroPredicciones(ruta)
    registro.registrar("p1", {"hora": 8}, 1.0)
    with ruta.open("a") as f:
        f.write('{"pregunta": "p1", "entradas": {"ho\n\n')  # escritura cortada
    registro.registrar("p1", {"hora": 9}, 2.0)
    assert len(registro.leer()) == 2


def test_resumen(tmp_path):
    registro = RegistroPredicciones(tmp_path / "r.jsonl")
    assert registro.resumen() == {"activo": True, "peticiones": 0, "ventana": 10_000,
                                  "por_pregunta": {}}
    for i in range(10):
        registro.registrar("p1", {"hora": i}, float(i), modelo="champion", latencia_ms=i)
    for clase in ["vaciado", "normal", "normal", "saturación"]:
        registro.registrar("p2", {"hora": 8}, clase, modelo="champion", latencia_ms=1)
    r = registro.resumen()
    assert r["peticiones"] == 14
    assert r["por_pregunta"]["p1"]["peticiones"] == 10
    assert r["por_pregunta"]["p1"]["prediccion_media"] == pytest.approx(4.5)
    assert r["por_pregunta"]["p1"]["latencia_ms_p50"] == pytest.approx(4.5)
    assert r["por_pregunta"]["p2"]["distribucion_prediccion"] == {
        "normal": 0.5, "saturación": 0.25, "vaciado": 0.25}
    assert registro.resumen(ventana=4)["peticiones"] == 4  # solo las últimas


# ---------------------------------------------------------------------------
# Integración con la API
# ---------------------------------------------------------------------------
def test_api_registra_cada_prediccion(cliente):
    assert cliente.post("/predict/p1", json=P1).status_code == 200
    assert cliente.post("/predict/p2", json=P2).status_code == 200
    filas = [json.loads(linea) for linea in cliente.ruta.read_text().splitlines()]
    assert [f["pregunta"] for f in filas] == ["p1", "p2"]
    assert filas[0]["entradas"] == P1
    assert filas[0]["prediccion"] == pytest.approx(4.2)
    assert filas[0]["modelo"] == "champion" and filas[0]["run_id"] == "run-p1"
    assert filas[0]["latencia_ms"] >= 0
    assert filas[1]["prediccion"] == "saturación" and filas[1]["run_id"] is None


def test_api_no_registra_entradas_invalidas(cliente):
    assert cliente.post("/predict/p1", json={**P1, "hora": 99}).status_code == 422
    assert not cliente.ruta.exists()


def test_endpoint_resumen(cliente):
    for _ in range(3):
        cliente.post("/predict/p1", json=P1)
    cliente.post("/predict/p2", json=P2)
    r = cliente.get("/monitoreo/resumen")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["activo"] is True and cuerpo["peticiones"] == 4
    assert cuerpo["por_pregunta"]["p1"]["prediccion_media"] == pytest.approx(4.2)
    assert cuerpo["por_pregunta"]["p2"]["distribucion_prediccion"] == {"saturación": 1.0}
    assert cliente.get("/monitoreo/resumen", params={"ventana": 0}).status_code == 422


def test_api_sin_registro_por_defecto(monkeypatch, tmp_path):
    """Sin la variable de entorno la API se comporta como antes y no escribe nada."""
    ruta = tmp_path / "r.jsonl"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.delenv(ENV_ACTIVO, raising=False)
    monkeypatch.setenv(ENV_RUTA, str(ruta))
    modelos = _modelos()
    with (
        patch.object(api, "cargar_modelo_registry", side_effect=lambda p, *a: modelos[p]),
        TestClient(api.app) as c,
    ):
        assert c.post("/predict/p1", json=P1).status_code == 200
        assert c.get("/monitoreo/resumen").json()["activo"] is False
    assert not ruta.exists()


def test_falla_del_registro_no_tumba_la_api(cliente):
    with patch.object(RegistroPredicciones, "registrar", side_effect=OSError("disco lleno")):
        r = cliente.post("/predict/p1", json=P1)
    assert r.status_code == 200 and r.json()["pred_salidas_t1"] == pytest.approx(4.2)


# ---------------------------------------------------------------------------
# Drift sobre el tráfico real (conjuntos de entrada al reporte de Evidently)
# ---------------------------------------------------------------------------
def test_conjuntos_produccion(tabla_p1, tmp_path):
    ruta = tmp_path / "r.jsonl"
    registro = RegistroPredicciones(ruta)
    fila = {c: 1 for c in FEATURES_P1}
    for i in range(60):
        registro.registrar("p1", {**fila, "hora": i % 24}, 1.0)
    registro.registrar("p2", {"hora": 1}, "normal")  # otra pregunta: no debe colarse

    # `cargar_conjuntos` busca `p1_demanda.parquet` en processed_dir: se usa la tabla sintética
    (tmp_path / "p1_demanda.parquet").write_bytes(tabla_p1.read_bytes())
    ref, act = monitoreo.cargar_conjuntos_produccion("p1", ruta, processed_dir=tmp_path)
    assert list(act.columns) == list(ref.columns) == FEATURES_P1
    assert len(act) == 60
    assert (act.dtypes == ref.dtypes).all()
    pd.testing.assert_series_equal(act["hora"], pd.Series([i % 24 for i in range(60)],
                                                          name="hora", dtype=ref["hora"].dtype))


def test_conjuntos_produccion_sin_trafico(tmp_path):
    with pytest.raises(ValueError, match="REGISTRO_PREDICCIONES"):
        monitoreo.cargar_conjuntos_produccion("p1", tmp_path / "no_existe.jsonl")


def test_reporte_produccion(tabla_p1, tmp_path):
    pytest.importorskip("evidently")
    ruta = tmp_path / "r.jsonl"
    registro = RegistroPredicciones(ruta)
    rng = np.random.default_rng(0)
    for _ in range(80):
        entradas = {c: float(rng.poisson(5)) for c in FEATURES_P1}
        entradas.update(hora=int(rng.integers(0, 24)), dia_semana=int(rng.integers(0, 7)),
                        es_fin_de_semana=0, es_festivo=0, es_semana_santa=0,
                        latitud=20.65, longitud=-103.35)
        registro.registrar("p1", entradas, 1.0)
    (tmp_path / "p1_demanda.parquet").write_bytes(tabla_p1.read_bytes())
    salida = tmp_path / "drift"
    info = monitoreo.generar_reporte_produccion("p1", ruta, salida=salida, processed_dir=tmp_path)
    assert info["filas_actual"] == 80
    assert (salida / "drift_p1_produccion.html").exists()
    assert (salida / "drift_p1_produccion.json").exists()
