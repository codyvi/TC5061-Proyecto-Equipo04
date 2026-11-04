"""Pruebas de la API (FastAPI) con modelos simulados; no requieren servidor MLflow."""

from unittest.mock import MagicMock, patch

from app import main as api
from fastapi.testclient import TestClient
import numpy as np
import pytest

from mibici_mlops_module.features import FEATURES_P1, FEATURES_P2

P1 = api.EntradaP1.model_config["json_schema_extra"]["example"]
P2 = api.EntradaP2.model_config["json_schema_extra"]["example"]


def _modelos():
    p1, p2 = MagicMock(), MagicMock()
    p1.predict.return_value = np.array([4.2])
    p2.predict.return_value = np.array(["saturación"])
    return {"p1": p1, "p2": p2}


@pytest.fixture
def client(monkeypatch):
    modelos = _modelos()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    with (
        patch.object(api, "cargar_modelo_registry", side_effect=lambda p, *a: modelos[p]),
        TestClient(api.app) as c,
    ):
        c.modelos = modelos
        yield c


def test_esquemas_coinciden_con_features():
    assert list(api.EntradaP1.model_fields) == FEATURES_P1
    assert list(api.EntradaP2.model_fields) == FEATURES_P2


def test_health(client):
    r = client.get("/health")
    assert r.json() == {"estado": "ok", "modelos_cargados": ["p1", "p2"]}


def test_predict_p1(client):
    r = client.post("/predict/p1", json=P1)
    assert r.status_code == 200
    assert r.json() == {"pred_salidas_t1": pytest.approx(4.2), "modelo": "p1"}
    X = client.modelos["p1"].predict.call_args[0][0]
    assert list(X.columns) == FEATURES_P1


def test_predict_p2(client):
    r = client.post("/predict/p2", json=P2)
    assert r.status_code == 200
    assert r.json() == {"pred_riesgo_t1": "saturación", "modelo": "p2"}


@pytest.mark.parametrize(
    "cambio",
    [{"hora": 24}, {"dia_semana": -1}, {"es_festivo": 2}, {"salidas_t": -1}, {"latitud": 0}, {"hora": "x"}],
)
def test_entrada_invalida(client, cambio):
    assert client.post("/predict/p1", json={**P1, **cambio}).status_code == 422


def test_campo_faltante_o_extra(client):
    faltante = {k: v for k, v in P1.items() if k != "salidas_t"}
    assert client.post("/predict/p1", json=faltante).status_code == 422
    assert client.post("/predict/p1", json={**P1, "otro": 1}).status_code == 422


def test_modelo_no_disponible(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    with (
        patch.object(api, "cargar_modelo_registry", side_effect=RuntimeError("sin registry")),
        TestClient(api.app) as c,
    ):
        assert c.get("/health").json()["estado"] == "degradado"
        assert c.post("/predict/p1", json=P1).status_code == 503
