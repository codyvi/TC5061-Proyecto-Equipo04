
"""Pruebas de carga de modelos desde MLflow Model Registry."""

from unittest.mock import MagicMock, patch

import pytest

from mibici_mlops_module.model_loader import cargar_modelo_registry


@patch("mibici_mlops_module.model_loader.mlflow.pyfunc.load_model")
@patch("mibici_mlops_module.model_loader.mlflow.set_tracking_uri")
def test_cargar_champion(mock_tracking, mock_load, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mock_load.return_value = MagicMock()

    modelo = cargar_modelo_registry("p1", "champion")

    mock_tracking.assert_called_once_with("http://localhost:5000")
    mock_load.assert_called_once_with(
        "models:/mibici-equipo04-p1-demanda@champion"
    )
    assert modelo is mock_load.return_value


@patch("mibici_mlops_module.model_loader.mlflow.pyfunc.load_model")
def test_cargar_version(mock_load, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    cargar_modelo_registry("p2", version=3)

    mock_load.assert_called_once_with(
        "models:/mibici-equipo04-p2-riesgo/3"
    )


def test_pregunta_invalida():
    with pytest.raises(ValueError, match="Pregunta no válida"):
        cargar_modelo_registry("p4")


def test_sin_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        cargar_modelo_registry("p1")


@patch("mibici_mlops_module.model_loader.mlflow.pyfunc.load_model")
def test_error_carga(mock_load, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mock_load.side_effect = Exception("Modelo no encontrado")

    with pytest.raises(RuntimeError, match="No se pudo cargar"):
        cargar_modelo_registry("p1")
