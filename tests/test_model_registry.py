"""Pruebas unitarias del registro de modelos de MiBici."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from mibici_mlops_module.modeling.registry import (
    nombre_registro,
    validar_origen,
    version_con_alias,
)


# PRUEBA 1: Nombres correctos para los tres modelos
@pytest.mark.parametrize(
    "pregunta,esperado",
    [
        ("p1", "mibici-equipo04-p1-demanda"),
        ("p2", "mibici-equipo04-p2-riesgo"),
        ("p3", "mibici-equipo04-p3-grupos"),
    ],
)
def test_nombre_registro_valido(pregunta, esperado):
    assert nombre_registro(pregunta) == esperado


# PRUEBA 2: Rechazar preguntas inexistentes
def test_nombre_registro_invalido():
    with pytest.raises(KeyError):
        nombre_registro("p4")


# PRUEBA 3: Aceptar una corrida válida
def test_validar_origen_correcto():
    cliente = Mock()

    corrida = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(
            params={"fraccion_entrenamiento": "1.0"}
        ),
    )

    cliente.get_run.return_value = corrida

    guardado = SimpleNamespace(
        run_id="run_prueba_123",
        pregunta="p2",
    )

    resultado = validar_origen(cliente, guardado)

    assert resultado is corrida
    cliente.get_run.assert_called_once_with("run_prueba_123")


# PRUEBA 4: Rechazar una corrida que no terminó
def test_validar_origen_corrida_incompleta():
    cliente = Mock()

    cliente.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FAILED"),
        data=SimpleNamespace(
            params={"fraccion_entrenamiento": "1.0"}
        ),
    )

    guardado = SimpleNamespace(
        run_id="run_fallido",
        pregunta="p2",
    )

    with pytest.raises(ValueError, match="no terminó"):
        validar_origen(cliente, guardado)


# PRUEBA 5: Rechazar entrenamiento con datos parciales
def test_validar_origen_fraccion_incompleta():
    cliente = Mock()

    cliente.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(
            params={"fraccion_entrenamiento": "0.1"}
        ),
    )

    guardado = SimpleNamespace(
        run_id="run_parcial",
        pregunta="p2",
    )

    with pytest.raises(ValueError, match="fracción"):
        validar_origen(cliente, guardado)


# PRUEBA 6: Recuperar versión con alias existente
def test_version_con_alias_existente():
    cliente = Mock()

    version = SimpleNamespace(
        version="1",
        run_id="run_prueba_123",
    )

    cliente.get_model_version_by_alias.return_value = version

    resultado = version_con_alias(
        cliente,
        "mibici-equipo04-p2-riesgo",
        "champion",
    )

    assert resultado is version


# PRUEBA 7: Alias inexistente
def test_version_con_alias_inexistente():
    cliente = Mock()

    cliente.get_model_version_by_alias.side_effect = (
        MlflowException("Alias no encontrado")
    )

    resultado = version_con_alias(
        cliente,
        "mibici-equipo04-p2-riesgo",
        "champion",
    )

    assert resultado is None