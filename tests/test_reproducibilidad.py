"""Pruebas de reproducibilidad de modelos de Machine Learning."""

import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


@pytest.fixture
def datos_sinteticos():
    """Genera datos sintéticos reproducibles para clasificación."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )

    return X[:240], X[240:], y[:240], y[240:]


def entrenar_modelo(X, y, semilla):
    """Entrena un Random Forest con parámetros fijos."""
    modelo = RandomForestClassifier(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=semilla,
        n_jobs=1,
    )

    modelo.fit(X, y)
    return modelo


def test_predicciones_reproducibles(datos_sinteticos):
    """La misma semilla debe producir predicciones idénticas."""
    X_train, X_test, y_train, _ = datos_sinteticos

    modelo_1 = entrenar_modelo(X_train, y_train, semilla=42)
    modelo_2 = entrenar_modelo(X_train, y_train, semilla=42)

    predicciones_1 = modelo_1.predict(X_test)
    predicciones_2 = modelo_2.predict(X_test)

    np.testing.assert_array_equal(predicciones_1, predicciones_2)


def test_metricas_reproducibles(datos_sinteticos):
    """La misma semilla debe producir la misma balanced accuracy."""
    X_train, X_test, y_train, y_test = datos_sinteticos

    modelo_1 = entrenar_modelo(X_train, y_train, semilla=42)
    modelo_2 = entrenar_modelo(X_train, y_train, semilla=42)

    metrica_1 = balanced_accuracy_score(
        y_test, modelo_1.predict(X_test)
    )

    metrica_2 = balanced_accuracy_score(
        y_test, modelo_2.predict(X_test)
    )

    assert metrica_1 == pytest.approx(metrica_2, abs=1e-12)


def test_probabilidades_reproducibles(datos_sinteticos):
    """Las probabilidades también deben ser idénticas."""
    X_train, X_test, y_train, _ = datos_sinteticos

    modelo_1 = entrenar_modelo(X_train, y_train, semilla=42)
    modelo_2 = entrenar_modelo(X_train, y_train, semilla=42)

    np.testing.assert_allclose(
        modelo_1.predict_proba(X_test),
        modelo_2.predict_proba(X_test),
        rtol=0,
        atol=0,
    )