"""Reproducibilidad del entrenamiento con el código real de `P1Trainer`.

Complementa `test_reproducibilidad.py` (que usa un modelo de juguete): aquí se entrenan los
candidatos que el proyecto define, a partir de la misma tabla y de la semilla del proyecto.
"""

import numpy as np
import pytest

from mibici_mlops_module.config import SEED
from mibici_mlops_module.modeling.train import P1Trainer

HP_RAPIDOS = {"n_estimators": 8, "max_depth": 4, "min_samples_leaf": 5, "max_samples": 0.5}


def _entrenar(tabla, seed, modelo="rf"):
    """Entrena el candidato `modelo` de P1 y devuelve (entrenador, candidato)."""
    t = P1Trainer(seed=seed, tabla_path=tabla, modelo=modelo, hp=HP_RAPIDOS)
    t.cargar()
    cand = next(c for c in t.candidatos() if not c.baseline)
    t.ajustar(cand.modelo)
    return t, cand


def test_misma_semilla_mismas_predicciones_y_metricas(tabla_p1):
    t1, c1 = _entrenar(tabla_p1, SEED)
    t2, c2 = _entrenar(tabla_p1, SEED)
    X = t1.eval["validacion"][t1.features]

    # Tolerancia mínima, no igualdad exacta: con `n_jobs=-1` los árboles se promedian en un orden
    # que varía entre hilos y cambia el último decimal (~1e-15), sin relación con la semilla.
    np.testing.assert_allclose(c1.modelo.predict(X), c2.modelo.predict(X), rtol=1e-9, atol=0)
    m1, m2 = t1.evaluar(c1), t2.evaluar(c2)
    assert m1.keys() == m2.keys()
    assert m1 == pytest.approx(m2, rel=1e-9)


def test_semilla_distinta_cambia_el_modelo(tabla_p1):
    """Control negativo: si la semilla no influyera, la prueba anterior no demostraría nada."""
    t1, c1 = _entrenar(tabla_p1, 1)
    _, c2 = _entrenar(tabla_p1, 2)
    X = t1.eval["validacion"][t1.features]
    assert not np.array_equal(c1.modelo.predict(X), c2.modelo.predict(X))


def test_submuestreo_de_entrenamiento_es_reproducible(tabla_p1):
    """Límite: con fraccion < 1 el submuestreo usa la semilla y elige las mismas filas."""
    filas = []
    for _ in range(2):
        t = P1Trainer(fraccion=0.5, seed=SEED, tabla_path=tabla_p1, modelo="rf", hp=HP_RAPIDOS)
        t.cargar()
        filas.append(t.train.index.to_list())
    assert filas[0] == filas[1]
    assert len(filas[0]) == 200  # 400 filas de entrenamiento * 0.5


def test_el_entrenamiento_solo_usa_las_variables_declaradas(tabla_p1):
    """Límite: el modelo recibe FEATURES_P1, no 'todas las columnas menos el objetivo'."""
    t, cand = _entrenar(tabla_p1, SEED)
    assert list(cand.modelo.feature_names_in_) == t.features
    assert t.objetivo not in cand.modelo.feature_names_in_


def test_modelo_desconocido_se_rechaza(tabla_p1):
    import typer

    with pytest.raises(typer.BadParameter):
        P1Trainer(tabla_path=tabla_p1, modelo="no-existe")
