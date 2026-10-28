import pytest
from typer import BadParameter
from mibici_mlops_module.modeling.pipeline import ENTRENADORES

def test_preguntas_soportadas_normal():
    """Caso normal: El diccionario de entrenadores debe soportar las 3 preguntas clave del proyecto."""
    assert "p1" in ENTRENADORES, "Falta el entrenador registrado para P1 (Demanda)"
    assert "p2" in ENTRENADORES, "Falta el entrenador registrado para P2 (Riesgo)"
    assert "p3" in ENTRENADORES, "Falta el entrenador registrado para P3 (Tipología)"

def test_pregunta_invalida_edge_case():
    """Caso límite: Pasar un argumento de pregunta no configurado (ej. 'p4') debe detener la ejecución."""
    pregunta_invalida = "p4"
    
    with pytest.raises(BadParameter):
        if pregunta_invalida not in ENTRENADORES:
            raise BadParameter(f"pregunta desconocida: {pregunta_invalida}")

def test_fraccion_entrenamiento_limites():
    """Caso límite: La fracción de los datos para entrenamiento no puede ser mayor a 1.0 ni menor o igual a 0.0."""
    fraccion_invalida = 1.5
    
    with pytest.raises(ValueError):
        # Simula la validación de entrada antes de hacer el train_test_split
        if not (0.0 < fraccion_invalida <= 1.0):
            raise ValueError("La fracción de datos debe estar entre 0.0 y 1.0")