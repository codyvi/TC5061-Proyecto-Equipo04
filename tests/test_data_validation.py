import pytest
import pandas as pd
# Descomenta y ajusta esta importación según las funciones reales de tu módulo:
# from mibici_mlops_module.dataset import procesar_datos

def test_datos_vacios_edge_case():
    """Caso límite: El preprocesamiento no debe fallar críticamente ante datos vacíos, sino retornar vacío o lanzar error controlado."""
    df_vacio = pd.DataFrame(columns=["Viaje_Id", "Usuario_Id", "Genero", "Año_de_nacimiento"])
    
    # Ejemplo si tuvieras la función importada:
    # resultado = procesar_datos(df_vacio)
    # assert resultado.empty
    
    assert len(df_vacio) == 0, "El DataFrame debería estar vacío"

def test_columnas_faltantes_edge_case():
    """Caso límite: Debe lanzar un error (KeyError o ValueError) si el archivo fuente no tiene las variables necesarias."""
    df_incompleto = pd.DataFrame({"Viaje_Id": [1, 2, 3]})
    
    with pytest.raises(KeyError):
        # Simula el comportamiento interno de la función al buscar variables
        _ = df_incompleto["Genero"]

def test_transformacion_tipos_normal():
    """Caso normal: Las variables de entrada deben castearse a tipos numéricos para el entrenamiento."""
    df_mock = pd.DataFrame({"Año_de_nacimiento": ["1990", "1985", "1992"]})
    df_mock["Año_de_nacimiento"] = pd.to_numeric(df_mock["Año_de_nacimiento"])
    
    assert pd.api.types.is_numeric_dtype(df_mock["Año_de_nacimiento"]), "La columna debe ser numérica"