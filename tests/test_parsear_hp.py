"""Dos pruebas unitarias de `parsear_hp`, distintas de las de tests/test_utilidades.py."""

from mibici_mlops_module.utils import parsear_hp


def test_parsear_hp_interpreta_listas_diccionarios_falso_y_nulo():
    """Estructuras YAML: lista, mapa, booleano falso y nulo."""
    hp = parsear_hp(
        [
            "capas=[64, 32]",
            "pesos={a: 1, b: 2}",
            "activo=false",
            "opcional=null",
        ]
    )
    assert hp == {
        "capas": [64, 32],
        "pesos": {"a": 1, "b": 2},
        "activo": False,
        "opcional": None,
    }


def test_parsear_hp_clave_repetida_queda_con_el_ultimo_valor():
    """La clave repetida conserva el último valor y un '=' extra forma parte del valor."""
    hp = parsear_hp(["  max_depth = 4", "max_depth=12", "nota=a=b"])
    assert hp == {"max_depth": 12, "nota": "a=b"}
