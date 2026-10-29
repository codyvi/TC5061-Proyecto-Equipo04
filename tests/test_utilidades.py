"""Pruebas de utilidades de datos y de entrenamiento (normalización, hashes, hiperparámetros)."""

import hashlib

import pytest

from mibici_mlops_module.dataset import (
    detectar_codificacion,
    normalizar_nombre,
    sha256_archivo,
    verificar_manifest,
)
from mibici_mlops_module.modeling.train import etiqueta, nombre_y_params
from mibici_mlops_module.utils import parsear_hp


# ---- normalizar_nombre --------------------------------------------------------------------
@pytest.mark.parametrize(
    "crudo,esperado",
    [
        ("Año_de_nacimiento", "ano_de_nacimiento"),
        ("Viaje_Id", "viaje_id"),
        ("  Origen Id  ", "origen_id"),
        ("Fecha-Hora (UTC)", "fecha_hora_utc"),
        ("", ""),  # límite: cadena vacía
        ("___", ""),  # límite: solo separadores
    ],
)
def test_normalizar_nombre(crudo, esperado):
    assert normalizar_nombre(crudo) == esperado


def test_normalizar_nombre_es_idempotente():
    una_vez = normalizar_nombre("Año de Nacimiento")
    assert normalizar_nombre(una_vez) == una_vez


# ---- sha256_archivo -----------------------------------------------------------------------
def test_sha256_coincide_con_hashlib(tmp_path):
    ruta = tmp_path / "a.csv"
    ruta.write_bytes(b"abc")
    assert sha256_archivo(ruta) == hashlib.sha256(b"abc").hexdigest()


def test_sha256_archivo_vacio(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_bytes(b"")
    assert sha256_archivo(ruta) == hashlib.sha256(b"").hexdigest()


def test_sha256_por_bloques_igual_que_de_una_vez(tmp_path):
    """Límite: el contenido es mayor que el bloque, así que se lee en varias vueltas."""
    contenido = bytes(range(256)) * 10
    ruta = tmp_path / "grande.bin"
    ruta.write_bytes(contenido)
    assert sha256_archivo(ruta, bloque=7) == hashlib.sha256(contenido).hexdigest()


# ---- detectar_codificacion ----------------------------------------------------------------
def test_codificacion_utf8(tmp_path):
    ruta = tmp_path / "u.csv"
    ruta.write_text("Año_de_nacimiento\n", encoding="utf-8")
    assert detectar_codificacion(ruta) == "utf-8"


def test_codificacion_latin1_cuando_la_cabecera_no_es_utf8(tmp_path):
    ruta = tmp_path / "l.csv"
    ruta.write_bytes("Año_de_nacimiento\n".encode("latin-1"))
    assert detectar_codificacion(ruta) == "latin-1"


def test_codificacion_archivo_vacio_es_utf8(tmp_path):
    ruta = tmp_path / "v.csv"
    ruta.write_bytes(b"")
    assert detectar_codificacion(ruta) == "utf-8"


# ---- verificar_manifest -------------------------------------------------------------------
def _manifest_de(ruta):
    return {"raw": [{"archivo": ruta.name, "sha256": sha256_archivo(ruta)}]}


def test_manifest_sin_problemas(tmp_path):
    ruta = tmp_path / "m.csv"
    ruta.write_text("a,b\n1,2\n")
    assert verificar_manifest(_manifest_de(ruta), tmp_path) == []


def test_manifest_detecta_archivo_modificado(tmp_path):
    ruta = tmp_path / "m.csv"
    ruta.write_text("a,b\n1,2\n")
    manifest = _manifest_de(ruta)
    ruta.write_text("a,b\n1,3\n")
    assert verificar_manifest(manifest, tmp_path) == ["hash distinto en m.csv"]


def test_manifest_detecta_archivo_faltante(tmp_path):
    ruta = tmp_path / "m.csv"
    ruta.write_text("a,b\n")
    manifest = _manifest_de(ruta)
    ruta.unlink()
    assert verificar_manifest(manifest, tmp_path) == ["falta m.csv"]


# ---- parsear_hp / etiqueta / nombre_y_params ----------------------------------------------
def test_parsear_hp_convierte_tipos():
    hp = parsear_hp(["n_estimators=300", "learning_rate=0.05", "early=true", "modo=rapido"])
    assert hp == {"n_estimators": 300, "learning_rate": 0.05, "early": True, "modo": "rapido"}


def test_parsear_hp_omite_valores_vacios_y_pares_sin_igual():
    """Límite: `mlflow run` pasa todos los parámetros del MLproject, también los vacíos."""
    assert parsear_hp(["n_estimators=300", "max_depth=", "sin_igual", "x=   "]) == {
        "n_estimators": 300
    }


def test_parsear_hp_lista_vacia():
    assert parsear_hp([]) == {}


def test_etiqueta_abrevia_y_conserva_orden():
    assert etiqueta({"n_estimators": 100, "max_depth": 14, "otro": 1}) == "n100-prof14-otro1"


def test_nombre_y_params_usa_nombre_del_yaml():
    nombre, params = nombre_y_params({"nombre": "base", "n_estimators": 100})
    assert (nombre, params) == ("base", {"n_estimators": 100})


def test_nombre_y_params_deriva_nombre_si_falta():
    nombre, params = nombre_y_params({"max_depth": 6})
    assert (nombre, params) == ("prof6", {"max_depth": 6})


# ---- _abrir -------------------------------------------------------------------------------
@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/a.csv", "http://host/a.csv", "/etc/passwd"])
def test_abrir_rechaza_esquemas_distintos_de_https(url):
    from mibici_mlops_module.dataset import _abrir

    with pytest.raises(ValueError, match="https"):
        _abrir(url)
