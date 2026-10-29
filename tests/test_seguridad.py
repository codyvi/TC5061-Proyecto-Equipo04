"""Chequeos de seguridad básicos: secretos fuera del repositorio y análisis estático.

El análisis de dependencias (pip-audit) necesita red y se corre con `make security`.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from mibici_mlops_module.config import PROJ_ROOT

if not (PROJ_ROOT / ".git").exists() or shutil.which("git") is None:
    pytest.skip("requiere un repositorio Git", allow_module_level=True)

# Nombres de archivo que casi siempre contienen credenciales.
NOMBRES_SENSIBLES = re.compile(
    r"(^|/)(\.env(\..*)?|id_rsa.*|credentials.*\.json|.*\.(pem|key|p12|pfx))$"
)
PERMITIDOS = {".env.example"}

# Formas conocidas de credenciales (llaves de nube, tokens de GitHub, llaves privadas, etc.).
PATRONES_SECRETOS = {
    "llave AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "token GitHub": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "token Slack": re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    "llave de API tipo sk-": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "llave privada": re.compile(r"-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "contraseña en URL": re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s:@]{3,}@[^/\s]+"),
    "asignación de secreto": re.compile(
        r"(?i)\b(password|passwd|secret|api[_-]?key|token)\b\s*[=:]\s*['\"][^'\"\s]{8,}['\"]"
    ),
}
MAX_BYTES = 2_000_000


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
    )


def _archivos_versionados():
    return [f for f in _git("ls-files").stdout.splitlines() if (PROJ_ROOT / f).is_file()]


def test_env_esta_en_gitignore():
    for ruta in (".env", ".env.local", "llave.pem", "id_rsa"):
        assert _git("check-ignore", "-q", ruta).returncode == 0, f"{ruta} debería estar ignorado"


def test_env_example_si_se_versiona():
    """La plantilla de variables sí se sube (sin valores reales)."""
    assert _git("check-ignore", "-q", ".env.example").returncode != 0


def test_no_hay_archivos_de_credenciales_versionados():
    sensibles = [
        f for f in _archivos_versionados() if NOMBRES_SENSIBLES.search(f) and f not in PERMITIDOS
    ]
    assert sensibles == []


def test_nunca_se_subio_un_env_al_historial():
    historial = _git("log", "--all", "--name-only", "--pretty=format:").stdout.splitlines()
    subidos = {f for f in historial if NOMBRES_SENSIBLES.search(f)} - PERMITIDOS
    assert subidos == set()


def test_no_hay_secretos_en_los_archivos_versionados():
    hallazgos = []
    for f in _archivos_versionados():
        ruta = PROJ_ROOT / f
        if ruta.stat().st_size > MAX_BYTES or f == "tests/test_seguridad.py":
            continue
        try:
            texto = ruta.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # binarios (parquet, joblib, imágenes)
            continue
        for nombre, patron in PATRONES_SECRETOS.items():
            if patron.search(texto):
                hallazgos.append(f"{f}: {nombre}")
    assert hallazgos == []


def test_detector_de_secretos_funciona():
    """Control negativo: cada patrón reconoce un ejemplo, así que un 'sin hallazgos' es creíble."""
    ejemplos = {
        "llave AWS": "AKIAABCDEFGHIJKLMNOP",
        "token GitHub": "ghp_" + "a" * 36,
        "llave privada": "-----BEGIN RSA PRIVATE KEY-----",
        "contraseña en URL": "postgresql://admin:hunter22@db.example.com/mlflow",
        "asignación de secreto": 'password = "SuperSecreta123"',
    }
    for nombre, ejemplo in ejemplos.items():
        assert PATRONES_SECRETOS[nombre].search(ejemplo), nombre


def test_config_lee_el_servidor_de_mlflow_del_entorno():
    """La configuración sensible llega por variable de entorno, no escrita en el código.

    Va en un subproceso: `config` quita el handler 0 de loguru al importarse y no se puede
    recargar dentro del proceso de pytest.
    """
    entorno = {**os.environ, "MLFLOW_TRACKING_URI": "http://mlflow.interno:5000"}
    r = subprocess.run(
        [sys.executable, "-c", "from mibici_mlops_module.config import MLFLOW_TRACKING_URI as u; print(u)"],
        cwd=PROJ_ROOT, env=entorno, capture_output=True, text=True, check=False,
    )
    assert r.stdout.strip().splitlines()[-1] == "http://mlflow.interno:5000"


def test_bandit_no_reporta_severidad_alta():
    """Análisis estático del paquete: ningún hallazgo de severidad alta."""
    pytest.importorskip("bandit")
    r = subprocess.run(
        [sys.executable, "-m", "bandit", "-q", "-c", "pyproject.toml", "-r", "mibici_mlops_module", "-f", "json"],
        cwd=PROJ_ROOT, capture_output=True, text=True, check=False,
    )
    altos = [
        f"{x['filename']}:{x['line_number']} {x['test_id']}"
        for x in json.loads(r.stdout)["results"]
        if x["issue_severity"] == "HIGH"
    ]
    assert altos == []
