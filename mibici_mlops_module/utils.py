"""Utilidades compartidas entre la ingesta y el entrenamiento."""

import subprocess

from mibici_mlops_module.config import PROJ_ROOT


def info_git() -> dict[str, str]:
    """Devuelve el commit de Git y si hay cambios sin confirmar.

    Por qué: un run solo se puede reproducir si se sabe con qué código corrió. `git_dirty=true`
    avisa que el commit por sí solo no basta para reproducirlo.
    """

    def git(*args: str) -> str:
        """Ejecuta `git <args>` en la raíz del repositorio y devuelve su salida (vacía si falla)."""
        return subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD") or "desconocido",
        "git_dirty": str(bool(git("status", "--porcelain"))).lower(),
    }
