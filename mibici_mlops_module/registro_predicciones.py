"""Registro de las predicciones que sirve la API (monitoreo en producción).

Hasta ahora la API no guardaba las peticiones, así que el drift solo se podía medir contra junio
(un periodo que *simula* producción). Este módulo guarda cada predicción servida en un archivo
JSON Lines (una línea por petición) con:

    fecha_utc, pregunta, entradas (las variables del modelo), prediccion, modelo (alias o
    versión del Registry), run_id del modelo cargado y latencia_ms.

Con ese registro se puede: (1) resumir el tráfico real en `GET /monitoreo/resumen` y (2) correr el
reporte de Evidently tomando el tráfico real como conjunto "actual" (ver `monitoreo.py`).

Variables de entorno (las lee la API al arrancar):
    REGISTRO_PREDICCIONES       1/true/si para activarlo. Apagado por defecto: la API se comporta
                                igual que antes y no escribe nada.
    REGISTRO_PREDICCIONES_RUTA  archivo JSONL (por defecto data/interim/predicciones_api.jsonl,
                                que ya está en .gitignore).

No se registran datos de usuarios: las entradas son variables agregadas por estación y hora.

    python -m mibici_mlops_module.registro_predicciones resumen
    python -m mibici_mlops_module.registro_predicciones drift --pregunta p1
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_mlops_module.config import INTERIM_DATA_DIR

ENV_ACTIVO = "REGISTRO_PREDICCIONES"
ENV_RUTA = "REGISTRO_PREDICCIONES_RUTA"
RUTA_POR_DEFECTO = INTERIM_DATA_DIR / "predicciones_api.jsonl"
VALORES_VERDADEROS = {"1", "true", "si", "sí", "yes", "on"}
VENTANA_POR_DEFECTO = 10_000  # últimas peticiones que se consideran en el resumen

app = typer.Typer(help="Registro de predicciones de la API y drift sobre el tráfico real.")


def registro_activo() -> bool:
    """Indica si la variable de entorno pide registrar las predicciones."""
    return os.getenv(ENV_ACTIVO, "").strip().lower() in VALORES_VERDADEROS


def ruta_registro() -> Path:
    """Ruta del archivo JSONL (variable de entorno o la ruta por defecto en data/interim/)."""
    return Path(os.getenv(ENV_RUTA) or RUTA_POR_DEFECTO)


def _a_json(valor: Any) -> Any:
    """Convierte tipos de numpy a tipos nativos para poder serializarlos."""
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


class RegistroPredicciones:
    """Guarda y lee las predicciones servidas por la API en un archivo JSON Lines.

    Es seguro entre hilos (FastAPI corre los endpoints síncronos en un pool de hilos): cada
    escritura es una sola línea agregada bajo un candado.
    """

    def __init__(self, ruta: Path | str, activo: bool = True):
        self.ruta = Path(ruta)
        self.activo = activo
        self._candado = threading.Lock()

    @classmethod
    def desde_entorno(cls) -> "RegistroPredicciones":
        return cls(ruta_registro(), activo=registro_activo())

    # ------------------------------------------------------------------ escritura
    def registrar(
        self,
        pregunta: str,
        entradas: dict[str, Any],
        prediccion: Any,
        modelo: str | None = None,
        run_id: str | None = None,
        latencia_ms: float | None = None,
    ) -> dict[str, Any] | None:
        """Agrega una predicción al registro. Si está apagado no hace nada y devuelve None."""
        if not self.activo:
            return None
        fila = {
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "pregunta": pregunta,
            "entradas": {k: _a_json(v) for k, v in entradas.items()},
            "prediccion": _a_json(prediccion),
            "modelo": modelo,
            "run_id": run_id,
            "latencia_ms": None if latencia_ms is None else round(float(latencia_ms), 3),
        }
        linea = json.dumps(fila, ensure_ascii=False)
        with self._candado:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            with self.ruta.open("a", encoding="utf-8") as f:
                f.write(linea + "\n")
        return fila

    # ------------------------------------------------------------------ lectura
    def leer(self, pregunta: str | None = None, ventana: int | None = None) -> pd.DataFrame:
        """Devuelve el registro como tabla: una fila por petición y una columna por variable.

        Las entradas se expanden a columnas (las mismas de FEATURES_P1/P2). Las líneas dañadas
        (por ejemplo, una escritura cortada) se omiten con un aviso en lugar de fallar.
        """
        if not self.ruta.exists():
            return pd.DataFrame()
        filas, dañadas = [], 0
        with self._candado, self.ruta.open(encoding="utf-8") as f:
            for linea in f:
                if not linea.strip():
                    continue
                try:
                    fila = json.loads(linea)
                except json.JSONDecodeError:
                    dañadas += 1
                    continue
                if pregunta is None or fila.get("pregunta") == pregunta:
                    filas.append(fila)
        if dañadas:
            logger.warning(f"Se omitieron {dañadas} líneas dañadas de {self.ruta}")
        if ventana:
            filas = filas[-ventana:]
        if not filas:
            return pd.DataFrame()
        entradas = pd.json_normalize([f.get("entradas") or {} for f in filas])
        meta = pd.DataFrame([{k: v for k, v in f.items() if k != "entradas"} for f in filas])
        df = pd.concat([meta, entradas], axis=1)
        df["fecha_utc"] = pd.to_datetime(df["fecha_utc"], utc=True, format="ISO8601")
        return df

    def resumen(self, ventana: int | None = VENTANA_POR_DEFECTO) -> dict[str, Any]:
        """Resumen del tráfico: peticiones, periodo, latencia y distribución de predicciones."""
        df = self.leer(ventana=ventana)
        salida: dict[str, Any] = {
            "activo": self.activo,
            "peticiones": int(len(df)),
            "ventana": ventana,
            "por_pregunta": {},
        }
        if df.empty:
            return salida
        for pregunta, grupo in df.groupby("pregunta", sort=True):
            info: dict[str, Any] = {
                "peticiones": int(len(grupo)),
                "desde": grupo["fecha_utc"].min().isoformat(),
                "hasta": grupo["fecha_utc"].max().isoformat(),
                "modelos": sorted(grupo["modelo"].dropna().astype(str).unique().tolist()),
            }
            latencia = pd.to_numeric(grupo["latencia_ms"], errors="coerce").dropna()
            if not latencia.empty:
                info["latencia_ms_p50"] = round(float(latencia.quantile(0.50)), 3)
                info["latencia_ms_p95"] = round(float(latencia.quantile(0.95)), 3)
            pred = grupo["prediccion"]
            numerica = pd.to_numeric(pred, errors="coerce")
            if numerica.notna().all():  # regresión (P1)
                info["prediccion_media"] = round(float(numerica.mean()), 4)
                info["prediccion_p95"] = round(float(numerica.quantile(0.95)), 4)
            else:  # clasificación (P2): proporción de cada clase
                proporcion = pred.astype(str).value_counts(normalize=True).sort_index()
                info["distribucion_prediccion"] = {
                    k: round(float(v), 4) for k, v in proporcion.items()
                }
            salida["por_pregunta"][str(pregunta)] = info
        return salida


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------
@app.command()
def resumen(
    ruta: Path = typer.Option(
        None, help="Archivo JSONL (por defecto el de la variable de entorno)"
    ),
    ventana: int = typer.Option(VENTANA_POR_DEFECTO, help="Últimas peticiones a considerar"),
):
    """Imprime el resumen del tráfico registrado."""
    registro = RegistroPredicciones(ruta or ruta_registro())
    print(json.dumps(registro.resumen(ventana=ventana), indent=2, ensure_ascii=False))


@app.command()
def drift(
    pregunta: str = typer.Option("todas", help="p1, p2 o todas"),
    ruta: Path = typer.Option(
        None, help="Archivo JSONL (por defecto el de la variable de entorno)"
    ),
    minimo: int = typer.Option(50, help="Peticiones mínimas para comparar"),
):
    """Reporte de Evidently: entrenamiento (referencia) vs. tráfico real de la API (actual)."""
    from mibici_mlops_module.monitoreo import PREGUNTAS, generar_reporte_produccion

    preguntas = list(PREGUNTAS) if pregunta == "todas" else [pregunta]
    for p in preguntas:
        try:
            info = generar_reporte_produccion(p, ruta=ruta or ruta_registro(), minimo=minimo)
            logger.success(f"[{p}] {info}")
        except ValueError as exc:
            logger.warning(f"[{p}] {exc}")


if __name__ == "__main__":
    app()
