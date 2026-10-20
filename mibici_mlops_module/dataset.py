"""Ingesta reproducible de MiBici: descarga, normalización, Parquet y versionado.

python -m mibici_mlops_module.dataset                  # meses de config.MESES
python -m mibici_mlops_module.dataset --meses 2026_01 --meses 2026_02
python -m mibici_mlops_module.dataset --no-descargar   # usa los CSV de data/raw
python -m mibici_mlops_module.dataset --mlflow         # registra la versión en MLflow
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Annotated
import unicodedata
from urllib.request import Request, urlopen

from loguru import logger
import pandas as pd
import typer

from mibici_mlops_module.config import (
    DATOS_ABIERTOS_URL,
    DUR_MAX,
    DUR_MIN,
    EQUIPO,
    INTERIM_DATA_DIR,
    MANIFEST_PATH,
    MESES,
    MLFLOW_EXPERIMENT_EQUIPO,
    MLFLOW_TRACKING_URI,
    PORTAL_URL,
    RAW_DATA_DIR,
)
from mibici_mlops_module.utils import anidada, corrida_proyecto, info_git

app = typer.Typer()

# Columnas de viajes que se conservan (ya normalizadas). Las personales
# (usuario_id, genero, año_de_nacimiento) nunca se cargan.
RENOMBRAR_VIAJES = {
    "viaje_id": "viaje_id",
    "inicio_del_viaje": "inicio_viaje",
    "fin_del_viaje": "fin_viaje",
    "origen_id": "origen_id",
    "destino_id": "destino_id",
}
RENOMBRAR_NOMENCLATURA = {
    "id": "estacion_id",
    "name": "nombre",
    "latitude": "latitud",
    "longitude": "longitud",
    "dpcapacity": "capacidad",
}
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"
USER_AGENT = "Mozilla/5.0 (mibici_mlops_module; proyecto academico)"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def normalizar_nombre(col: str) -> str:
    """'Año_de_nacimiento' -> 'ano_de_nacimiento': sin acentos, minúsculas y snake_case."""
    sin_acentos = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9a-z]+", "_", sin_acentos.lower()).strip("_")


def sha256_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    """Hash SHA-256 de un archivo, leído por bloques.

    Por qué: es la huella con la que el manifiesto detecta si un CSV cambió, base de la versión de
    los datos.
    """
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def detectar_codificacion(ruta: Path) -> str:
    """UTF-8 si la cabecera lo es; si no, Latin-1 (enero de 2026 viene así)."""
    with open(ruta, "rb") as f:
        cabecera = f.readline()
    try:
        cabecera.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _abrir(url: str):
    """Abre una URL del portal con un `User-Agent` identificable y un tiempo máximo de espera."""
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MiBiciDataset:
    """Descarga, normaliza y limpia los CSV de MiBici."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, interim_dir: Path = INTERIM_DATA_DIR):
        """Define las carpetas de datos crudos (`raw_dir`) y normalizados (`interim_dir`)."""
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)

    # -- Descarga -----------------------------------------------------------
    @staticmethod
    def listar_enlaces() -> dict[str, str]:
        """Lee el portal y devuelve {nombre_archivo: url}. Los enlaces llevan un id
        numérico impredecible, por eso se extraen de la página."""
        html = _abrir(DATOS_ABIERTOS_URL).read().decode("utf-8", errors="ignore")
        patron = r"/site/assets/files/\d+/((?:datos_abiertos|nomenclatura)_\d{4}_\d{2})\.csv"
        return {f"{m.group(1)}.csv": PORTAL_URL + m.group(0) for m in re.finditer(patron, html)}

    def descargar(self, meses: list[str], force: bool = False) -> dict[str, str]:
        """Descarga los meses pedidos y la nomenclatura más reciente a data/raw.
        Devuelve {nombre_archivo: url}. No vuelve a bajar lo que ya existe, salvo force."""
        enlaces = self.listar_enlaces()
        pedidos = [f"datos_abiertos_{mes}.csv" for mes in meses]
        nomenclaturas = sorted(n for n in enlaces if n.startswith("nomenclatura_"))
        if not nomenclaturas:
            raise RuntimeError("No se encontró la nomenclatura en el portal de MiBici")
        pedidos.append(nomenclaturas[-1])
        faltan = [n for n in pedidos if n not in enlaces]
        if faltan:
            raise FileNotFoundError(f"El portal no publica: {faltan}")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        urls = {}
        for nombre in pedidos:
            urls[nombre] = enlaces[nombre]
            destino = self.raw_dir / nombre
            if destino.exists() and not force:
                logger.info(f"Ya existe {nombre}, se omite la descarga")
                continue
            logger.info(f"Descargando {nombre}...")
            parcial = destino.with_suffix(".part")
            with _abrir(enlaces[nombre]) as r, open(parcial, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if parcial.stat().st_size == 0:
                raise OSError(f"Descarga vacía: {nombre}")
            parcial.replace(destino)
        return urls

    # -- Lectura y normalización ------------------------------------------
    def leer_viajes_mes(self, mes: str) -> tuple[pd.DataFrame, str]:
        """Lee un CSV mensual: detecta la codificación, normaliza los nombres de columna,
        convierte las fechas y descarta las columnas personales."""
        ruta = self.raw_dir / f"datos_abiertos_{mes}.csv"
        enc = detectar_codificacion(ruta)
        df = pd.read_csv(
            ruta, encoding=enc, usecols=lambda c: normalizar_nombre(c) in RENOMBRAR_VIAJES
        )
        df.columns = [normalizar_nombre(c) for c in df.columns]
        faltan = set(RENOMBRAR_VIAJES) - set(df.columns)
        if faltan:
            raise ValueError(f"{ruta.name}: faltan columnas {sorted(faltan)}")
        df = df.rename(columns=RENOMBRAR_VIAJES)[list(RENOMBRAR_VIAJES.values())]
        for c in ("inicio_viaje", "fin_viaje"):
            df[c] = pd.to_datetime(df[c], format=FORMATO_FECHA)
        for c in ("viaje_id", "origen_id", "destino_id"):
            df[c] = df[c].astype("int32")
        return df, enc

    def leer_nomenclatura(self) -> tuple[pd.DataFrame, str, str]:
        """Lee la nomenclatura de estaciones más reciente y normaliza sus columnas.

        Por qué: la nomenclatura es la lista de estaciones válidas y trae la capacidad (`dpcapacity`) de
        cada una, necesaria para definir el riesgo de P2.
        """
        archivos = sorted(self.raw_dir.glob("nomenclatura_*.csv"))
        if not archivos:
            raise FileNotFoundError(f"No hay nomenclatura en {self.raw_dir}")
        ruta = archivos[-1]
        enc = detectar_codificacion(ruta)
        cat = pd.read_csv(ruta, encoding=enc)
        cat.columns = [normalizar_nombre(c) for c in cat.columns]
        faltan = set(RENOMBRAR_NOMENCLATURA) - set(cat.columns)
        if faltan:
            raise ValueError(f"{ruta.name}: faltan columnas {sorted(faltan)}")
        cat = cat.rename(columns=RENOMBRAR_NOMENCLATURA)[list(RENOMBRAR_NOMENCLATURA.values())]
        cat["estacion_id"] = cat["estacion_id"].astype("int32")
        cat["capacidad"] = cat["capacidad"].astype("int32")
        return cat, enc, ruta.name

    # -- Limpieza ------------------------------------------------------------
    @staticmethod
    def limpiar(v: pd.DataFrame, estaciones_validas) -> tuple[pd.DataFrame, dict]:
        """Quita duplicados, nulos, viajes de menos de 1 min o más de 3 h y viajes
        de estaciones que no están en la nomenclatura. Devuelve los viajes y un resumen."""
        r = {"viajes_crudos": len(v)}
        r["duplicados"] = int(v["viaje_id"].duplicated().sum())
        r["nulos"] = int(v.isna().any(axis=1).sum())
        v = v.drop_duplicates("viaje_id").dropna()
        dur = (v["fin_viaje"] - v["inicio_viaje"]).dt.total_seconds() / 60
        r["menos_1_min"] = int((dur < DUR_MIN).sum())
        r["mas_180_min"] = int((dur > DUR_MAX).sum())
        v = v[(dur >= DUR_MIN) & (dur <= DUR_MAX)]
        fuera = ~v["origen_id"].isin(estaciones_validas) | ~v["destino_id"].isin(
            estaciones_validas
        )
        r["estaciones_fuera_catalogo"] = sorted(
            int(e) for e in (set(v["origen_id"]) | set(v["destino_id"])) - set(estaciones_validas)
        )
        r["viajes_estaciones_fuera"] = int(fuera.sum())
        v = v[~fuera]
        r["viajes_limpios"] = len(v)
        return v.reset_index(drop=True), r

    # -- Pipeline completo ---------------------------------------------------
    def build(
        self,
        meses: list[str],
        urls: dict[str, str] | None = None,
        manifest_path: Path = MANIFEST_PATH,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
        """Normaliza cada mes, guarda Parquet, limpia y escribe el manifiesto.
        Devuelve (viajes limpios, nomenclatura, resumen de limpieza, manifiesto)."""
        urls = urls or {}
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        raw, interim, partes = [], [], []

        for mes in meses:
            nombre = f"datos_abiertos_{mes}.csv"
            df, enc = self.leer_viajes_mes(mes)
            salida = self.interim_dir / f"viajes_{mes}.parquet"
            df.to_parquet(salida, index=False)
            partes.append(df)
            raw.append(_entrada(self.raw_dir / nombre, urls.get(nombre), enc, len(df)))
            interim.append(_entrada(salida, None, None, len(df)))

        cat, enc_cat, nombre_cat = self.leer_nomenclatura()
        salida_cat = self.interim_dir / "nomenclatura.parquet"
        cat.to_parquet(salida_cat, index=False)
        raw.append(_entrada(self.raw_dir / nombre_cat, urls.get(nombre_cat), enc_cat, len(cat)))
        interim.append(_entrada(salida_cat, None, None, len(cat)))

        viajes, resumen = self.limpiar(pd.concat(partes, ignore_index=True), cat["estacion_id"])
        salida_limpios = self.interim_dir / "viajes_limpios.parquet"
        viajes.to_parquet(salida_limpios, index=False)
        interim.append(_entrada(salida_limpios, None, None, len(viajes)))

        manifest = construir_manifest(raw, interim, meses)
        resumen["data_version"] = manifest["data_version"]
        guardar_json(manifest, manifest_path)
        guardar_json(resumen, self.interim_dir / "resumen_limpieza.json")
        return viajes, cat, resumen, manifest


# ---------------------------------------------------------------------------
# Manifiesto de versiones
# ---------------------------------------------------------------------------
def _entrada(ruta: Path, url: str | None, codificacion: str | None, registros: int) -> dict:
    """Describe un archivo para el manifiesto: nombre, tamaño, registros, hash y, si aplica, URL y codificación."""
    e = {
        "archivo": ruta.name,
        "bytes": ruta.stat().st_size,
        "registros": registros,
        "sha256": sha256_archivo(ruta),
    }
    if url:
        e["url"] = url
    if codificacion:
        e["codificacion"] = codificacion
    return e


def construir_manifest(raw: list[dict], interim: list[dict], meses: list[str]) -> dict:
    """La versión de los datos es el hash de los CSV crudos y de los parámetros de limpieza.
    No depende de los Parquet, cuyos bytes pueden cambiar con la versión de pyarrow."""
    parametros = {"duracion_min": DUR_MIN, "duracion_max": DUR_MAX, "meses": sorted(meses)}
    huella = json.dumps(
        {"raw": sorted((e["archivo"], e["sha256"]) for e in raw), "parametros": parametros},
        sort_keys=True,
    )
    return {
        "data_version": hashlib.sha256(huella.encode()).hexdigest()[:12],
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuente": DATOS_ABIERTOS_URL,
        "parametros_limpieza": parametros,
        "raw": raw,
        "interim": interim,
    }


def guardar_json(obj: dict, ruta: Path) -> None:
    """Escribe un diccionario como JSON legible (crea la carpeta si hace falta)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=int) + "\n", encoding="utf-8"
    )


def cargar_manifest(ruta: Path = MANIFEST_PATH) -> dict:
    """Lee `data/manifest.json`.

    Por qué: el entrenamiento toma de aquí la `data_version`, de modo que cada run de MLflow
    indique con qué datos se ejecutó.
    """
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def verificar_manifest(manifest: dict, raw_dir: Path = RAW_DATA_DIR) -> list[str]:
    """Compara los CSV de data/raw con el manifiesto. Devuelve los problemas encontrados."""
    problemas = []
    for e in manifest["raw"]:
        ruta = Path(raw_dir) / e["archivo"]
        if not ruta.exists():
            problemas.append(f"falta {e['archivo']}")
        elif sha256_archivo(ruta) != e["sha256"]:
            problemas.append(f"hash distinto en {e['archivo']}")
    return problemas


def registrar_en_mlflow(manifest: dict, resumen: dict | None = None) -> None:
    """Registra la versión de los datos en MLflow (en la corrida activa, si hay una)."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_EQUIPO)
    with mlflow.start_run(run_name=f"datos-{manifest['data_version']}", nested=anidada()):
        mlflow.set_tags(
            {
                "equipo": EQUIPO,
                "etapa": "ingesta",
                "data_version": manifest["data_version"],
                **info_git(),
            }
        )
        mlflow.log_params(
            {
                "data_version": manifest["data_version"],
                "meses": ",".join(manifest["parametros_limpieza"]["meses"]),
                "duracion_min": DUR_MIN,
                "duracion_max": DUR_MAX,
            }
        )
        if resumen:
            mlflow.log_metrics(
                {k: v for k, v in resumen.items() if isinstance(v, int) and k != "data_version"}
            )
        mlflow.log_artifact(str(MANIFEST_PATH))


@app.command()
def main(
    meses: Annotated[
        list[str], typer.Option("--meses", help="Meses a ingerir (AAAA_MM), repetible")
    ] = MESES,
    descargar: Annotated[
        bool, typer.Option("--descargar/--no-descargar", help="Descargar del portal")
    ] = True,
    force: Annotated[
        bool, typer.Option("--force", help="Volver a descargar aunque ya existan")
    ] = False,
    mlflow: Annotated[
        bool, typer.Option("--mlflow", help="Registrar la versión en MLflow")
    ] = False,
    raw_dir: Path = RAW_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    manifest_path: Path = MANIFEST_PATH,
):
    """Descarga, normaliza y limpia los datos, genera el manifiesto y, con `--mlflow`, registra la versión."""
    ds = MiBiciDataset(raw_dir, interim_dir)
    urls = ds.descargar(meses, force) if descargar else {}
    logger.info("Normalizando, limpiando y guardando en Parquet...")
    viajes, _, resumen, manifest = ds.build(meses, urls, manifest_path)
    if mlflow:
        with corrida_proyecto("Ingesta de datos"):
            registrar_en_mlflow(manifest, resumen)
    logger.success(
        f"Datos {manifest['data_version']}: {resumen['viajes_crudos']:,} -> "
        f"{len(viajes):,} viajes. Manifiesto en {manifest_path}"
    )


if __name__ == "__main__":
    app()
