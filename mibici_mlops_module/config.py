import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Parámetros del proyecto MiBici
import pandas as pd

# Fuente de los datos
PORTAL_URL = "https://www.mibici.net"
DATOS_ABIERTOS_URL = f"{PORTAL_URL}/es/datos-abiertos/"
MESES = [f"2026_{m:02d}" for m in range(1, 7)]  # meses a ingerir (AAAA_MM)

# Versionado de datos y seguimiento
MANIFEST_PATH = DATA_DIR / "manifest.json"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
EQUIPO = "equipo04"
MLFLOW_EXPERIMENT_EQUIPO = f"mibici-{EQUIPO}"  # experimento único: datos, entrenamiento y evaluación

SEED = 42  # semilla única para muestreos y modelos
INICIO = pd.Timestamp("2026-01-01 00:00")
FIN = pd.Timestamp("2026-06-30 23:00")
DUR_MIN, DUR_MAX = 1, 180  # duración válida de un viaje, en minutos
U_RIESGO = 0.20  # umbral P2 sobre (llegadas - salidas) / capacidad
FIN_ENTRENAMIENTO = pd.Timestamp("2026-05-01")  # entrenamiento: hasta abril
FIN_VALIDACION = pd.Timestamp("2026-06-01")  # validación: mayo; prueba: junio
# Días festivos oficiales (LFT, art. 74) en el periodo
FESTIVOS = pd.to_datetime(["2026-01-01", "2026-02-02", "2026-03-16", "2026-05-01"])
# Semana Santa 2026 (vacaciones escolares, no festivo oficial)
SEMANA_SANTA = pd.date_range("2026-03-30", "2026-04-10")

MODELS_DIR = PROJ_ROOT / "models"
HIPERPARAMETROS_PATH = PROJ_ROOT / "configs" / "hiperparametros.yaml"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
