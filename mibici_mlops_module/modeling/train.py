import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import typer
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Importamos las rutas y variables que tu equipo ya definió
from mibici_mlops_module.config import PROCESSED_DATA_DIR
from mibici_mlops_module.features import FEATURES_P1, OBJETIVO_P1

app = typer.Typer()

def obtener_commit():
    """Obtiene el hash del commit actual de Git para las etiquetas de MLFlow."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "desconocido"

@app.command()
def main(
    p1_path: Path = PROCESSED_DATA_DIR / "p1_demanda.parquet",
    manifest_path: Path = Path("data/manifest.json")
):
    logger.info("Cargando datos para baseline P1...")
    df_p1 = pd.read_parquet(p1_path)
    
    # Leer versión de datos del manifest que hizo tu compañero
    data_version = "unknown"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_version = manifest.get("data_version", "unknown")

    # Separar conjuntos usando la columna que tu equipo ya construyó
    df_train = df_p1[df_p1["conjunto"] == "entrenamiento"]
    df_test = df_p1[df_p1["conjunto"] == "validacion"]

    X_train = df_train[FEATURES_P1]
    y_train = df_train[OBJETIVO_P1]
    X_test = df_test[FEATURES_P1]
    y_test = df_test[OBJETIVO_P1]

    logger.info(f"Entrenando baseline P1 con {len(X_train)} registros...")
    
    # Configurar MLflow (puedes cambiar "equipo04" por el de tu equipo si lo requieren)
    mlflow.set_experiment("mibici-equipo04")
    
    with mlflow.start_run(run_name="P1_baseline_regresion"):
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        predicciones = modelo.predict(X_test)
        test_mae = mean_absolute_error(y_test, predicciones)
        
        # 1. Registrar Parámetros
        mlflow.log_param("modelo", "LinearRegression")
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("features", ", ".join(FEATURES_P1))
        
        # 2. Registrar Métricas
        mlflow.log_metric("validacion_mae", test_mae)
        
        # 3. Registrar Etiquetas
        mlflow.set_tag("pregunta", "P1")
        mlflow.set_tag("equipo", "Equipo04")
        mlflow.set_tag("commit", obtener_commit())
        
        # 4. Registrar Artefacto (el modelo)
        mlflow.sklearn.log_model(modelo, "modelo_baseline_p1")
        
    logger.success(f"Entrenamiento completado y registrado en MLflow. MAE de validación: {test_mae:.4f}")

if __name__ == "__main__":
    app()