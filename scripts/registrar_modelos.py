"""Registro de modelos ganadores de MiBici - Equipo 04.

Parte C: Model Registry.
Registra modelos existentes sin volver a entrenarlos.

Uso:
    python scripts/registrar_modelos.py --listar
    python scripts/registrar_modelos.py --pregunta p1
    python scripts/registrar_modelos.py --pregunta p3
"""

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient


MODELOS = {
    "p1": {
        "nombre_registry": "mibici-equipo04-p1-demanda",
        "modelo_esperado": "hist-gradient-boosting-poisson-base",
        "run_id": "ef9cd98fceee4b19ad8b50f74e465d97",
        "descripcion": (
            "Modelo de prediccion de demanda de bicicletas por "
            "estacion y hora. HistGradientBoosting Poisson."
        ),
    },
    "p2": {
        "nombre_registry": "mibici-equipo04-p2-riesgo",
        "modelo_esperado": "random-forest-class-weight-bosque-grande",
        "run_id": "a67a4eb903044e5fbaab292eaa3fc048",
        "descripcion": (
            "Modelo de clasificacion de riesgo de vaciado o "
            "saturacion de estaciones. Random Forest balanceado."
        ),
    },
    "p3": {
        "nombre_registry": "mibici-equipo04-p3-grupos",
        "modelo_esperado": "kmeans-k3-n_init10",
        "run_id": "95ae7930af9247ff8916df81ddc4961d",
        "descripcion": (
            "Modelo de agrupamiento de estaciones de MiBici "
            "en tres perfiles mediante K-Means."
        ),
    },
}


def registrar_modelo(pregunta):
    config = MODELOS[pregunta]

    if not config["run_id"]:
        print(
            f"[PENDIENTE] {pregunta.upper()}: "
            "no se ha localizado el run ganador."
        )
        return

    client = MlflowClient()
    run = client.get_run(config["run_id"])

    modelo_real = run.data.tags.get("modelo")

    if modelo_real != config["modelo_esperado"]:
        raise ValueError(
            f"Modelo incorrecto: esperado "
            f"{config['modelo_esperado']}, encontrado {modelo_real}"
        )

    if run.info.status != "FINISHED":
        raise ValueError("La corrida no finalizo correctamente.")

    modelos = client.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
    )

    modelos = [m for m in modelos if m.name == "modelo"]

    if len(modelos) != 1:
        raise ValueError(
            f"Se esperaban 1 modelo MLflow para {pregunta}, "
            f"pero se encontraron {len(modelos)}."
        )

    modelo_origen = modelos[0]

    registrado = mlflow.register_model(
        model_uri=f"models:/{modelo_origen.model_id}",
        name=config["nombre_registry"],
    )

    client.update_registered_model(
        name=config["nombre_registry"],
        description=config["descripcion"],
    )

    tags = {
        "question": pregunta.upper(),
        "source_run_id": run.info.run_id,
        "git_commit": run.data.tags.get("git_commit", "desconocido"),
        "data_version": run.data.params.get(
            "data_version", "desconocido"
        ),
        "modelo": modelo_real,
    }

    for clave, valor in tags.items():
        client.set_model_version_tag(
            name=config["nombre_registry"],
            version=registrado.version,
            key=clave,
            value=str(valor),
        )

    client.set_registered_model_alias(
        name=config["nombre_registry"],
        alias="champion",
        version=registrado.version,
    )

    print(
        f"[OK] {config['nombre_registry']} "
        f"version {registrado.version} -> champion"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pregunta",
        choices=["p1", "p2", "p3"],
    )
    parser.add_argument(
        "--listar",
        action="store_true",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )
    )

    if args.listar:
        for pregunta, config in MODELOS.items():
            print(
                pregunta.upper(),
                config["nombre_registry"],
                config["run_id"] or "PENDIENTE",
            )
        return

    if args.pregunta:
        registrar_modelo(args.pregunta)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()