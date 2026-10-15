# mibici_mlops_equipo04

<div align="center">

**MLOps Team 04 - MiBici**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![Cookiecutter](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter&logoColor=white)](https://cookiecutter-data-science.drivendata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

Proyecto MiBici - Predicción de demanda, clasificación de riesgos y clustering de estaciones con MLflow.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed (viajes_2026_01..06.parquet,
│   │                     nomenclatura.parquet, viajes_limpios.parquet
│   │                     y resumen_limpieza.json).
│   ├── processed      <- The final, canonical data sets for modeling (p1_demanda.parquet,
│   │                     p2_riesgo.parquet, p3_perfiles.parquet y resumen.json).
│   ├── raw            <- The original, immutable data dump (datos_abiertos_2026_01..06.csv y
│   │                     nomenclatura_2026_09.csv de MiBici).
│   └── manifest.json  <- Versionado de datos (sha256, tamaño, registros y data_version);
│                         único archivo de data/ que se sube a Git.
│
├── configs            <- Hiperparámetros de los modelos (hiperparametros.yaml)
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Modelos entrenados (p1.joblib, p2.joblib, p3.joblib; no se versionan)
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│   └── 1.0-dcm-eda-mibici.ipynb  <- EDA de viajes, P1, P2 y P3
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_mlops_module and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── tests              <- Pruebas automatizadas (pytest)
│
└── mibici_mlops_module   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_mlops_module a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── estimadores.py      <- Baselines propios y guardado/carga de modelos (joblib)
    │   ├── pipeline.py         <- Entrena y evalúa de punta a punta, registrando ambos en MLflow
    │   ├── predict.py          <- Predictores OOP (P1Predictor, P2Predictor, P3Predictor)
    │   └── train.py            <- Entrenadores OOP (P1Trainer, P2Trainer, P3Trainer) con MLflow
    │
    └── plots.py                <- Code to create visualizations
```

## Entorno virtual

Desde la raíz del proyecto:

```bash
# Crear el entorno virtual "mibici"
python3 -m venv mibici

# Activarlo (WSL/Linux)
source mibici/bin/activate

# Instalar dependencias
pip install -U pip
pip install -r requirements.txt

# Registrarlo como kernel de Jupyter
python -m ipykernel install --user --name mibici --display-name "Python (mibici)"
```

Si `python3 -m venv` falla en WSL con un error de `ensurepip`, instala antes `sudo apt install python3-venv`.

**Uso diario**

- Activar: `source mibici/bin/activate` (el prompt muestra `(mibici)`).
- Desactivar: `deactivate`.
- Notebooks: elige el kernel **Python (mibici)** en VS Code o en `jupyter lab`.
- Nueva dependencia: `pip install paquete` y agrégala a `requirements.txt`.
- Si VS Code no detecta el entorno, usa *Python: Select Interpreter* y apunta a `mibici/bin/python`.


## Datos

Los datos vienen de [MiBici datos abiertos](https://www.mibici.net/es/datos-abiertos/) (enero a junio de 2026 y la nomenclatura de estaciones). El pipeline es reproducible desde la terminal, sin pasos manuales:

```bash
python -m mibici_mlops_module.dataset            # descarga, normaliza, limpia y versiona
python -m mibici_mlops_module.dataset --mlflow   # además registra la versión en MLflow
python -m mibici_mlops_module.features           # construye las tablas P1, P2 y P3
```

Opciones útiles de `dataset`: `--meses 2026_01 --meses 2026_02` (otros meses), `--no-descargar` (usa los CSV que ya estén en `data/raw/`) y `--force` (vuelve a descargar).

| Etapa | Carpeta | Contenido |
| --- | --- | --- |
| Crudo | `data/raw/` | CSV originales, inmutables (los de enero vienen en Latin-1) |
| Normalizado | `data/interim/` | Parquet por mes, `nomenclatura.parquet` y `viajes_limpios.parquet`, con columnas en snake_case y fechas como `datetime` |
| Modelado | `data/processed/` | `p1_demanda`, `p2_riesgo`, `p3_perfiles` y `resumen.json` |

**Versionado de los datos.** `data/manifest.json` guarda, por cada archivo, su `sha256`, tamaño y número de registros, y una `data_version` (hash de los CSV crudos y de los parámetros de limpieza). Es el único archivo de `data/` que se sube a Git: si cambia un CSV, cambia la versión. Con `--mlflow` la versión queda como parámetro de una corrida en el experimento `mibici-equipo04`; en el entrenamiento se puede leer con `cargar_manifest()["data_version"]`.

El resto de `data/` está en `.gitignore`: los datos no se redistribuyen y se usan solo con fines académicos, citando la fuente. Después de generar las tablas, abre `notebooks/1.0-dcm-eda-mibici.ipynb` para el análisis exploratorio.

## Ejecución de principio a fin

Pasos para reproducir todo desde cero, hasta ver los resultados en MLflow. Se ejecutan desde la raíz del repositorio en WSL/Linux.

```bash
# 1. Entorno virtual e instalación (ver "Entorno virtual" arriba)
python3 -m venv mibici
source mibici/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2. Datos: descarga los CSV, los limpia, genera data/manifest.json y registra la versión en MLflow
python -m mibici_mlops_module.dataset --mlflow

# 3. Variables: construye las tablas P1, P2 y P3 en data/processed/
python -m mibici_mlops_module.features

# 4. Entrenamiento + evaluación de P1, P2 y P3 (registra todo en MLflow)
python -m mibici_mlops_module.modeling.pipeline

# 5. Ver los resultados
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Después abre <http://127.0.0.1:5000> y entra al experimento **mibici-equipo04**.

Notas:

- El paso 4 entrena 29 modelos (5 combinaciones de hiperparámetros por modelo, más los baselines) y tarda varias decenas de minutos. Para una prueba rápida usa `--fraccion 0.1` (10 % del entrenamiento).
- Conviene hacer `git commit` antes del paso 4: cada run guarda el commit con el que se ejecutó (`git_commit`) y marca `git_dirty=true` si hay cambios sin confirmar.
- Corre `mlflow ui` desde la raíz del repositorio: `mlflow.db` se crea ahí y la ruta `sqlite:///mlflow.db` es relativa.

**Qué queda en MLflow** (experimento `mibici-equipo04`):

- Un run de datos, `datos-<data_version>` (etapa `ingesta`), con la versión de los datos, los meses, los criterios de limpieza, los conteos y `manifest.json`.
- Un run por candidato, con nombre descriptivo (`P1-demanda-random-forest-base`, `P3-tipologia-kmeans-k4`, etc.). Cada run guarda parámetros (fechas de cada partición, `data_version`, hiperparámetros), métricas, figuras, el modelo con su firma y las etiquetas `equipo`, `pregunta`, `git_commit` y `git_dirty`.
- Un run de evaluación por pregunta (`P1-demanda-evaluacion-<modelo>`): carga el mejor modelo desde `models/` con el predictor y registra sus métricas en validación y prueba, las figuras y `max_dif_vs_entrenamiento` (debe ser 0: el modelo guardado reproduce lo entrenado).
- El mejor candidato de cada pregunta, según la validación, queda en `models/pX.joblib`.

## Scripts disponibles

Todos se ejecutan como módulo (`python -m ...`) y aceptan `--help`.

| Script | Para qué sirve | Ejemplo |
| --- | --- | --- |
| `mibici_mlops_module.dataset` | Descarga, normaliza, limpia y versiona los datos (`data/manifest.json`). Opciones: `--meses`, `--no-descargar`, `--force`, `--mlflow` (registra la versión en el experimento `mibici-equipo04`). | `python -m mibici_mlops_module.dataset --mlflow` |
| `mibici_mlops_module.features` | Construye `p1_demanda`, `p2_riesgo`, `p3_perfiles` y `resumen.json`, y valida que no haya fuga de información. | `python -m mibici_mlops_module.features` |
| `mibici_mlops_module.modeling.pipeline` | Entrena y evalúa de punta a punta, registrando ambos pasos en MLflow. **Es el recomendado.** | `python -m mibici_mlops_module.modeling.pipeline --pregunta p1` |
| `mibici_mlops_module.modeling.train` | Solo entrena (un run por candidato) y guarda el mejor en `models/`. | `python -m mibici_mlops_module.modeling.train --pregunta p2` |
| `mibici_mlops_module.modeling.predict` | Genera predicciones con el modelo guardado y las escribe en `data/processed/pX_predicciones.parquet`. | `python -m mibici_mlops_module.modeling.predict --pregunta p1 --conjunto prueba` |
| `pytest` (carpeta `tests/`) | Pruebas automáticas de `features.py`: rezagos y medias móviles sin fuga del futuro, objetivos en t+1, clases de riesgo según el umbral, partición temporal y validación con muestras pequeñas. No necesitan los datos reales. | `python -m pytest -q` |

Opciones de `pipeline` y `train`:

- `--pregunta p1|p2|p3|todas` (por defecto `todas`).
- `--fraccion 0.1`: usa solo esa fracción del entrenamiento (pruebas rápidas).
- `--seed 42`: semilla de los modelos.
- `--config ruta.yaml`: hiperparámetros alternativos (por defecto `configs/hiperparametros.yaml`).

Opciones de `predict`: `--pregunta`, `--conjunto entrenamiento|validacion|prueba` (P1 y P2), `--entrada` y `--salida` (rutas Parquet).

**Hiperparámetros.** `configs/hiperparametros.yaml` define las combinaciones de cada modelo. Cada elemento de la lista es un run; el campo opcional `nombre` forma parte del nombre del run (`P1-demanda-random-forest-<nombre>`) y, si se omite, se arma con los hiperparámetros. Agrega o quita combinaciones y vuelve a correr.
