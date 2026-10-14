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
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
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
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
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

**Versionado de los datos.** `data/manifest.json` guarda, por cada archivo, su `sha256`, tamaño y número de registros, y una `data_version` (hash de los CSV crudos y de los parámetros de limpieza). Es el único archivo de `data/` que se sube a Git: si cambia un CSV, cambia la versión. Con `--mlflow` la versión queda como parámetro de una corrida en el experimento `mibici`; en el entrenamiento se puede leer con `cargar_manifest()["data_version"]`.

El resto de `data/` está en `.gitignore`: los datos no se redistribuyen y se usan solo con fines académicos, citando la fuente. Después de generar las tablas, abre `notebooks/1.0-dcm-eda-mibici.ipynb` para el análisis exploratorio.

--------
