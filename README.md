<div align="center">

**MLOps Team 04 - MiBici**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![Cookiecutter](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter&logoColor=white)](https://cookiecutter-data-science.drivendata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

Proyecto MiBici - Predicción de demanda, clasificación de riesgos y clustering de estaciones con MLflow.

## ML Canvas y arquitectura

- [ML Canvas](references/ml_canvas_mibici.png)
- [Diagrama de arquitectura](references/arquitectura_mibici.png)

## Organización del proyecto

```
├── LICENSE            <- Licencia de código abierto
├── Makefile           <- Comandos de conveniencia, como `make data` o `make train`
├── README.md          <- README principal para quien use este proyecto
├── data
│   ├── external       <- Datos de fuentes externas
│   ├── interim        <- Datos intermedios ya transformados (viajes_2026_01..06.parquet,
│   │                     nomenclatura.parquet, viajes_limpios.parquet
│   │                     y resumen_limpieza.json).
│   ├── processed      <- Conjuntos finales para modelar (p1_demanda.parquet,
│   │                     p2_riesgo.parquet, p3_perfiles.parquet y resumen.json).
│   ├── raw            <- Datos originales e inmutables (datos_abiertos_2026_01..06.csv y
│   │                     nomenclatura_2026_09.csv de MiBici).
│   └── manifest.json  <- Versionado de datos (sha256, tamaño, registros y data_version);
│                         único archivo de data/ que se sube a Git.
│
├── configs            <- Hiperparámetros de los modelos (hiperparametros.yaml)
│
├── docs               <- Proyecto mkdocs por defecto; ver www.mkdocs.org
│
├── models             <- Modelos entrenados (p1.joblib, p2.joblib, p3.joblib; no se versionan)
│
├── notebooks          <- Jupyter notebooks. Convención de nombre: un número (para ordenar),
│                         las iniciales de quien lo crea y una descripción corta separada
│                         por `-`, p. ej. `1.0-jqp-exploracion-inicial`.
│   └── 1.0-dcm-eda-mibici.ipynb  <- EDA de viajes, P1, P2 y P3
│
├── pyproject.toml     <- Metadatos del paquete mibici_mlops_module y
│                         configuración de herramientas como ruff
│
├── references         <- Material explicativo del proyecto.
│   ├── ml_canvas_mibici.png     <- ML Canvas
│   └── arquitectura_mibici.png  <- Diagrama de arquitectura
│
├── reports            <- Análisis generados en HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficas y figuras generadas para los reportes
│
├── requirements.txt   <- Dependencias con versión fija para reproducir el entorno
│                         (generado con `pip freeze`)
│
├── tests              <- Pruebas automatizadas (pytest)
│
└── mibici_mlops_module   <- Código fuente del proyecto
    │
    ├── __init__.py             <- Convierte mibici_mlops_module en un módulo de Python
    │
    ├── config.py               <- Variables y configuración central
    │
    ├── dataset.py              <- Scripts para descargar o generar los datos
    │
    ├── features.py             <- Código para crear las variables del modelado
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── baseline.py         <- Baselines propios y guardado/carga de modelos (joblib)
    │   ├── pipeline.py         <- Entrena y evalúa de punta a punta, registrando ambos en MLflow
    │   ├── predict.py          <- Predictores OOP (P1Predictor, P2Predictor, P3Predictor)
    │   └── train.py            <- Entrenadores OOP (P1Trainer, P2Trainer, P3Trainer) con MLflow
    │
    ├── plots.py                <- Código para crear visualizaciones
    │
    └── utils.py                <- Utilidades compartidas (commit de Git para las etiquetas de MLflow)
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
# 1. Clonar el repositorio y entrar a la carpeta
git clone <url-del-repositorio>
cd TC5061-Proyecto-Equipo04

# 2. Crear entorno virtual y activarlo
python3 -m venv mibici
source mibici/bin/activate

# 3. Instalar dependencias
pip install -U pip
pip install -r requirements.txt

# 4. Datos: descarga los CSV, los limpia, genera data/manifest.json y registra la versión en MLflow
python -m mibici_mlops_module.dataset --mlflow

# 5. Variables: construye las tablas P1, P2 y P3 en data/processed/
python -m mibici_mlops_module.features

# 6. Entrenamiento + evaluación de P1, P2 y P3 (registra todo en MLflow)
python -m mibici_mlops_module.modeling.pipeline

# 7. Ver los resultados
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Después abre <http://127.0.0.1:5000> y entra al experimento **mibici-equipo04**.

Notas:

- El paso 6 registra 9 runs con la configuración actual (una combinación por modelo, más los baselines): 4 de P1, 3 de P2 y 2 de P3. Para probar más combinaciones, quita el `#` a las que están comentadas en `configs/hiperparametros.yaml`; con todas son más de 20 runs y tarda varias decenas de minutos. Para una prueba rápida usa `--fraccion 0.1` (10 % del entrenamiento).
- Conviene hacer `git commit` antes del paso 4: cada run guarda el commit con el que se ejecutó (`git_commit`) y marca `git_dirty=true` si hay cambios sin confirmar.
- Corre `mlflow ui` desde la raíz del repositorio: `mlflow.db` se crea ahí y la ruta `sqlite:///mlflow.db` es relativa.

**Qué queda en MLflow** (experimento `mibici-equipo04`):

- Un run de datos, `datos-<data_version>` (etapa `ingesta`), con la versión de los datos, el commit de Git, los meses, los criterios de limpieza, los conteos y `manifest.json`.
- Un run por candidato, con nombre descriptivo (`P1-demanda-random-forest-base`, `P3-tipologia-kmeans-k4`, etc.). Cada run guarda parámetros (fechas de cada partición, `data_version`, hiperparámetros), métricas, figuras, el modelo con su firma y las etiquetas `equipo`, `pregunta`, `git_commit` y `git_dirty`.
- Un run de evaluación por pregunta (`P1-demanda-evaluacion-<modelo>`): carga el mejor modelo desde `models/` con el predictor y registra sus métricas en validación y prueba, las figuras y `max_dif_vs_entrenamiento` (debe ser 0: el modelo guardado reproduce lo entrenado).
- El mejor candidato de cada pregunta, según la validación, queda en `models/pX.joblib`.

## Flujo de trabajo con Git y GitHub

La rama `main` está protegida por un ruleset de GitHub: nadie sube directo, no se permite reescribir su historial ni borrarla, y todo cambio entra por *Pull Request* con al menos una aprobación. El único método de integración permitido es **rebase**, por lo que el historial de `main` es lineal.

1. **Actualizar `main`:** `git switch main && git pull`.
2. **Crear una rama** por cambio, con prefijo según el tipo: `feat/<tema>` (funcionalidad), `fix/<tema>` (corrección) o `docs/<tema>` (documentación). Ejemplo: `git switch -c feat/refactor`.
3. **Commits** con el formato `tipo(alcance): descripción en español`, donde el tipo es `feat`, `fix`, `docs`, `test` o `refactor`. Ejemplo: `feat(modeling): entrenamiento y evaluación de P1, P2 y P3 con MLflow`.
4. **Antes de abrir el PR:** `python -m pytest -q` y `ruff check`. No se suben datos, modelos, `mlflow.db`, `mlruns/` ni `.env`: ya están en `.gitignore`.
5. **Abrir el Pull Request** hacia `main` con un resumen de qué cambia y por qué. Debe revisarlo y aprobarlo al menos otra persona del equipo; quien lo abre no lo aprueba.
6. **Integrar con *Rebase and merge*** y borrar la rama.

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
