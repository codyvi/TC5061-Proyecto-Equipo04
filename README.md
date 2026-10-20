<div align="center">

**MLOps Team 04 - MiBici**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![Cookiecutter](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter&logoColor=white)](https://cookiecutter-data-science.drivendata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Propósito

El principal costo operativo de un sistema de bicicleta pública son las estaciones vacías o llenas, que obligan a rebalancear con camionetas. Este proyecto construye, con datos abiertos de MiBici (Guadalajara), una base **reproducible** de experimentos de aprendizaje automático para apoyar esa operación: ingesta versionada de los datos, variables sin fuga de información, modelos con *baseline* y seguimiento completo de cada experimento en MLflow.

## Las tres preguntas

| Pregunta | Qué se predice o agrupa | Granularidad y horizonte | Quién la usa y qué decide |
| --- | --- | --- | --- |
| **P1. Demanda** (regresión) | Número de viajes que inician en cada estación | Estación × hora, siguiente hora (t+1) | Operaciones: cuántas bicicletas llevar a cada estación |
| **P2. Riesgo** (clasificación) | Clase `vaciado`, `normal` o `saturación`, según el índice de flujo (llegadas − salidas) / capacidad con umbral de ±0.20 | Estación × hora, siguiente hora (t+1) | Logística: qué estaciones visitar primero en la ruta |
| **P3. Tipología** (clustering) | Grupos de estaciones con patrones de uso similares | Estación, sin horizonte (perfil de enero a abril) | Planeación: qué estrategia aplicar por tipo de estación |

Los datos solo registran viajes, no el inventario real de bicicletas; por eso la clase de P2 es una aproximación al riesgo. Detalle en el [ML Canvas](references/ml_canvas_mibici.png).

## Integrantes y roles

| Integrante | Matrícula | Rol | Aportes |
| --- | --- | --- | --- |
| David Alonso Cantú Martínez (`codyvi`) | A00822455 | MLOps Engineer / Data Engineer | Estructura del repositorio (CCDS), ingesta y versionado de datos, variables, y entrenamiento y evaluación con MLflow |
| Martín Rogelio Cantú González | A01797809 | Software Engineer | Pruebas automáticas (pytest) y refactorización del código |
| Luis Alfredo Ojeda Benito | A01750598 | Data Scientist | Entrenamiento y registro en MLflow de P1 (demanda) |
| Carlo Eugenio Leviaguirre Chavez | A01022264 | Machine Learning Engineer | Ajuste de hiperparámetros y entrenamiento de modelos |

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

## Servidor de tracking y Model Registry

El Model Registry requiere un almacenamiento respaldado por base de datos, así que todas las corridas se registran en un servidor local con SQLite. `mlflow.db` está en `.gitignore`.

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

`config.py` lee `MLFLOW_TRACKING_URI` (también se puede poner en un `.env`); si no está definida usa `sqlite:///<raíz>/mlflow.db` directamente.

## MLflow Project

Todas las corridas se lanzan con `mlflow run` desde el proyecto definido en [`MLproject`](MLproject), que crea un entorno virtual con las versiones fijas de [`python_env.yaml`](python_env.yaml) (las mismas de `requirements.txt` que usa el código; la primera vez tarda en instalarlo). Se usa `--experiment-name mibici-equipo04` para que la corrida del proyecto y las anidadas queden en el mismo experimento.

**Requisito:** `mlflow run` instala por su cuenta el Python de `python_env.yaml` (3.12), sin importar el que tengas en tu equipo. Para eso necesita una de estas dos herramientas; sin ninguna falla con `PermissionError: Permission denied: 'None'`:

1. **`uv` (recomendado):** `pip install uv`. MLflow lo usa automáticamente, descarga Python 3.12 sin compilarlo y crea el entorno en segundos.
2. **`pyenv`:** instálalo con `curl https://pyenv.run | bash`, agrega a `~/.bashrc` las líneas que indica el instalador (`PYENV_ROOT`, `PATH` y `eval "$(pyenv init - bash)"`), abre una terminal nueva e instala las [dependencias de compilación](https://github.com/pyenv/pyenv/wiki#suggested-build-environment). La primera corrida compila Python 3.12 y tarda varios minutos.

En ambos casos el entorno se crea una sola vez y las siguientes corridas lo reutilizan.

| Punto de entrada | Qué ejecuta | Parámetros |
| --- | --- | --- |
| `ingest` | `mibici_mlops_module.dataset --mlflow` | ninguno: meses, rutas y criterios de limpieza vienen de `config.py` |
| `features` | `mibici_mlops_module.features` | ninguno |
| `train` | `mibici_mlops_module.modeling.pipeline` (entrena y evalúa) | `question` (`p1`, `p2`, `p3`; vacío = todas), `model` (ver abajo; vacío = todos los de la pregunta), e hiperparámetros: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_samples`, `learning_rate`, `max_iter`, `max_leaf_nodes`, `l2_regularization`, `k`, `n_init` |

Rutas, fechas, particiones y semilla no son parámetros: salen de `config.py`. Un hiperparámetro vacío toma el valor de `configs/hiperparametros.yaml`; si se indica, sustituye al del YAML. Con `model` se entrena solo ese modelo (más los baselines de la pregunta). Alias de `model`: P1 `rf` (Random Forest), `hgb` (HistGradientBoosting Poisson), `lr` (regresión lineal); P2 `dt` (árbol balanceado), `rf` (Random Forest con class weight); P3 `kmeans`.

**Hiperparámetros por pregunta y modelo** (los que se pueden pasar con `-P`; los valores son los de la combinación `base` de `configs/hiperparametros.yaml`):

| Pregunta | `model` | Modelo | Hiperparámetros (`-P`) | Valor base |
| --- | --- | --- | --- | --- |
| P1 | `rf` | Random Forest (regresión) | `n_estimators` | 100 |
| | | | `max_depth` | 14 |
| | | | `min_samples_leaf` | 20 |
| | | | `max_samples` | 0.3 |
| P1 | `hgb` | HistGradientBoosting (Poisson) | `learning_rate` | 0.10 |
| | | | `max_iter` | 300 |
| | | | `max_leaf_nodes` | 63 |
| | | | `l2_regularization` | 0.0 |
| P1 | `lr` | Regresión lineal | ninguno | n/a |
| P2 | `dt` | Árbol balanceado | `max_depth` | 10 |
| | | | `min_samples_leaf` | 50 |
| P2 | `rf` | Random Forest con class weight | `n_estimators` | 100 |
| | | | `max_depth` | 14 |
| | | | `min_samples_leaf` | 20 |
| | | | `max_samples` | 0.3 |
| P3 | `kmeans` | K-Means | `k` | 3 |
| | | | `n_init` | 10 |

Pasar un hiperparámetro que el modelo no usa (por ejemplo `-P k=3` con `model=rf`) provoca un error, así que indica solo los de su fila. Si pasas alguno, el nombre del run se deriva de los valores (`P1-demanda-random-forest-n100-prof14-...`); si no pasas ninguno, se usa el `nombre` del YAML.

### Qué queda en MLflow

- Un run padre por cada `mlflow run`, del que cuelgan los demás. Lo crea `mlflow run` con un nombre aleatorio (p. ej. `puzzled-bat-941`), por lo que el código lo renombra: `Ingesta de datos` para `ingest`, `Construcción de variables` para `features` y `Pregunta 1 - demanda por hora (rf)`, `Pregunta 2 - clasificación` o `Pregunta 3 - grupos` para `train`, según `question` y `model`. Si pasas `model` o hiperparámetros, se agregan entre paréntesis para distinguir corridas de la misma pregunta (p. ej. `Pregunta 1 - demanda por hora (rf n_estimators=200 max_depth=12)`).
- Un run de datos, `datos-<data_version>` (etapa `ingesta`), con la versión de los datos, el commit de Git, los meses, los criterios de limpieza, los conteos y `manifest.json`.
- Un run por candidato, con nombre descriptivo (`P1-demanda-random-forest-base`, `P3-tipologia-kmeans-k3`, etc.). Cada run guarda parámetros (fechas de cada partición, `data_version`, hiperparámetros), métricas, figuras, el modelo con su firma y las etiquetas `equipo`, `pregunta`, `git_commit` y `git_dirty`.
- Un run de evaluación por pregunta (`P1-demanda-evaluacion-<modelo>`): carga el mejor modelo desde `models/` con el predictor y registra sus métricas en validación y prueba, las figuras y `max_dif_vs_entrenamiento` (debe ser 0: el modelo guardado reproduce lo entrenado).
- El mejor candidato de cada pregunta, según la validación, queda en `models/pX.joblib`.

### Cómo leer los runs en MLflow

Ejemplo con `mlflow run . -e train --experiment-name mibici-equipo04 -P question=p1 -P model=rf -P n_estimators=100 -P max_depth=14 -P min_samples_leaf=20 -P max_samples=0.3`. En la pestaña **Runs** del experimento aparece un run padre (`Pregunta 1 - demanda por hora (rf n_estimators=100 max_depth=14 min_samples_leaf=20 max_samples=0.3)`, con un **+** para expandirlo) y, dentro de él, estos runs hijos:

| Run | Qué es |
|---|---|
| `P1-demanda-baseline-persistencia` | Baseline sin aprendizaje: predice que la demanda de la hora siguiente es la de la hora actual. Es el piso mínimo. |
| `P1-demanda-baseline-estacional-semanal` | Baseline sin aprendizaje: predice la demanda de la misma hora de la semana anterior (168 h atrás). Es un piso más exigente. |
| `P1-demanda-random-forest-n100-prof14-hoja20-muestra0.3` | El candidato entrenado. El nombre se deriva de los hiperparámetros: 100 árboles (`n100`), profundidad máxima 14 (`prof14`), mínimo 20 muestras por hoja (`hoja20`) y 30 % de los datos por árbol (`muestra0.3`). Guarda parámetros, métricas de validación, figuras y el modelo con su firma. |
| `P1-demanda-evaluacion-random-forest-n100-prof14-hoja20-muestra0.3` | No entrena: carga de `models/` el mejor modelo y lo evalúa con el predictor sobre validación y prueba. Registra `max_dif_vs_entrenamiento`, que debe ser 0. |

Para interpretarlos:

1. **Compara las métricas de validación** del candidato contra los dos baselines. Un modelo solo vale la pena si los supera.
2. **Revisa el run de evaluación:** el nombre indica qué candidato ganó, y `max_dif_vs_entrenamiento` en 0 confirma que el modelo guardado reproduce lo entrenado.
3. **Si la tabla se ve plana,** activa **Group by** en la barra superior o busca la fila del padre. Los runs sueltos y sin **+** salieron de ejecutar el script directo (`python -m ...`) y no de `mlflow run`.
4. Al entrenar con otros hiperparámetros, cada combinación crea su propio run (el nombre cambia con los valores), de modo que puedes compararlas seleccionándolas y pulsando **Compare**.

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

## Ejecución de principio a fin

Pasos para reproducir todo desde cero, hasta ver los resultados en MLflow. Se ejecutan desde la raíz del repositorio en WSL/Linux.

> Si el paso 6 falla con `PermissionError: Permission denied: 'None'`, falta `uv` o `pyenv`: consulta el [requisito en MLflow Project](#mlflow-project).

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

# 4. Servidor de tracking local con base de datos (terminal aparte, déjalo corriendo)
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

# 5. En la terminal de trabajo, apunta MLflow al servidor
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# 6. Datos: descarga los CSV, los limpia, genera data/manifest.json y registra la versión en MLflow
mlflow run . -e ingest --experiment-name mibici-equipo04

# 7. Variables: construye las tablas P1, P2 y P3 en data/processed/
mlflow run . -e features --experiment-name mibici-equipo04

# 8. Entrenamiento + evaluación del modelo elegido de cada pregunta (Ejemplos con Random Forest y KMeans; ver `configs/hiperparametros.yaml` para más combinaciones)

# P1: demanda por estación y hora
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p1 -P model=rf -P n_estimators=100 -P max_depth=14 -P min_samples_leaf=20 -P max_samples=0.3

# P2: riesgo de vaciado o saturación
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p2 -P model=rf -P n_estimators=100 -P max_depth=14 -P min_samples_leaf=20 -P max_samples=0.3

# P3: tipología de estaciones
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p3 -P model=kmeans -P k=3 -P n_init=10
```

Después abre <http://127.0.0.1:5000> y entra al experimento **mibici-equipo04**.

Notas:

- Entrenar P1, P2 y P3 sin indicar `model` registra 9 runs con la configuración actual (una combinación por modelo, más los baselines): 4 de P1, 3 de P2 y 2 de P3. Para probar más combinaciones, quita el `#` a las que están comentadas en `configs/hiperparametros.yaml`; con todas son más de 20 runs y tarda varias decenas de minutos. Para una prueba rápida usa `--fraccion 0.1` (10 % del entrenamiento).
- Conviene hacer `git commit` antes del paso 6: cada run guarda el commit con el que se ejecutó (`git_commit`) y marca `git_dirty=true` si hay cambios sin confirmar.
- Levanta `mlflow server` desde la raíz del repositorio: `mlflow.db` se crea ahí y la ruta `sqlite:///mlflow.db` es relativa.
