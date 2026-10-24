

MLOps Team 04 - MiBici








Propósito

El principal costo operativo de un sistema de bicicleta pública son las estaciones vacías o llenas, que obligan a rebalancear con camionetas. Este proyecto construye, con datos abiertos de MiBici (Guadalajara), una base reproducible de experimentos de aprendizaje automático para apoyar esa operación: ingesta versionada de los datos, variables sin fuga de información, modelos con baseline y seguimiento completo de cada experimento en MLflow.

Las tres preguntas







Pregunta



Qué se predice o agrupa



Granularidad y horizonte



Quién la usa y qué decide





P1. Demanda (regresión)



Número de viajes que inician en cada estación



Estación × hora, siguiente hora (t+1)



Operaciones: cuántas bicicletas llevar a cada estación





P2. Riesgo (clasificación)



Clase vaciado, normal o saturación, según el índice de flujo (llegadas − salidas) / capacidad con umbral de ±0.20



Estación × hora, siguiente hora (t+1)



Logística: qué estaciones visitar primero en la ruta





P3. Tipología (clustering)



Grupos de estaciones con patrones de uso similares



Estación, sin horizonte (perfil de enero a abril)



Planeación: qué estrategia aplicar por tipo de estación

Los datos solo registran viajes, no el inventario real de bicicletas; por eso la clase de P2 es una aproximación al riesgo. Detalle en el ML Canvas.

Para reproducir todo desde cero, ve a Ejecución de principio a fin.

Integrantes y roles







Integrante



Matrícula



Rol



Aportes





David Alonso Cantú Martínez (codyvi)



A00822455



MLOps Engineer / Data Engineer



Estructura del repositorio (CCDS), ingesta y versionado de datos, variables, y entrenamiento y evaluación con MLflow





Martín Rogelio Cantú González



A01797809



Software Engineer



Pruebas automáticas (pytest) y refactorización del código





Luis Alfredo Ojeda Benito



A01750598



Data Scientist



Entrenamiento y registro en MLflow de P1 (demanda)





Carlo Eugenio Leviaguirre Chavez



A01022264



Machine Learning Engineer



Ajuste de hiperparámetros y entrenamiento de modelos



ML Canvas y arquitectura





ML Canvas



Diagrama de arquitectura



Organización del proyecto

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



Entorno virtual

Desde la raíz del proyecto:

# Crear el entorno virtual "mibici"
python3 -m venv mibici

# Activarlo (WSL/Linux)
source mibici/bin/activate

# Instalar dependencias
pip install -U pip
pip install -r requirements.txt

# Registrarlo como kernel de Jupyter
python -m ipykernel install --user --name mibici --display-name "Python (mibici)"

Si python3 -m venv falla en WSL con un error de ensurepip, instala antes sudo apt install python3-venv.

Uso diario





Activar: source mibici/bin/activate (el prompt muestra (mibici)).



Desactivar: deactivate.



Notebooks: elige el kernel Python (mibici) en VS Code o en jupyter lab.



Nueva dependencia: pip install paquete y agrégala a requirements.txt.



Si VS Code no detecta el entorno, usa Python: Select Interpreter y apunta a mibici/bin/python.



Datos

Los datos vienen de MiBici datos abiertos (enero a junio de 2026 y la nomenclatura de estaciones). El pipeline es reproducible desde la terminal, sin pasos manuales:

python -m mibici_mlops_module.dataset            # descarga, normaliza, limpia y versiona
python -m mibici_mlops_module.dataset --mlflow   # además registra la versión en MLflow
python -m mibici_mlops_module.features           # construye las tablas P1, P2 y P3

Opciones útiles de dataset: --meses 2026_01 --meses 2026_02 (otros meses), --no-descargar (usa los CSV que ya estén en data/raw/) y --force (vuelve a descargar).







Etapa



Carpeta



Contenido





Crudo



data/raw/



CSV originales, inmutables (los de enero vienen en Latin-1)





Normalizado



data/interim/



Parquet por mes, nomenclatura.parquet y viajes_limpios.parquet, con columnas en snake_case y fechas como datetime





Modelado



data/processed/



p1_demanda, p2_riesgo, p3_perfiles y resumen.json

Versionado de los datos. data/manifest.json guarda, por cada archivo, su sha256, tamaño y número de registros, y una data_version (hash de los CSV crudos y de los parámetros de limpieza). Es el único archivo de data/ que se sube a Git: si cambia un CSV, cambia la versión. Con --mlflow la versión queda como parámetro de una corrida en el experimento mibici-equipo04; en el entrenamiento se puede leer con cargar_manifest()["data_version"].

El resto de data/ está en .gitignore: los datos no se redistribuyen y se usan solo con fines académicos, citando la fuente. Después de generar las tablas, abre notebooks/1.0-dcm-eda-mibici.ipynb para el análisis exploratorio.

Servidor de tracking y Model Registry

El Model Registry requiere un almacenamiento respaldado por base de datos, así que todas las corridas se registran en un servidor local con SQLite. mlflow.db está en .gitignore.

mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

config.py lee MLFLOW_TRACKING_URI (también se puede poner en un .env); si no está definida usa sqlite:///<raíz>/mlflow.db directamente.

MLflow Project

Todas las corridas se lanzan con mlflow run desde el proyecto definido en [MLproject](MLproject), que crea un entorno virtual con las versiones fijas de [python_env.yaml](python_env.yaml) (las mismas de requirements.txt que usa el código; la primera vez tarda en instalarlo). Se usa --experiment-name mibici-equipo04 para que la corrida del proyecto y las anidadas queden en el mismo experimento.

Requisito: mlflow run instala por su cuenta el Python de python_env.yaml (3.12), sin importar el que tengas en tu equipo. Para eso necesita una de estas dos herramientas; sin ninguna falla con PermissionError: Permission denied: 'None':





uv (recomendado): pip install uv. MLflow lo usa automáticamente, descarga Python 3.12 sin compilarlo y crea el entorno en segundos.



pyenv: instálalo con curl https://pyenv.run | bash, agrega a ~/.bashrc las líneas que indica el instalador (PYENV_ROOT, PATH y eval "$(pyenv init - bash)"), abre una terminal nueva e instala las dependencias de compilación. La primera corrida compila Python 3.12 y tarda varios minutos.

En ambos casos el entorno se crea una sola vez y las siguientes corridas lo reutilizan.







Punto de entrada



Qué ejecuta



Parámetros





ingest



mibici_mlops_module.dataset --mlflow



ninguno: meses, rutas y criterios de limpieza vienen de config.py





features



mibici_mlops_module.features



ninguno





train



mibici_mlops_module.modeling.pipeline (entrena y evalúa)



question (p1, p2, p3; vacío = todas), model (ver abajo; vacío = todos los de la pregunta), seed (semilla de los modelos; por defecto 42, el SEED de config.py), e hiperparámetros: n_estimators, max_depth, min_samples_leaf, max_samples, learning_rate, max_iter, max_leaf_nodes, l2_regularization, k, n_init





register



mibici_mlops_module.modeling.registry (registra el modelo de models/ en el Model Registry)



question (p1, p2, p3; vacío = todas), alias (champion por defecto, o challenger)

Rutas, fechas y particiones no son parámetros: salen de config.py. La semilla sí se puede cambiar con seed para medir la estabilidad entre corridas (-P seed=1); sin indicarla se usa 42. Un hiperparámetro vacío toma el valor de configs/hiperparametros.yaml; si se indica, sustituye al del YAML. Con model se entrena solo ese modelo (más los baselines de la pregunta). Sin hiperparámetros se corren todas sus combinaciones del YAML (la búsqueda anidada); con alguno, una sola combinación. Alias de model: P1 rf (Random Forest), hgb (HistGradientBoosting Poisson), lr (regresión lineal); P2 dt (árbol balanceado), rf (Random Forest con class weight); P3 kmeans.

Hiperparámetros por pregunta y modelo (los que se pueden pasar con -P; los valores son los de la combinación base de configs/hiperparametros.yaml):







Pregunta



model



Modelo



Hiperparámetros (-P)



Valor base





P1



rf



Random Forest (regresión)



n_estimators



100

















max_depth



14

















min_samples_leaf



20

















max_samples



0.3





P1



hgb



HistGradientBoosting (Poisson)



learning_rate



0.10

















max_iter



300

















max_leaf_nodes



63

















l2_regularization



0.0





P1



lr



Regresión lineal



ninguno



n/a





P2



dt



Árbol balanceado



max_depth



10

















min_samples_leaf



50





P2



rf



Random Forest con class weight



n_estimators



100

















max_depth



14

















min_samples_leaf



20

















max_samples



0.3





P3



kmeans



K-Means



k



3

















n_init



10

Pasar un hiperparámetro que el modelo no usa (por ejemplo -P k=3 con model=rf) provoca un error, así que indica solo los de su fila. Si pasas alguno, el nombre del run se deriva de los valores (P1-demanda-random-forest-n100-prof14-...); si no pasas ninguno, se usa el nombre del YAML.

Ejemplos con hiperparámetros (ver configs/hiperparametros.yaml para más combinaciones):

# P1: Random Forest con una combinación puntual
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p1 -P model=rf -P n_estimators=100 -P max_depth=14 -P min_samples_leaf=20 -P max_samples=0.3

# P2: Random Forest con class weight
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p2 -P model=rf -P n_estimators=100 -P max_depth=14 -P min_samples_leaf=20 -P max_samples=0.3

# P3: K-Means con k=3
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p3 -P model=kmeans -P k=3 -P n_init=10

# Cualquiera de los anteriores con otra semilla
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p1 -P model=rf -P n_estimators=300 -P seed=1



Qué queda en MLflow





Un run padre por cada mlflow run, del que cuelgan los demás. Lo crea mlflow run con un nombre aleatorio (p. ej. puzzled-bat-941), por lo que el código lo renombra: Ingesta de datos para ingest, Construcción de variables para features y Pregunta 1 - demanda por hora (rf), Pregunta 2 - clasificación o Pregunta 3 - grupos para train, según question y model. Si pasas model o hiperparámetros, se agregan entre paréntesis para distinguir corridas de la misma pregunta (p. ej. Pregunta 1 - demanda por hora (rf n_estimators=200 max_depth=12)).



Un run de datos, datos-<data_version> (etapa ingesta), con la versión de los datos, el commit de Git, los meses, los criterios de limpieza, los conteos y manifest.json.



Un run padre de búsqueda por pregunta (P1-busqueda-demanda, stage_of_work=search) y, anidado en él, un run por candidato con nombre descriptivo (P1-demanda-random-forest-base, P3-tipologia-kmeans-k3, etc.). Cada run guarda parámetros (fechas de cada partición, data_version, semilla, hiperparámetros), métricas (con tiempo_entrenamiento_s y tiempo_inferencia_ms_por_1000_filas), el conjunto de entrenamiento (log_input), el modelo con su firma y ejemplo de entrada, y las etiquetas equipo, question, model_family, stage_of_work (baseline o candidate), git_commit y git_dirty. Los artefactos van en subcarpetas: plots/ (P1: real contra predicho; P2: matriz de confusión y curva precisión-recall; P3: mapa y perfil de los grupos), data/ (manifest.json) y config/ (hiperparametros.yaml). No se registran datos de usuarios: las tablas son por estación y hora.



Un run de evaluación por pregunta (P1-demanda-evaluacion-<modelo>): carga el mejor modelo desde models/ con el predictor y registra sus métricas en validación y prueba, las figuras y max_dif_vs_entrenamiento (debe ser 0: el modelo guardado reproduce lo entrenado).



El mejor candidato de cada pregunta, según la validación, queda en models/pX.joblib.



Cómo leer los runs en MLflow

Ejemplo con mlflow run . -e train --experiment-name mibici-equipo04 -P question=p1 -P model=rf -P n_estimators=100 -P max_depth=14 -P min_samples_leaf=20 -P max_samples=0.3. En la pestaña Runs del experimento aparece un run padre (Pregunta 1 - demanda por hora (rf n_estimators=100 max_depth=14 min_samples_leaf=20 max_samples=0.3), con un + para expandirlo) y, dentro de él, estos runs hijos:







Run



Qué es





P1-demanda-baseline-persistencia



Baseline sin aprendizaje: predice que la demanda de la hora siguiente es la de la hora actual. Es el piso mínimo.





P1-demanda-baseline-estacional-semanal



Baseline sin aprendizaje: predice la demanda de la misma hora de la semana anterior (168 h atrás). Es un piso más exigente.





P1-demanda-random-forest-n100-prof14-hoja20-muestra0.3



El candidato entrenado. El nombre se deriva de los hiperparámetros: 100 árboles (n100), profundidad máxima 14 (prof14), mínimo 20 muestras por hoja (hoja20) y 30 % de los datos por árbol (muestra0.3). Guarda parámetros, métricas de validación, figuras y el modelo con su firma.





P1-demanda-evaluacion-random-forest-n100-prof14-hoja20-muestra0.3



No entrena: carga de models/ el mejor modelo y lo evalúa con el predictor sobre validación y prueba. Registra max_dif_vs_entrenamiento, que debe ser 0.

Para interpretarlos:





Compara las métricas de validación del candidato contra los dos baselines. Un modelo solo vale la pena si los supera.



Revisa el run de evaluación: el nombre indica qué candidato ganó, y max_dif_vs_entrenamiento en 0 confirma que el modelo guardado reproduce lo entrenado.



Si la tabla se ve plana, activa Group by en la barra superior o busca la fila del padre. Los runs sueltos y sin + salieron de ejecutar el script directo (python -m ...) y no de mlflow run.



Al entrenar con otros hiperparámetros, cada combinación crea su propio run (el nombre cambia con los valores), de modo que puedes compararlas seleccionándolas y pulsando Compare.



Flujo de trabajo con Git y GitHub

La rama main está protegida por un ruleset de GitHub: nadie sube directo, no se permite reescribir su historial ni borrarla, y todo cambio entra por Pull Request con al menos una aprobación. El único método de integración permitido es rebase, por lo que el historial de main es lineal.





Actualizar main: git switch main && git pull.



Crear una rama por cambio, con prefijo según el tipo: feat/<tema> (funcionalidad), fix/<tema> (corrección) o docs/<tema> (documentación). Ejemplo: git switch -c feat/refactor.



Commits con el formato tipo(alcance): descripción en español, donde el tipo es feat, fix, docs, test o refactor. Ejemplo: feat(modeling): entrenamiento y evaluación de P1, P2 y P3 con MLflow.



Antes de abrir el PR: python -m pytest -q y ruff check. No se suben datos, modelos, mlflow.db, mlruns/ ni .env: ya están en .gitignore.



Abrir el Pull Request hacia main con un resumen de qué cambia y por qué. Debe revisarlo y aprobarlo al menos otra persona del equipo; quien lo abre no lo aprueba.



Integrar con Rebase and merge y borrar la rama.



Scripts disponibles

Todos se ejecutan como módulo (python -m ...) y aceptan --help.







Script



Para qué sirve



Ejemplo





mibici_mlops_module.dataset



Descarga, normaliza, limpia y versiona los datos (data/manifest.json). Opciones: --meses, --no-descargar, --force, --mlflow (registra la versión en el experimento mibici-equipo04).



python -m mibici_mlops_module.dataset --mlflow





mibici_mlops_module.features



Construye p1_demanda, p2_riesgo, p3_perfiles y resumen.json, y valida que no haya fuga de información.



python -m mibici_mlops_module.features





mibici_mlops_module.modeling.pipeline



Entrena y evalúa de punta a punta, registrando ambos pasos en MLflow. Es el recomendado.



python -m mibici_mlops_module.modeling.pipeline --pregunta p1





mibici_mlops_module.modeling.train



Solo entrena (un run por candidato) y guarda el mejor en models/.



python -m mibici_mlops_module.modeling.train --pregunta p2





mibici_mlops_module.modeling.predict



Genera predicciones con el modelo guardado y las escribe en data/processed/pX_predicciones.parquet.



python -m mibici_mlops_module.modeling.predict --pregunta p1 --conjunto prueba





pytest (carpeta tests/)



Pruebas automáticas de features.py: rezagos y medias móviles sin fuga del futuro, objetivos en t+1, clases de riesgo según el umbral, partición temporal y validación con muestras pequeñas. No necesitan los datos reales.



python -m pytest -q

Opciones de pipeline y train:





--pregunta p1|p2|p3|todas (por defecto todas).



--fraccion 0.1: usa solo esa fracción del entrenamiento (pruebas rápidas).



--seed 42: semilla de los modelos.



--config ruta.yaml: hiperparámetros alternativos (por defecto configs/hiperparametros.yaml).

Opciones de predict: --pregunta, --conjunto entrenamiento|validacion|prueba (P1 y P2), --entrada y --salida (rutas Parquet).

Hiperparámetros. configs/hiperparametros.yaml define las combinaciones de cada modelo. Cada elemento de la lista es un run; el campo opcional nombre forma parte del nombre del run (P1-demanda-random-forest-<nombre>) y, si se omite, se arma con los hiperparámetros. Agrega o quita combinaciones y vuelve a correr.

Selección del mejor modelo

Para cada pregunta se corrió la búsqueda completa en el mismo experimento (mibici-equipo04), con la misma partición temporal (entrenamiento enero a abril, validación mayo, prueba junio) y las mismas métricas que en la entrega anterior. Los comandos fueron:

mlflow run . -e train --experiment-name mibici-equipo04 -P question=p1
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p2
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p3

Cada pregunta se repitió con las semillas 1, 2, 3, 4 y 42 (-P seed=<n>; la 42 es la predeterminada y la que queda en models/). Para generar reports/estabilidad.csv (media y desviación de la métrica de decisión, y tiempos de entrenamiento e inferencia, por pregunta y modelo) una vez terminadas las corridas:



Nota: El comando necesita que MLFLOW_TRACKING_URI esté apuntando a tu servidor, igual que al entrenar. Si no, leerá el mlflow.db directo y puede mostrar resultados distintos o vacíos.

python -m mibici_mlops_module.modeling.estabilidad --salida reports/estabilidad.csv

Con --pregunta p1 (o p2, p3) resume solo esa pregunta; sin --salida solo imprime las tablas en la terminal.

Cada comando crea un run padre de búsqueda (stage_of_work=search) y un run hijo por candidato (baseline o candidate), con las etiquetas question y model_family. Para filtrarlos en la interfaz de MLflow: tags.question = "P1" and tags.stage_of_work = "candidate".

Cómo se eligió. El código guarda en models/ el candidato que no es baseline con la mejor métrica de validación (nunca de prueba, para que la prueba siga siendo una estimación honesta). Después se revisó que la elección se sostuviera entre semillas y frente al costo.







Pregunta



Métrica de decisión



Por qué esa métrica





P1 (demanda)



validacion_mae (menor es mejor)



error típico en bicicletas por estación y hora, que es lo que Operaciones usa para decidir el rebalanceo





P2 (riesgo)



validacion_balanced_accuracy (mayor es mejor)



las clases de vaciado y saturación son raras; la exactitud simple las ignoraría





P3 (tipología)



entrenamiento_silhouette (mayor es mejor)



no hay etiqueta real; mide qué tan compactos y separados quedan los grupos

Resultados. Media ± desviación estándar de 5 semillas (fuente: reports/estabilidad.csv). El entrenamiento está en segundos y la inferencia en milisegundos por cada 1000 filas.

P1 (MAE de validación, menor es mejor)







Candidato



MAE



Entrenamiento (s)



Inferencia (ms)





Baseline persistencia



0.9689 ± 0.0000



0.001



0.05





Baseline estacional semanal



0.8911 ± 0.0000



0.001



0.05





HistGradientBoosting Poisson base



0.7207 ± 0.0006



7.7



8.8





HistGradientBoosting Poisson lento-lr0.05



0.7244 ± 0.0003



11.0



15.2





Random Forest muy-profundo



0.7326 ± 0.0003



13.1



55.6





Random Forest base



0.7391 ± 0.0001



16.8



53.8





Random Forest bosque-grande



0.7441 ± 0.0001



41.4



99.1

P2 (balanced accuracy de validación, mayor es mejor)







Candidato



Balanced accuracy



Entrenamiento (s)



Inferencia (ms)





Baseline clase mayoritaria



0.3333 ± 0.0000



0.7



0.08





Árbol balanceado muy-profundo



0.6401 ± 0.0002



8.8



1.5





Árbol balanceado simple



0.6445 ± 0.0000



6.1



1.6





Árbol balanceado base



0.6720 ± 0.0000



7.3



1.4





Random Forest class weight muy-profundo



0.5808 ± 0.0022



15.8



78.5





Random Forest class weight base



0.6985 ± 0.0006



18.6



73.2





Random Forest class weight bosque-grande



0.7014 ± 0.0007



39.5



132.3

P3 (silhouette, mayor es mejor)







Candidato



Silhouette



Entrenamiento (s)



Inferencia (ms)





Baseline cuantiles de volumen



0.0801 ± 0.0000



0.001



0.6





K-Means k3



0.3079 ± 0.0729



0.025



13.0





K-Means k5



0.1939 ± 0.0069



0.024



14.0





K-Means k7



0.2308 ± 0.0003



0.025



13.0

Modelo elegido por pregunta. P1: HistGradientBoosting Poisson base. P2: Random Forest con class weight bosque-grande. P3: K-Means con k=3.

Consideraciones críticas.





P1. HistGradientBoosting base supera a las variantes de Random Forest en MAE y además es el más barato en inferencia (8.8 ms frente a 54 a 99 ms). La ventaja sobre lento-lr0.05 es de solo 0.004, pero se sostiene en las 5 semillas (las desviaciones son 0.0006 y 0.0003) y base entrena y predice más rápido. Frente al baseline estacional, el MAE baja de 0.891 a 0.721 (una mejora de alrededor de 19 %).



P2. bosque-grande supera a base por 0.003 de balanced accuracy; la diferencia es mayor que la dispersión entre semillas (0.0007), así que es real pero pequeña. Cuesta el doble en entrenamiento (39.5 s frente a 18.6 s) y casi el doble en inferencia (132 ms frente a 73 ms). El árbol balanceado base queda 0.03 por debajo, pero entrena y predice mucho más rápido (1.4 ms por cada 1000 filas). Se elige el bosque porque la meta es detectar vaciados y saturaciones, que son raros; si el costo de inferencia fuera una restricción, el árbol sería una alternativa razonable.



P3. k3 es el mejor en promedio, pero es el candidato menos estable: su silhouette cambia entre semillas (desviación de 0.073) y el 0.3396 de la semilla 42 es un valor favorable frente a la media de 0.308. Aun así, incluso su peor caso razonable queda por encima de k7 (0.231) y de k5 (0.194), cuya variación es mínima. El silhouette mide separación estadística, no utilidad para Planeación; por eso cada run guarda el mapa y los perfiles por grupo en plots/. Elegir k3 también resulta más sencillo de operar que manejar 5 o 7 perfiles.



Costo. P1 y P3 no tienen un costo relevante. En P2 el Random Forest es el candidato más caro, pero entrena en menos de un minuto y predice 1000 filas en 0.13 s, lo que es manejable para un uso por hora.



Límites. La comparación usa una sola partición temporal (validación en mayo, prueba en junio), así que no captura estacionalidad anual. La estabilidad se mide solo respecto a la semilla de los modelos, no a otros periodos.



Model Registry y gestión de modelos en producción



Objetivo

Administrar los modelos elegidos de cada pregunta con MLflow Model Registry, conservando su linaje y permitiendo cargarlos por alias sin depender de un número de versión.

Modelos registrados







Pregunta



Nombre en el registro



Modelo elegido



Alias





P1 (demanda)



mibici-equipo04-p1-demanda



HistGradientBoosting Poisson base



champion





P2 (riesgo)



mibici-equipo04-p2-riesgo



Random Forest class weight bosque-grande



champion





P3 (grupos)



mibici-equipo04-p3-grupos



K-Means k3



champion

El nombre sigue el patrón mibici-<equipo>-<pregunta>-<tarea>, con el equipo tomado de config.py.

Cómo se registra

El módulo [registry.py](mibici_mlops_module/modeling/registry.py) registra el modelo que está en models/, que es el que el entrenamiento eligió por validación. De ese archivo salen la corrida de origen, el commit y la versión de los datos, así que no hay identificadores escritos a mano y un reentrenamiento se registra con el mismo comando. Requiere el servidor de tracking y MLFLOW_TRACKING_URI configurados (ver "Servidor de tracking y Model Registry").

# P1, P2 y P3 como champion (producción)
mlflow run . -e register --experiment-name mibici-equipo04
# o, sin MLflow Project
python -m mibici_mlops_module.modeling.registry
python -m mibici_mlops_module.modeling.registry --pregunta p1

# Siguiente mejor versión, sin mover el champion
python -m mibici_mlops_module.modeling.registry --pregunta p1 --alias challenger

Cada versión registrada conserva su linaje:





Experimento y corrida de origen: etiquetas experiment y source_run_id, además del vínculo de la versión con su corrida.



Código y datos: git_commit, git_dirty y data_version.



Selección: modelo, seed, criterio y valor_criterio (la métrica de validación con la que se eligió).



Descripción: el modelo registrado describe la decisión que apoya, y cada versión indica el modelo, su métrica, la versión de los datos y el commit.

El registro es idempotente: si el alias ya apunta a una versión de la misma corrida, no crea otra. Y se rechaza registrar un modelo entrenado con una fracción del entrenamiento (--fraccion), porque models/ se sobrescribe también con las pruebas rápidas.

Esquema de ciclo de vida: aliases

Se usan aliases (champion y, opcionalmente, challenger) en lugar de las etapas clásicas (Production, Staging). Las etapas están en desuso en MLflow, y los aliases permiten:





apuntar a una versión sin cambiar el código de inferencia: se carga models:/<nombre>@champion;



cambiar de versión, o volver a la anterior, moviendo el alias, sin tocar las versiones;



tener más de un alias a la vez (por ejemplo, la versión en producción y la siguiente en evaluación).

Cuando un candidato nuevo supere al champion, se registra con --alias challenger, se compara, y si gana se vuelve a ejecutar el registro con champion para moverle el alias. En esta entrega solo se asignó champion.

Verificación de carga

Después de asignar el alias, el comando carga el modelo por su nombre y alias, y falla si no puede. También se puede comprobar a mano:

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo04-p1-demanda@champion")



Reproducibilidad

El registro vive en la base de MLflow del entorno donde se ejecuta; las versiones y los alias no se suben a Git (mlflow.db está en .gitignore). Para reproducirlo en otro entorno hay que entrenar con las instrucciones de Ejecución de principio a fin y ejecutar el registro contra el servidor de ese entorno.

Ejecución de principio a fin

Pasos para reproducir todo desde cero, hasta ver los resultados en MLflow. Se ejecutan desde la raíz del repositorio en WSL/Linux. 



Nota: La ejecución por defecto usa los hiperparámetros de configs/hiperparametros.yaml para entrenar los modelos, si quieres probar otros, pasa --config ruta.yaml a mlflow run . -e train o pasa los hiperparámetros con -P (ver MLflow Project).



Si el paso 6 falla con PermissionError: Permission denied: 'None', falta uv o pyenv: consulta el requisito en MLflow Project.

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

# 8. Entrenamiento + evaluación del modelo elegido de cada pregunta 

# P1: demanda por estación y hora
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p1

# P2: riesgo de vaciado o saturación
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p2

# P3: tipología de estaciones
mlflow run . -e train --experiment-name mibici-equipo04 -P question=p3

# 9. Model Registry: registra el modelo de models/ de P1, P2 y P3 con el alias champion
mlflow run . -e register --experiment-name mibici-equipo04 --run-name registro-champion

Después abre http://127.0.0.1:5000 y entra al experimento mibici-equipo04; los modelos registrados están en la pestaña Models (detalle en Model Registry y gestión de modelos en producción).

Notas:





Entrenar una pregunta sin indicar model corre sus baselines y todas las combinaciones de configs/hiperparametros.yaml (7 candidatos en P1, 7 en P2 y 4 en P3), anidados bajo un run padre de búsqueda. Para una prueba rápida usa --fraccion 0.1 (10 % del entrenamiento) desde python -m mibici_mlops_module.modeling.pipeline.



El paso 9 registra lo que haya en models/, así que ejecútalo después de entrenar las tres preguntas (o pasa -P question=p1 para registrar solo una). Es idempotente y rechaza modelos entrenados con --fraccion.



Conviene hacer git commit antes del paso 6: cada run guarda el commit con el que se ejecutó (git_commit) y marca git_dirty=true si hay cambios sin confirmar.



Levanta mlflow server desde la raíz del repositorio: mlflow.db se crea ahí y la ruta sqlite:///mlflow.db es relativa.

