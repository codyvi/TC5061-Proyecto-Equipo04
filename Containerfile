# Imagen de la API de inferencia de MiBici (compatible con OCI: podman build / docker build).
#
#   podman build -t mibici-api -f Containerfile .
#   podman run --rm -p 8000:8000 \
#     -e MLFLOW_TRACKING_URI=http://host.containers.internal:5000 mibici-api
#
# Los modelos no van en la imagen: se recuperan del MLflow Model Registry al arrancar.
# Python 3.12 es la versión con la que se entrenaron y registraron los modelos.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Primero las dependencias: esta capa se reutiliza mientras requirements-api.txt no cambie
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY mibici_mlops_module/ mibici_mlops_module/
COPY app/ app/

# Sin privilegios de root; la ruta de `data/` y `models/` no se usa en inferencia
RUN useradd --create-home --uid 1000 api && chown -R api /app
USER api

# Obligatoria al ejecutar: dirección del servidor MLflow (ver README)
ENV MLFLOW_TRACKING_URI="" \
    MODELO_ALIAS=champion

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
