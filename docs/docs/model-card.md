# Model Card — MiBici Equipo 04

## 1. Descripción general

El proyecto MiBici aplica técnicas de aprendizaje automático para analizar la operación de un sistema de bicicletas compartidas.

Se desarrollaron tres modelos:

- P1: Predicción de demanda de bicicletas.
- P2: Clasificación del riesgo de vaciado o llenado de estaciones.
- P3: Agrupación de estaciones mediante clustering.

Los modelos se gestionan mediante MLflow Tracking y Model Registry.

## 2. Modelos desarrollados

### P1 — Predicción de demanda

**Objetivo:** Estimar la cantidad de salidas de bicicletas durante la siguiente hora.

- Tipo de problema: Regresión.
- Modelo evaluado: HistGradientBoosting con pérdida Poisson.
- MAE de validación reportado: 0.7207.
- MAE de referencia reportado: 0.8911.
- Mejora aproximada respecto a la referencia: 19 %.

La predicción puede apoyar decisiones de redistribución de bicicletas entre estaciones.

### P2 — Riesgo de vaciado o llenado

**Objetivo:** Clasificar el riesgo operativo de una estación durante la siguiente hora.

- Tipo de problema: Clasificación.
- Modelo evaluado: Random Forest con ponderación de clases.
- Balanced Accuracy reportada: 0.7014.

El modelo puede apoyar la identificación de estaciones que requieren atención operativa.

### P3 — Agrupación de estaciones

**Objetivo:** Identificar grupos de estaciones con características similares.

- Tipo de problema: Aprendizaje no supervisado.
- Algoritmo: K-means.
- Número de grupos: 3.
- Silhouette Score reportado: 0.3079.

Los grupos pueden utilizarse para analizar patrones de operación y apoyar la planificación.

## 3. Registro y versionado

Los modelos se registran en MLflow Model Registry:

- mibici-equipo04-p1-demanda
- mibici-equipo04-p2-riesgo
- mibici-equipo04-p3-grupos

Se utiliza el alias `champion` para identificar la versión seleccionada de cada modelo.

Las versiones alternativas pueden evaluarse antes de promoverlas a producción.

## 4. Evaluación A/B del modelo P1

Se evaluaron tres versiones sobre 279,312 observaciones de junio de 2026.

| Versión | MAE | RMSE |
| --- | ---: | ---: |
| Champion v1 | 0.7859 | 1.4464 |
| Challenger v2 | 0.7840 | 1.4885 |
| Challenger-hp v3 | 0.7873 | 1.3951 |

El criterio de promoción exige reducir el MAE, no empeorar el RMSE más de 1 % y obtener un intervalo de confianza del 95 % para la diferencia diaria de MAE que excluya cero.

Ninguna alternativa cumplió todos los criterios. Por ello, la decisión registrada fue conservar `champion`.

Estos resultados corresponden a la evaluación A/B y no deben confundirse con las métricas de validación inicial.

## 5. Monitoreo de datos

Se utilizó Evidently AI para comparar datos de referencia y datos actuales.

Cada análisis utilizó 50,000 registros de referencia y 50,000 registros actuales.

| Modelo | Variables con drift | Porcentaje |
| --- | ---: | ---: |
| P1 | 2 de 12 | 16.67 % |
| P2 | 1 de 18 | 5.56 % |

Las variables identificadas fueron:

- P1: `salidas_media_24h` y `es_semana_santa`.
- P2: `es_semana_santa`.

El umbral individual utilizado fue 0.1 y el umbral global configurado fue 50 % de las variables.

Ambos modelos presentan cambios en algunas variables, pero no superan el umbral global configurado.

La detección de drift debe complementarse con evaluaciones de desempeño antes de decidir un reentrenamiento.

## 6. Despliegue e inferencia

Los modelos P1 y P2 están disponibles mediante una API desarrollada con FastAPI y ejecutada en Docker.

Endpoints principales:

- `GET /health`
- `POST /predict/p1`
- `POST /predict/p2`
- `GET /monitoreo/resumen`

La API registra solicitudes, latencias y estadísticas de predicción cuando el monitoreo está habilitado.

El modelo P3 se encuentra registrado en MLflow, pero no dispone de un endpoint de predicción en la API verificada.

## 7. Limitaciones y riesgos

- El desempeño puede cambiar cuando evolucionan los patrones de movilidad.
- Los resultados dependen de la calidad y representatividad de los datos de entrada.
- Las predicciones no garantizan la disponibilidad real de bicicletas.
- La clasificación de riesgo puede producir falsos positivos y falsos negativos.
- Los grupos de K-means son descriptivos y no representan categorías operativas garantizadas.
- El drift de datos no implica automáticamente una pérdida de desempeño predictivo.
- Las métricas reportadas no garantizan resultados equivalentes en otros periodos o sistemas de bicicletas compartidas.

## 8. Uso previsto y recomendaciones

Los modelos están destinados a apoyar el análisis y la planificación operativa de MiBici.

Se recomienda:

- Mantener supervisión humana sobre las decisiones operativas.
- Monitorear periódicamente los datos y el desempeño.
- Evaluar nuevas versiones antes de modificar el alias `champion`.
- Documentar los cambios de modelos y sus resultados en MLflow.
- Evitar utilizar las predicciones como única fuente para decisiones críticas.

## 9. Evidencias del proyecto

- MLflow Tracking y Model Registry.
- GitHub Actions para integración continua.
- Docker y FastAPI para inferencia.
- Locust para pruebas de carga.
- Evidently AI para análisis de drift.
- Evaluación A/B de versiones del modelo P1.
- Análisis FinOps de escenarios de infraestructura.

## 10. Referencias internas

- `README.md`
- `reports/ab_testing/resultados.json`
- `reports/drift/drift_p1.json`
- `reports/drift/drift_p2.json`
- `reports/finops/finops.json`
- `.github/workflows/ci.yml`

**Equipo responsable:** Equipo 04 — Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey.
