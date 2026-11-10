# Monitoreo de data drift (Evidently)

Reproducir: `python -m mibici_mlops_module.monitoreo` (≈15 s). Genera `drift_p1.html|json` y `drift_p2.html|json`
en esta carpeta; el `.html` se abre directo en el navegador.

## Conjuntos de datos

| Conjunto | Periodo | Filas (tabla P1/P2) | Origen |
| --- | --- | --- | --- |
| **Referencia** | 7 ene – 30 abr 2026 | 999,024 | `conjunto == "entrenamiento"`: los datos con los que se entrenaron los modelos |
| **Actual** | 1 – 30 jun 2026 | 279,312 | `conjunto == "prueba"`: el mes posterior más reciente de los datos abiertos de MiBici |

**Por qué estos periodos**

- La referencia es exactamente el periodo de entrenamiento, que es contra lo que el modelo "espera" que se parezcan
  los datos en producción.
- Como actual se eligió junio y no mayo porque (1) es el periodo más tardío disponible (los datos abiertos locales
  llegan a junio), (2) deja un mes completo de separación y (3) incluye un cambio de temporada real: inicio de lluvias
  y fin del ciclo escolar, lo que sí puede mover el uso de la bicicleta. Mayo ya se usó como validación.
- La API de la semana 8 no registra las peticiones que recibe, así que no hay tráfico real que usar; junio simula
  producción.
- Se compara el espacio de variables de entrada de cada modelo (`FEATURES_P1` y `FEATURES_P2` de `features.py`), no los
  objetivos. P3 (tipología) se excluye: es un perfil por estación sin dimensión temporal.
- Con ~1M vs. ~280k filas casi cualquier prueba estadística marca diferencias "significativas" por tamaño de muestra.
  Por eso se toma una muestra aleatoria de **50,000 filas por conjunto** (`random_state=42`).
- Calendario (`hora`, `dia_semana`, `es_*`) se trata como categórico; el resto como numérico.

## Configuración de Evidently (0.7.x)

`Report([DataDriftPreset(), DataSummaryPreset()])`: drift por variable (Wasserstein normalizado para numéricas y
Jensen-Shannon para categóricas, umbral 0.1) y calidad de datos (nulos, duplicados, columnas constantes, rangos y
estadísticos por variable).

## Resultados

| | P1 | P2 |
| --- | --- | --- |
| Variables con drift | 2 de 12 (16.7 %) | 1 de 18 (5.6 %) |
| Valores nulos | 0 | 0 |
| Filas duplicadas (en la muestra de 50k) | 327 | 1,779 |
| Columnas constantes | 2 | 2 |

Reportes completos de Evidently (interactivos): [P1](drift_p1.html) · [P2](drift_p2.html). Imágenes exportadas:

![Drift por variable, P1](drift_p1.png)

![Drift por variable, P2](drift_p2.png)

Variables con drift:

- `es_semana_santa` (JS = 0.196): efecto esperado del calendario. Existe en la referencia (30 mar – 10 abr) y no en
  junio, donde siempre vale 0. No indica un problema.
- `salidas_media_24h` en P1 (Wasserstein = 0.108, apenas sobre el umbral de 0.1): la actividad reciente cambió un poco
  entre periodos. Es la señal con más interés real.
- Cerca del umbral pero sin drift: `indice_flujo_acum_3h` (0.081) y `es_festivo` (0.080).
- El resto (ubicación, capacidad, salidas/llegadas y rezagos) quedó por debajo de 0.07.

Las 2 columnas constantes son `es_festivo` y `es_semana_santa`: en junio valen siempre 0 (no hay festivos oficiales ni
Semana Santa), cuando en la referencia sí varían. Es calendario, no un fallo de captura.

Las duplicadas son esperables (muchas estaciones-hora con valores enteros pequeños idénticos, p. ej. 0 salidas); no son
un error de datos.

## Limitaciones

- Es drift de variables de entrada; no mide degradación del desempeño del modelo (eso requiere el objetivo observado).
- El umbral 0.1 es el predeterminado de Evidently, no calibrado para MiBici.
