import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

carpeta = Path("reports/recursos")
archivo = carpeta / "recursos_20_usuarios.csv"

fechas = []
cpu = []
ram = []

with open(archivo, newline="", encoding="utf-8") as f:
    lector = csv.DictReader(f)
    for fila in lector:
        fechas.append(datetime.fromisoformat(fila["timestamp"]))
        cpu.append(float(fila["cpu_pct"]))
        ram.append(float(fila["mem_mib"]))

if not fechas:
    raise ValueError("El archivo CSV no contiene mediciones.")

inicio_locust = datetime.fromisoformat("2026-11-17T20:38:23")

def generar_grafica(valores, titulo, ylabel, nombre):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(fechas, valores, marker="o", linewidth=2)
    ax.axvline(
        inicio_locust,
        color="red",
        linestyle="--",
        label="Inicio de Locust"
    )

    ax.set_title(titulo)
    ax.set_xlabel("Hora")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    fig.autofmt_xdate()
    fig.tight_layout()

    destino = carpeta / nombre
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfica generada: {destino}")

generar_grafica(
    cpu,
    "MiBici - Consumo de CPU durante prueba de carga",
    "CPU (%)",
    "cpu_20_usuarios.png"
)

generar_grafica(
    ram,
    "MiBici - Consumo de RAM durante prueba de carga",
    "Memoria RAM (MiB)",
    "ram_20_usuarios.png"
)

print(f"Mediciones procesadas: {len(fechas)}")
print(f"CPU máxima: {max(cpu):.2f}%")
print(f"RAM máxima: {max(ram):.2f} MiB")
