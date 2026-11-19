"""Gráficas de la prueba de carga (Locust) y cálculo de costo por inferencia (FinOps).

Uso:  python analisis_finops.py
Lee reports/locust/prueba_20_usuarios_stats*.csv y reports/recursos/recursos_20_usuarios.csv.
Escribe reports/locust/*.png y reports/finops/finops.json.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOC = Path("reports/locust")
PREFIJO = "prueba_20_usuarios"
SALIDA_FIN = Path("reports/finops")

# Precios de referencia (AWS us-east-1, Linux on-demand, USD/hora; verificar antes de decidir).
INSTANCIAS = {
    "t3.small (2 vCPU, 2 GiB)": 0.0208,
    "t3.medium (2 vCPU, 4 GiB)": 0.0416,
    "t3.large (2 vCPU, 8 GiB)": 0.0832,
}
SPOT_DESCUENTO = 0.65  # ahorro típico de Spot
RESERVADA_DESCUENTO = 0.40  # ahorro típico de 1 año sin pago anticipado
HORAS_MES = 730
# docker stats reporta CPU como % de un núcleo (100 % = 1 vCPU)


def leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def graficas():
    hist = [r for r in leer(LOC / f"{PREFIJO}_stats_history.csv") if r["Name"] == "Aggregated"]
    t0 = int(hist[0]["Timestamp"])
    t = [int(r["Timestamp"]) - t0 for r in hist]
    usuarios = [int(r["User Count"]) for r in hist]
    rps = [num(r["Requests/s"]) for r in hist]
    fps = [num(r["Failures/s"]) for r in hist]
    p50 = [num(r["50%"]) for r in hist]
    p95 = [num(r["95%"]) for r in hist]
    p99 = [num(r["99%"]) for r in hist]
    tot = [num(r["Total Request Count"]) for r in hist]
    fal = [num(r["Total Failure Count"]) for r in hist]
    err = [100 * f / n if n else 0 for f, n in zip(fal, tot)]

    def serie(ax, ys, etiquetas, titulo, ylabel):
        for y, e in zip(ys, etiquetas):
            puntos = [(a, b) for a, b in zip(t, y) if b is not None]
            ax.plot([a for a, _ in puntos], [b for _, b in puntos], label=e, linewidth=2)
        ax.set_title(titulo)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    serie(ax[0, 0], [p50, p95, p99], ["p50", "p95", "p99"], "Latencia agregada", "ms")
    serie(ax[0, 1], [rps], ["Peticiones/s"], "Peticiones por segundo", "req/s")
    eje = ax[0, 1].twinx()
    eje.plot(t, usuarios, color="gray", linestyle="--")
    eje.set_ylabel("Usuarios virtuales")
    serie(ax[1, 0], [err], ["% de fallos acumulado"], "Tasa de error", "%")
    ax[1, 0].set_ylim(0, 5)
    ax[1, 1].axis("off")
    est = [r for r in leer(LOC / f"{PREFIJO}_stats.csv")]
    filas = [[r["Name"], r["Request Count"], r["Failure Count"],
              f'{float(r["Average Response Time"]):.0f}', r["95%"], r["99%"]] for r in est]
    tabla = ax[1, 1].table(cellText=filas, colLabels=["Endpoint", "Peticiones", "Fallos", "Media ms", "p95", "p99"],
                           loc="center")
    tabla.scale(1, 2)
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    fig.suptitle("Prueba de carga Locust: 20 usuarios, 2 minutos")
    fig.tight_layout()
    fig.savefig(LOC / f"{PREFIJO}_graficas.png", dpi=150)
    plt.close(fig)

    # Latencia por endpoint (percentiles finales)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["50%", "90%", "95%", "99%"]
    ancho = 0.35
    for i, r in enumerate(r for r in est if r["Name"] != "Aggregated"):
        ax.bar([j + i * ancho for j in range(4)], [float(r[c]) for c in cols], ancho, label=r["Name"])
    ax.set_xticks([j + ancho / 2 for j in range(4)])
    ax.set_xticklabels(cols)
    ax.set_ylabel("ms")
    ax.set_title("Percentiles de latencia por endpoint")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(LOC / f"{PREFIJO}_latencia_endpoints.png", dpi=150)
    plt.close(fig)
    return est


def finops(est):
    agg = next(r for r in est if r["Name"] == "Aggregated")
    rps = float(agg["Requests/s"])
    total = int(agg["Request Count"])
    rec = leer(Path("reports/recursos/recursos_20_usuarios.csv"))
    cpu = [float(r["cpu_pct"]) for r in rec]
    mem = [float(r["mem_mib"]) for r in rec]
    # Solo muestras con carga (descarta las de reposo ~0.2 %)
    carga = [c for c in cpu if c > 5]
    cpu_carga = sum(carga) / len(carga)
    res = {
        "peticiones_totales": total,
        "rps_prueba": round(rps, 2),
        "latencia_media_ms": round(float(agg["Average Response Time"]), 1),
        "p95_ms": float(agg["95%"]),
        "cpu_media_carga_pct_de_1_vcpu": round(cpu_carga, 1),
        "cpu_pico_pct": max(cpu),
        "mem_media_mib": round(sum(mem) / len(mem), 1),
        "mem_pico_mib": max(mem),
        "supuestos": {"horas_mes": HORAS_MES, "spot_descuento": SPOT_DESCUENTO,
                      "reservada_descuento": RESERVADA_DESCUENTO},
        "escenarios": [],
    }
    # Capacidad: con ~cpu_carga % de 1 vCPU a `rps` req/s, 1 vCPU sostiene aprox. rps*100/cpu_carga req/s
    # (extrapolación lineal; se usa un 70 % de utilización objetivo para dejar margen).
    cap_vcpu = rps * 100 / cpu_carga
    for nombre, precio in INSTANCIAS.items():
        vcpus = 2
        cap = cap_vcpu * vcpus * 0.7
        for etiqueta, factor in (("on-demand", 1), ("reservada 1 año", 1 - RESERVADA_DESCUENTO),
                                 ("spot", 1 - SPOT_DESCUENTO)):
            p = precio * factor
            # a) costo por millón de inferencias con la instancia al 100 % de capacidad útil
            c_cap = p / (cap * 3600) * 1_000_000
            # b) costo con la carga de la prueba (utilización real, 9 req/s en una instancia dedicada)
            c_real = p / (rps * 3600) * 1_000_000
            res["escenarios"].append({
                "instancia": nombre, "modalidad": etiqueta, "usd_hora": round(p, 4),
                "usd_mes": round(p * HORAS_MES, 2),
                "capacidad_req_s": round(cap),
                "usd_por_millon_a_capacidad": round(c_cap, 2),
                "usd_por_millon_a_carga_de_prueba": round(c_real, 2),
            })
    res["capacidad_1_vcpu_req_s"] = round(cap_vcpu)
    SALIDA_FIN.mkdir(parents=True, exist_ok=True)
    (SALIDA_FIN / "finops.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # Gráfica: costo por millón de inferencias según utilización
    fig, ax = plt.subplots(figsize=(8, 4.5))
    base = INSTANCIAS["t3.medium (2 vCPU, 4 GiB)"]
    cap = cap_vcpu * 2 * 0.7
    xs = [i for i in range(1, 101)]
    for etiqueta, factor in (("on-demand", 1), ("reservada", 1 - RESERVADA_DESCUENTO), ("spot", 1 - SPOT_DESCUENTO)):
        ys = [base * factor / (cap * u / 100 * 3600) * 1_000_000 for u in xs]
        ax.plot(xs, ys, label=f"t3.medium {etiqueta}", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Utilización de la capacidad (%)")
    ax.set_ylabel("USD por millón de inferencias (escala log)")
    ax.set_title("Costo por inferencia vs. utilización")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA_FIN / "costo_vs_utilizacion.png", dpi=150)
    plt.close(fig)
    return res


if __name__ == "__main__":
    est = graficas()
    r = finops(est)
    print(json.dumps(r, indent=2, ensure_ascii=False))
