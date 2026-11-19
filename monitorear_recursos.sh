#!/usr/bin/env bash
# Uso: ./monitorear_recursos.sh [contenedor]   (por defecto el de docker compose)
CONTENEDOR="${1:-mibici-api-service}"

mkdir -p reports/recursos

ARCHIVO="reports/recursos/recursos_20_usuarios.csv"

echo "timestamp,cpu_pct,mem_mib,mem_pct" > "$ARCHIVO"

for i in $(seq 1 24); do
    docker stats "$CONTENEDOR" --no-stream \
      --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" |
    awk -F',' -v fecha="$(date +%Y-%m-%dT%H:%M:%S)" '{
        cpu=$1
        gsub(/%/, "", cpu)

        split($2, memoria, " / ")
        ram=memoria[1]

        if (ram ~ /GiB/) {
            gsub(/GiB/, "", ram)
            ram=ram*1024
        } else {
            gsub(/MiB/, "", ram)
        }

        mem=$3
        gsub(/%/, "", mem)

        print fecha "," cpu "," ram "," mem
    }' >> "$ARCHIVO"

    sleep 5
done

echo "Monitoreo finalizado: $ARCHIVO"
