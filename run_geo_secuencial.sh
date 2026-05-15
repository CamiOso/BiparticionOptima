#!/bin/bash
# Ejecuta Geometric para filas pendientes de 20A, una a la vez
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

FILAS=(55 47 53 45 52)

for fila in "${FILAS[@]}"; do
    echo "========================================"
    echo "[GEO-SEQ] Iniciando fila $fila — $(date)"
    echo "========================================"
    python3 -u run_geo_single.py "$fila"
    echo "[GEO-SEQ] Fila $fila terminada — $(date)"
    echo ""
done

echo "[GEO-SEQ] TODAS LAS FILAS COMPLETADAS — $(date)"
