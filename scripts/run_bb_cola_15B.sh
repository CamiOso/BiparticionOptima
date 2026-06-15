#!/usr/bin/env bash
# Cola secuencial BranchBound k=2..5 para la hoja 15B.
# Nota: 15B tiene n_total más grandes → los k=3,4,5 serán heurísticos para subsistemas grandes.
# Uso: bash run_bb_cola_15B.sh [fila_inicio]
set -e
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
source "$PROJECT/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

START=${1:-6}
for fila in $(seq $START 55); do
    echo "=== 15B fila $fila ==="
    python3 "$PROJECT/run_bb_single.py" 15B "$fila" || echo "  [WARN] fila $fila falló"
    sleep 0.2
done
echo "=== Cola 15B COMPLETA ==="
