#!/usr/bin/env bash
# Cola secuencial BranchBound k=2..5 para la hoja 10A.
# Uso: bash run_bb_cola_10A.sh [fila_inicio]
set -e
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
source "$PROJECT/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

START=${1:-6}
for fila in $(seq $START 54); do
    echo "=== 10A fila $fila ==="
    python3 "$PROJECT/run_bb_single.py" 10A "$fila" || echo "  [WARN] fila $fila falló"
    sleep 0.2
done
echo "=== Cola 10A COMPLETA ==="
